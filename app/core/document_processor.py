import cv2
import numpy as np
import pytesseract
import asyncio
import json
import logging
from typing import Dict, List, Optional, Any  # Make sure all types are imported
from datetime import datetime
from enum import Enum
import redis.asyncio as redis
import re
from dataclasses import dataclass
import uuid

from ..utils.file_handlers import FileHandler
from ..utils.template_manager import TemplateManager
from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class DocumentType(Enum):
    KRA_PIN = "kra_pin"
    BUSINESS_CERT = "business_cert"
    KENYAN_ID = "kenyan_id"
    PASSPORT = "passport"
    DRIVERS_LICENSE = "drivers_license"
    UNKNOWN = "unknown"

@dataclass
class ClassificationResult:
    document_type: DocumentType
    confidence: float
    extracted_features: Dict[str, Any]

@dataclass
class ExtractionResult:
    success: bool
    data: Dict[str, Any]
    confidence: float
    errors: List[str]

class DocumentProcessor:
    def __init__(self):
        self.file_handler = FileHandler()
        self.template_manager = TemplateManager()
        self.redis_client = None
        self.processing_jobs = {}
        
    async def initialize(self):
        """Initialize processor components"""
        try:
            if settings.REDIS_URL:
                # Railway provides REDIS_URL
                self.redis_client = redis.from_url(
                    settings.REDIS_URL,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
            else:
                # Fallback to individual parameters
                self.redis_client = redis.Redis(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    db=settings.REDIS_DB,
                    password=settings.REDIS_PASSWORD if settings.REDIS_PASSWORD else None,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
            
            await self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Running without caching.")
            self.redis_client = None
        
        # Configure Tesseract
        if settings.TESSERACT_PATH:
            pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_PATH
            
        # Load templates
        self.template_manager.load_templates()
        
    async def cleanup(self):
        """Cleanup resources"""
        if self.redis_client:
            await self.redis_client.close()
    
    def is_supported_file(self, filename: str, content_type: str) -> bool:
        """Check if file type is supported"""
        return (content_type in settings.SUPPORTED_IMAGE_TYPES or 
                content_type in settings.SUPPORTED_PDF_TYPES or
                self.file_handler.is_supported_extension(filename))
    
    async def process_document_sync(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Process document synchronously"""
        try:
            # Convert file to images
            images = await self.file_handler.convert_to_images(file_content, filename)
            
            best_result = None
            best_confidence = 0
            
            # Process each page/image
            for i, image in enumerate(images):
                # Classify document
                classification = self.classify_document(image)
                
                if classification.document_type != DocumentType.UNKNOWN:
                    # Extract data using template
                    extraction_result = self.extract_data_with_template(
                        image, classification.document_type
                    )
                    
                    overall_confidence = (classification.confidence + extraction_result.confidence) / 2
                    
                    if overall_confidence > best_confidence:
                        best_confidence = overall_confidence
                        best_result = {
                            "document_type": classification.document_type.value,
                            "classification_confidence": classification.confidence,
                            "extraction_success": extraction_result.success,
                            "extracted_data": extraction_result.data,
                            "extraction_confidence": extraction_result.confidence,
                            "extraction_errors": extraction_result.errors,
                            "overall_confidence": overall_confidence,
                            "page_number": i + 1
                        }
            
            if best_result:
                return best_result
            else:
                return {
                    "document_type": "unknown",
                    "classification_confidence": 0,
                    "extraction_success": False,
                    "extracted_data": {},
                    "extraction_errors": ["Document type could not be determined"],
                    "overall_confidence": 0
                }
                
        except Exception as e:
            logger.error(f"Document processing error: {str(e)}")
            raise
    
    async def process_document_async(self, document_id: str, file_content: bytes, 
                                   filename: str, webhook_url: Optional[str] = None):
        """Process document asynchronously"""
        try:
            await self.update_job_status(document_id, "processing", 10)
            
            result = await self.process_document_sync(file_content, filename)
            
            await self.update_job_status(document_id, "completed", 100, result)
            
            if webhook_url:
                await self.send_webhook(webhook_url, document_id, result)
                
        except Exception as e:
            logger.error(f"Error processing document {document_id}: {str(e)}")
            error_result = {
                "error": str(e),
                "document_type": "unknown",
                "extraction_success": False
            }
            await self.update_job_status(document_id, "failed", 100, error_result)
    
    def classify_document(self, image: np.ndarray) -> ClassificationResult:
        """Classify document type"""
        # Enhanced preprocessing
        processed_image = self.preprocess_image(image)
        
        # Extract text
        text = self.extract_text_features(processed_image)
        
        # Classification rules for each document type
        classification_rules = {
            DocumentType.KRA_PIN: {
                'patterns': [
                    r'PIN Certificate',
                    r'Kenya Revenue Authority',
                    r'A\d{9}[A-Z]',
                    r'Tax Obligation'
                ],
                'weight': 1.0
            },
            DocumentType.KENYAN_ID: {
                'patterns': [
                    r'JAMHURI YA KENYA',
                    r'REPUBLIC OF KENYA',
                    r'\d{8}',
                    r'FULL NAME'
                ],
                'weight': 1.0
            },
            DocumentType.BUSINESS_CERT: {
                'patterns': [
                    r'Business Registration',
                    r'Certificate of Registration',
                    r'Registrar of Companies'
                ],
                'weight': 1.0
            }
        }
        
        scores = {}
        for doc_type, rules in classification_rules.items():
            score = 0
            for pattern in rules['patterns']:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                score += matches * 20
            
            scores[doc_type] = score * rules['weight']
        
        # Find best match
        if scores:
            best_type = max(scores, key=scores.get)
            confidence = min(scores[best_type], 100)
            
            if confidence < 30:
                best_type = DocumentType.UNKNOWN
                confidence = 0
        else:
            best_type = DocumentType.UNKNOWN
            confidence = 0
        
        return ClassificationResult(
            document_type=best_type,
            confidence=confidence,
            extracted_features={'text': text, 'scores': scores}
        )
    
    def extract_data_with_template(self, image: np.ndarray, doc_type: DocumentType) -> ExtractionResult:
        """Extract data using document template"""
        template = self.template_manager.get_template(doc_type.value)
        if not template:
            return ExtractionResult(
                success=False,
                data={},
                confidence=0,
                errors=[f"No template found for {doc_type.value}"]
            )
        
        extracted_data = {}
        errors = []
        confidence_scores = []
        
        h, w = image.shape[:2]
        
        for field in template.fields:
            try:
                # Extract region
                x1 = int(field.region.x1 * w)
                y1 = int(field.region.y1 * h)
                x2 = int(field.region.x2 * w)
                y2 = int(field.region.y2 * h)
                
                region = image[y1:y2, x1:x2]
                
                if region.size == 0:
                    errors.append(f"Empty region for field: {field.name}")
                    continue
                
                # Preprocess region if specified
                if field.preprocessing:
                    region = self.preprocess_region(region, field.preprocessing)
                
                # Extract text
                text = pytesseract.image_to_string(region, config=field.ocr_config).strip()
                
                # Post-process text
                if field.post_processing:
                    text = self.post_process_text(text, field.post_processing)
                
                # Validate
                if field.validation_pattern:
                    if re.match(field.validation_pattern, text):
                        extracted_data[field.name] = text
                        confidence_scores.append(80)
                    else:
                        errors.append(f"Validation failed for {field.name}: {text}")
                        extracted_data[field.name] = text
                        confidence_scores.append(30)
                else:
                    extracted_data[field.name] = text
                    confidence_scores.append(70)
                    
            except Exception as e:
                errors.append(f"Error extracting {field.name}: {str(e)}")
                confidence_scores.append(0)
        
        overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        return ExtractionResult(
            success=len(errors) == 0,
            data=extracted_data,
            confidence=overall_confidence,
            errors=errors
        )
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Noise reduction
        denoised = cv2.medianBlur(gray, 3)
        
        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        return enhanced
    
    def preprocess_region(self, region: np.ndarray, method: str) -> np.ndarray:
        """Preprocess specific region"""
        if method == 'enhance':
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            return clahe.apply(region)
        elif method == 'invert':
            return cv2.bitwise_not(region)
        elif method == 'table_cell':
            kernel = np.ones((2,2), np.uint8)
            cleaned = cv2.morphologyEx(region, cv2.MORPH_CLOSE, kernel)
            return cv2.threshold(cleaned, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return region
    
    def extract_text_features(self, image: np.ndarray) -> str:
        """Extract text using OCR"""
        configs = [
            '--psm 6',
            '--psm 4',
            '--psm 3'
        ]
        
        best_text = ""
        for config in configs:
            try:
                text = pytesseract.image_to_string(image, config=config)
                if len(text) > len(best_text):
                    best_text = text
            except Exception:
                continue
        
        return best_text.strip()
    
    def post_process_text(self, text: str, method: str) -> str:
        """Post-process extracted text"""
        if method == 'clean_name':
            text = re.sub(r'[^A-Za-z\s]', '', text)
            return ' '.join(text.upper().split())
        elif method == 'extract_email':
            email_match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text.lower())
            return email_match.group(0) if email_match else text.lower()
        return text
    
    async def update_job_status(self, document_id: str, status: str, progress: int, 
                               result: Optional[Dict] = None):
        """Update job status"""
        job_data = {
            "document_id": document_id,
            "status": status,
            "progress": progress,
            "result": result,
            "updated_at": datetime.now().isoformat()
        }
        
        if self.redis_client:
            try:
                await self.redis_client.setex(f"job:{document_id}", 3600, json.dumps(job_data))
            except Exception as e:
                logger.error(f"Redis update failed: {e}")
                self.processing_jobs[document_id] = job_data
        else:
            self.processing_jobs[document_id] = job_data
    
    async def get_job_status(self, document_id: str) -> Optional[Dict]:
        """Get job status"""
        if self.redis_client:
            try:
                data = await self.redis_client.get(f"job:{document_id}")
                return json.loads(data) if data else None
            except Exception:
                return self.processing_jobs.get(document_id)
        else:
            return self.processing_jobs.get(document_id)
    
    async def cleanup_document(self, document_id: str) -> bool:
        """Clean up document data"""
        if self.redis_client:
            try:
                deleted = await self.redis_client.delete(f"job:{document_id}")
                return deleted > 0
            except Exception:
                return document_id in self.processing_jobs and bool(self.processing_jobs.pop(document_id, None))
        else:
            return document_id in self.processing_jobs and bool(self.processing_jobs.pop(document_id, None))
    
    async def send_webhook(self, webhook_url: str, document_id: str, result: Dict):
        """Send webhook notification"""
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "document_id": document_id,
                    "status": "completed",
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                }
                async with session.post(webhook_url, json=payload, timeout=30) as response:
                    if response.status == 200:
                        logger.info(f"Webhook sent successfully for {document_id}")
        except Exception as e:
            logger.error(f"Webhook error for {document_id}: {str(e)}")
    
    def get_supported_documents(self) -> List[Dict]:
        """Get supported document types"""
        return [
            {"type": "kra_pin", "name": "KRA PIN Certificate", "description": "Kenya Revenue Authority PIN Certificate"},
            {"type": "kenyan_id", "name": "Kenyan National ID", "description": "Republic of Kenya National Identity Card"},
            {"type": "business_cert", "name": "Business Registration Certificate", "description": "Certificate of Business Registration"},
        ]
    
    async def health_check(self) -> Dict:
        """Health check"""
        try:
            test_image = np.ones((100, 100), dtype=np.uint8) * 255
            pytesseract.image_to_string(test_image)
            ocr_status = "ok"
        except Exception as e:
            ocr_status = f"error: {str(e)}"
        
        redis_status = "ok" if self.redis_client else "not_configured"
        if self.redis_client:
            try:
                await self.redis_client.ping()
            except Exception:
                redis_status = "error"
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "ocr": ocr_status,
                "redis": redis_status,
                "templates": len(self.template_manager.templates)
            }
        }