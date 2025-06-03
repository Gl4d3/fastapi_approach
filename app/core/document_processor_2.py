import cv2
import numpy as np
import pytesseract
import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import re
from dataclasses import dataclass
import uuid
import traceback
import io
from PIL import Image
import fitz

# Conditional imports for Railway
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from ..utils.file_handlers import FileHandler
from ..utils.template_manager import TemplateManager
from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class DocumentType(Enum):
    KRA_PIN = "kra_pin"
    BUSINESS_CERT = "business_cert"
    KENYAN_ID = "kenyan_id"
    FOREIGN_ID = "foreign_id"
    PASSPORT = "passport"
    DRIVERS_LICENSE = "drivers_license"
    UNKNOWN = "unknown"

@dataclass
class OCRResult:
    text: str
    confidence: float
    bbox: List[float]

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
        
        # Enhanced exclusion words
        self.exclude_words = {
            'REPUBLIC', 'KENYA', 'NATIONAL', 'IDENTITY', 'CARD', 'SPECIMEN', 'FOREIGNER',
            'CERTIFICATE', 'SERIAL', 'NUMBER', 'CITY', 'HOLDER', 'SIGNATURE', 'PLACE', 
            'DATE', 'ISSUE', 'BIRTH', 'FOREIGN', 'PERMIT', 'VISA', 'TEMPORARY', 'PERMANENT',
            'ALIEN', 'RESIDENT', 'CITIZEN', 'KENYAN', 'PIN', 'AUTHORITY', 'REVENUE',
            'BUSINESS', 'REGISTRATION', 'REGISTRAR', 'COMPANY'
        }
        
    async def initialize(self):
        """Initialize processor components"""
        try:
            # Redis setup with proper error handling
            if REDIS_AVAILABLE and settings.REDIS_URL:
                try:
                    self.redis_client = redis.from_url(
                        settings.REDIS_URL,
                        decode_responses=True,
                        socket_connect_timeout=5,
                        socket_timeout=5
                    )
                    await self.redis_client.ping()
                    logger.info("Redis connection established")
                except Exception as e:
                    logger.warning(f"Redis connection failed: {e}. Running without Redis.")
                    self.redis_client = None
            else:
                logger.info("Redis not configured, running without caching")
        
            # Configure Tesseract with debugging
            tesseract_path = settings.TESSERACT_PATH
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            
            # Test Tesseract
            try:
                test_image = np.ones((100, 100), dtype=np.uint8) * 255
                test_result = pytesseract.image_to_string(test_image)
                logger.info(f"Tesseract test successful at {tesseract_path}")
            except Exception as e:
                logger.error(f"Tesseract test failed: {e}")
                # Try alternative paths
                alternative_paths = ['/usr/bin/tesseract', '/usr/local/bin/tesseract', 'tesseract']
                for path in alternative_paths:
                    try:
                        pytesseract.pytesseract.tesseract_cmd = path
                        pytesseract.image_to_string(test_image)
                        logger.info(f"Tesseract working with path: {path}")
                        break
                    except:
                        continue
                else:
                    logger.error("No working Tesseract installation found!")
            
            # Load templates
            self.template_manager.load_templates()
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            logger.error(f"Initialization traceback: {traceback.format_exc()}")
            
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
        """Enhanced document processing with detailed debugging"""
        try:
            logger.info(f"Starting processing for file: {filename}")
            
            # Convert file to images
            logger.info("Converting to OpenCV image...")
            images = await self.file_handler.convert_to_images(file_content, filename)
            logger.info(f"Image converted. Shape: {images[0].shape if images else 'No images'}")
            
            if not images:
                return self._create_error_result("No images could be extracted from file")
            
            # Process first image (main page)
            image = images[0]
            
            # Enhanced OCR extraction with debugging
            logger.info("Starting OCR extraction...")
            ocr_results = self._enhanced_ocr_extraction_with_debugging(image)
            logger.info(f"OCR completed. Found {len(ocr_results)} text elements")
            
            # Log some OCR results for debugging
            if ocr_results:
                sample_texts = [r.text for r in ocr_results[:5]]
                logger.info(f"Sample OCR texts: {sample_texts}")
            else:
                logger.warning("No OCR results found!")
                # Try different preprocessing
                logger.info("Trying enhanced preprocessing...")
                enhanced_image = self._enhanced_preprocessing(image)
                ocr_results = self._fallback_ocr_extraction(enhanced_image)
                logger.info(f"Fallback OCR found {len(ocr_results)} text elements")
            
            # Auto-detect document type
            logger.info("Starting document classification...")
            classification = self._auto_detect_document_type(ocr_results)
            logger.info(f"Classification: {classification.document_type.value}, confidence: {classification.confidence}")
            
            # Enhanced field extraction
            logger.info("Starting field extraction...")
            extraction = self._enhanced_field_extraction(ocr_results, classification.document_type)
            logger.info(f"Extraction completed. Success: {extraction.success}, data: {extraction.data}")
            
            overall_confidence = (classification.confidence + extraction.confidence) / 2
            
            result = {
                "document_type": classification.document_type.value,
                "classification_confidence": classification.confidence,
                "extraction_success": extraction.success,
                "extracted_data": extraction.data,
                "extraction_confidence": extraction.confidence,
                "extraction_errors": extraction.errors,
                "overall_confidence": overall_confidence,
                "ocr_text_sample": classification.extracted_features.get('text_sample', ''),
                "debug_info": {
                    "ocr_count": len(ocr_results),
                    "image_shape": list(image.shape) if image is not None else None,
                    "tesseract_path": pytesseract.pytesseract.tesseract_cmd
                }
            }
            
            logger.info(f"Processing completed successfully: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            logger.error(f"Processing traceback: {traceback.format_exc()}")
            return self._create_error_result(f"Processing failed: {str(e)}")
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result"""
        return {
            "document_type": "unknown",
            "classification_confidence": 0,
            "extraction_success": False,
            "extracted_data": {},
            "extraction_confidence": 0,
            "extraction_errors": [error_message],
            "overall_confidence": 0,
            "ocr_text_sample": "",
            "debug_info": {"error": error_message}
        }
    
    def _enhanced_ocr_extraction_with_debugging(self, image: np.ndarray) -> List[OCRResult]:
        """Enhanced OCR with comprehensive debugging"""
        try:
            # Convert to grayscale with debugging
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                logger.info(f"Converted to grayscale. Shape: {gray.shape}")
            else:
                gray = image.copy()
                logger.info(f"Image already grayscale. Shape: {gray.shape}")
            
            # Check image properties
            logger.info(f"Image stats - Min: {gray.min()}, Max: {gray.max()}, Mean: {gray.mean():.2f}")
            
            # Enhanced preprocessing with debugging
            logger.info("Applying preprocessing...")
            
            # 1. Noise reduction
            denoised = cv2.medianBlur(gray, 3)
            logger.info("Applied median blur")
            
            # 2. Contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            logger.info("Applied CLAHE enhancement")
            
            # 3. Try different OCR configurations
            configs = [
                '--psm 6',  # Uniform block of text
                '--psm 4',  # Single column
                '--psm 3',  # Fully automatic
                '--psm 1',  # Automatic with OSD
                '--psm 11', # Sparse text
                '--psm 12', # Sparse text with OSD
            ]
            
            best_results = []
            best_text_length = 0
            
            for config in configs:
                try:
                    logger.info(f"Trying OCR config: {config}")
                    
                    # Get detailed OCR data
                    ocr_data = pytesseract.image_to_data(enhanced, config=config, output_type=pytesseract.Output.DICT)
                    
                    if not ocr_data or 'text' not in ocr_data:
                        logger.warning(f"No OCR data for config {config}")
                        continue
                    
                    # Process OCR results
                    current_results = []
                    h, w = enhanced.shape
                    
                    for i in range(len(ocr_data['text'])):
                        text = ocr_data['text'][i].strip()
                        if not text:
                            continue
                        
                        conf = float(ocr_data['conf'][i])
                        if conf < 10:  # Very low threshold for debugging
                            continue
                        
                        try:
                            # Get bounding box
                            left = ocr_data['left'][i]
                            top = ocr_data['top'][i]
                            width = ocr_data['width'][i]
                            height = ocr_data['height'][i]
                            
                            # Normalize coordinates
                            x = left / w if w > 0 else 0
                            y = top / h if h > 0 else 0
                            x2 = (left + width) / w if w > 0 else 1
                            y2 = (top + height) / h if h > 0 else 1
                            
                            current_results.append(OCRResult(
                                text=text,
                                confidence=conf / 100.0,
                                bbox=[x, y, x2, y2]
                            ))
                            
                        except (KeyError, IndexError, ZeroDivisionError) as e:
                            logger.warning(f"Skipping OCR result {i} due to bbox error: {e}")
                            continue
                    
                    total_text_length = sum(len(r.text) for r in current_results)
                    logger.info(f"Config {config}: {len(current_results)} results, {total_text_length} chars")
                    
                    if total_text_length > best_text_length:
                        best_text_length = total_text_length
                        best_results = current_results
                        
                except Exception as e:
                    logger.warning(f"OCR config {config} failed: {e}")
                    continue
            
            logger.info(f"Best OCR results: {len(best_results)} elements, {best_text_length} chars total")
            
            # If still no results, try fallback
            if not best_results:
                logger.warning("All OCR configs failed, trying simple string extraction")
                try:
                    simple_text = pytesseract.image_to_string(enhanced, config='--psm 6')
                    if simple_text.strip():
                        logger.info(f"Simple OCR found text: {simple_text[:100]}...")
                        return [OCRResult(
                            text=simple_text.strip(),
                            confidence=0.5,
                            bbox=[0.0, 0.0, 1.0, 1.0]
                        )]
                except Exception as e:
                    logger.error(f"Even simple OCR failed: {e}")
            
            return best_results
            
        except Exception as e:
            logger.error(f"OCR extraction error: {e}")
            logger.error(f"OCR extraction traceback: {traceback.format_exc()}")
            return []
    
    def _enhanced_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """Apply enhanced preprocessing for difficult images"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Multiple preprocessing attempts
            preprocessed_images = []
            
            # 1. Basic enhancement
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            preprocessed_images.append(enhanced)
            
            # 2. Threshold variations
            _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            preprocessed_images.append(thresh1)
            
            # 3. Morphological operations
            kernel = np.ones((2,2), np.uint8)
            morph = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
            preprocessed_images.append(morph)
            
            # Return the enhanced version for now
            return enhanced
            
        except Exception as e:
            logger.error(f"Enhanced preprocessing error: {e}")
            return image
    
    def _fallback_ocr_extraction(self, image: np.ndarray) -> List[OCRResult]:
        """Fallback OCR extraction with very basic approach"""
        try:
            # Just try to get any text
            text = pytesseract.image_to_string(image, config='--psm 6')
            if text.strip():
                return [OCRResult(
                    text=text.strip(),
                    confidence=0.5,
                    bbox=[0.0, 0.0, 1.0, 1.0]
                )]
            return []
        except Exception as e:
            logger.error(f"Fallback OCR failed: {e}")
            return []
    
    def _auto_detect_document_type(self, ocr_results: List[OCRResult]) -> ClassificationResult:
        """Enhanced auto-detect with better foreign document detection"""
        try:
            # Combine all text for analysis
            text_content = ' '.join([r.text.upper() for r in ocr_results]).strip()
            
            # Enhanced document type indicators
            foreign_indicators = [
                'FOREIGNER', 'FOREIGN', 'ALIEN', 'RESIDENT', 'PERMIT', 'VISA', 
                'TEMPORARY', 'PERMANENT', 'PASSPORT', 'MOLDOVA', 'REPUBLICA',
                'VALABIL', 'PASSEPORT', 'VALID FOR ALL COUNTRIES', 'CERTIFICAT'
            ]
            
            national_indicators = [
                'JAMHURI YA KENYA', 'REPUBLIC OF KENYA', 'NATIONAL IDENTITY',
                'HUDUMA NAMBA', 'KENYAN NATIONAL', 'KITAMBULISHO'
            ]
            
            kra_indicators = [
                'PIN CERTIFICATE', 'KENYA REVENUE AUTHORITY', 'TAX OBLIGATION',
                'KRA', 'TAXPAYER', 'TAX PIN'
            ]
            
            business_indicators = [
                'BUSINESS REGISTRATION', 'CERTIFICATE OF INCORPORATION',
                'REGISTRAR OF COMPANIES', 'LIMITED COMPANY', 'BUSINESS PERMIT'
            ]
            
            # Score with weighted importance
            foreign_score = 0
            national_score = 0
            kra_score = 0
            business_score = 0
            
            # Count indicators with weights
            for indicator in foreign_indicators:
                if indicator in text_content:
                    if indicator in ['PASSPORT', 'MOLDOVA', 'REPUBLICA', 'VALID FOR ALL COUNTRIES']:
                        foreign_score += 3
                    else:
                        foreign_score += 1
            
            for indicator in national_indicators:
                if indicator in text_content:
                    national_score += 2
            
            for indicator in kra_indicators:
                if indicator in text_content:
                    kra_score += 2
            
            for indicator in business_indicators:
                if indicator in text_content:
                    business_score += 1
            
            # Pattern-based scoring
            if re.search(r'A\d{9}[A-Z]', text_content):
                kra_score += 4
            if re.search(r'\d{8}', text_content) and 'KENYA' in text_content:
                national_score += 3
            if re.search(r'[A-Z]{2}\d{7}', text_content):  # Passport pattern
                foreign_score += 3
            
            logger.info(f"Enhanced scores - KRA: {kra_score}, National: {national_score}, Foreign: {foreign_score}, Business: {business_score}")
            
            # Determine document type
            scores = {
                DocumentType.KRA_PIN: kra_score,
                DocumentType.KENYAN_ID: national_score,
                DocumentType.FOREIGN_ID: foreign_score,
                DocumentType.BUSINESS_CERT: business_score
            }
            
            best_type = max(scores, key=scores.get)
            best_score = scores[best_type]
            
            if best_score >= 1:  # Lower threshold for debugging
                confidence = min(90, best_score * 15)
                return ClassificationResult(
                    document_type=best_type,
                    confidence=confidence,
                    extracted_features={
                        'text_sample': text_content[:300],
                        'scores': scores,
                        'detected_languages': self._detect_languages(text_content)
                    }
                )
            else:
                return ClassificationResult(
                    document_type=DocumentType.UNKNOWN,
                    confidence=0,
                    extracted_features={
                        'text_sample': text_content[:300],
                        'reason': 'No strong document type indicators found',
                        'scores': scores
                    }
                )
                
        except Exception as e:
            logger.error(f"Auto-detection error: {e}")
            return ClassificationResult(DocumentType.UNKNOWN, 0, {"error": str(e)})
    
    def _detect_languages(self, text: str) -> List[str]:
        """Simple language detection"""
        languages = []
        
        # Romanian/Moldovan indicators
        if any(word in text for word in ['REPUBLICA', 'MOLDOVA', 'ESTE', 'VALABIL', 'TOATE']):
            languages.append('romanian')
        
        # French indicators
        if any(word in text for word in ['PASSEPORT', 'REPUBLIQUE', 'POUR', 'TOUS', 'LES', 'PAYS']):
            languages.append('french')
        
        # English indicators
        if any(word in text for word in ['PASSPORT', 'VALID', 'ALL', 'COUNTRIES', 'REPUBLIC']):
            languages.append('english')
        
        # Swahili/Kenyan indicators
        if any(word in text for word in ['JAMHURI', 'YA', 'KENYA', 'KITAMBULISHO']):
            languages.append('swahili')
        
        return languages
    
    def _enhanced_field_extraction(self, ocr_results: List[OCRResult], doc_type: DocumentType) -> ExtractionResult:
        """Enhanced field extraction with spatial analysis"""
        try:
            extracted_data = {}
            errors = []
            confidence = 0
            
            if not ocr_results:
                return ExtractionResult(False, {}, 0, ["No OCR results to extract from"])
            
            # Extract fields based on document type
            if doc_type == DocumentType.KRA_PIN:
                confidence = self._extract_kra_fields(extracted_data, errors, ocr_results)
            elif doc_type == DocumentType.KENYAN_ID:
                confidence = self._extract_kenyan_id_fields(extracted_data, errors, ocr_results)
            elif doc_type == DocumentType.FOREIGN_ID:
                confidence = self._extract_foreign_id_fields(extracted_data, errors, ocr_results)
            elif doc_type == DocumentType.BUSINESS_CERT:
                confidence = self._extract_business_fields(extracted_data, errors, ocr_results)
            else:
                # Extract basic information for unknown types
                all_text = ' '.join([r.text for r in ocr_results])
                extracted_data['raw_text'] = all_text[:500]  # First 500 chars
                confidence = 20 if all_text.strip() else 0
            
            success = len(errors) == 0 and confidence > 10  # Lower threshold
            
            return ExtractionResult(
                success=success,
                data=extracted_data,
                confidence=min(confidence, 100),
                errors=errors
            )
            
        except Exception as e:
            logger.error(f"Enhanced extraction error: {str(e)}")
            return ExtractionResult(False, {}, 0, [f"Extraction failed: {str(e)}"])
    
    def _extract_kra_fields(self, extracted_data: Dict, errors: List, ocr_results: List[OCRResult]) -> int:
        """Extract KRA PIN certificate fields"""
        confidence_gained = 0
        
        # Extract PIN number
        pin_patterns = [r'A\d{9}[A-Z]', r'[A-Z]\d{9}[A-Z]']
        pin_found = False
        
        for r in ocr_results:
            for pattern in pin_patterns:
                pin_match = re.search(pattern, r.text.upper())
                if pin_match:
                    extracted_data['pin_number'] = pin_match.group()
                    confidence_gained += 40
                    pin_found = True
                    break
            if pin_found:
                break
        
        if not pin_found:
            errors.append("PIN number not found")
        
        # Extract name (look for uppercase text lines)
        name_candidates = []
        for r in ocr_results:
            text = r.text.strip()
            if (len(text) > 5 and 
                text.isupper() and 
                not any(c.isdigit() for c in text) and
                not any(word in self.exclude_words for word in text.split())):
                name_candidates.append((text, r.confidence))
        
        if name_candidates:
            best_name = max(name_candidates, key=lambda x: x[1])[0]
            extracted_data['taxpayer_name'] = best_name
            confidence_gained += 30
        
        return confidence_gained
    
    def _extract_kenyan_id_fields(self, extracted_data: Dict, errors: List, ocr_results: List[OCRResult]) -> int:
        """Extract Kenyan ID fields"""
        confidence_gained = 0
        
        # Extract ID number (8 digits)
        for r in ocr_results:
            digits = re.findall(r'\d{8}', r.text)
            if digits:
                extracted_data['id_number'] = digits[0]
                confidence_gained += 40
                break
        
        if 'id_number' not in extracted_data:
            errors.append("ID number not found")
        
        # Extract name
        name_candidates = []
        for r in ocr_results:
            text = r.text.strip()
            words = text.split()
            if (2 <= len(words) <= 4 and 
                all(word.isalpha() for word in words) and
                not any(word in self.exclude_words for word in words)):
                name_candidates.append((text.upper(), r.confidence))
        
        if name_candidates:
            best_name = max(name_candidates, key=lambda x: x[1])[0]
            extracted_data['full_name'] = best_name
            confidence_gained += 35
        
        return confidence_gained
    
    def _extract_foreign_id_fields(self, extracted_data: Dict, errors: List, ocr_results: List[OCRResult]) -> int:
        """Extract foreign ID fields"""
        confidence_gained = 0
        
        # Extract passport/document number
        for r in ocr_results:
            # Look for passport number patterns
            passport_patterns = [r'[A-Z]{2}\d{7}', r'[A-Z]\d{7,8}']
            for pattern in passport_patterns:
                match = re.search(pattern, r.text.upper())
                if match:
                    extracted_data['passport_number'] = match.group()
                    confidence_gained += 35
                    break
        
        # Extract name
        name_candidates = []
        for r in ocr_results:
            text = r.text.strip()
            words = text.split()
            if (2 <= len(words) <= 4 and 
                all(len(word) >= 2 for word in words) and
                not any(word in self.exclude_words for word in words)):
                name_candidates.append((text.upper(), r.confidence))
        
        if name_candidates:
            best_name = max(name_candidates, key=lambda x: x[1])[0]
            extracted_data['full_name'] = best_name
            confidence_gained += 35
        
        # Look for nationality
        for r in ocr_results:
            text_up = r.text.upper()
            if 'MOLDOVA' in text_up:
                extracted_data['nationality'] = 'Moldovan'
                confidence_gained += 20
                break
        
        return confidence_gained
    
    def _extract_business_fields(self, extracted_data: Dict, errors: List, ocr_results: List[OCRResult]) -> int:
        """Extract business certificate fields"""
        confidence_gained = 0
        
        # Extract business name (longest reasonable text)
        business_candidates = []
        for r in ocr_results:
            text = r.text.strip()
            if (10 <= len(text) <= 100 and 
                not re.search(r'\d{4,}', text) and
                text not in self.exclude_words):
                business_candidates.append((text, r.confidence))
        
        if business_candidates:
            best_name = max(business_candidates, key=lambda x: x[1])[0]
            extracted_data['business_name'] = best_name
            confidence_gained += 35
        
        return confidence_gained

    # Keep the rest of your methods (process_document_async, update_job_status, etc.)
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
        """Get supported document types (4 total)"""
        return [
            {
                "type": "kra_pin", 
                "name": "KRA PIN Certificate", 
                "description": "Kenya Revenue Authority PIN Certificate",
                "fields": ["pin_number", "taxpayer_name"]
            },
            {
                "type": "kenyan_id", 
                "name": "Kenyan National ID", 
                "description": "Republic of Kenya National Identity Card",
                "fields": ["id_number", "full_name", "sex", "date_of_birth"]
            },
            {
                "type": "foreign_id", 
                "name": "Foreign ID/Passport", 
                "description": "Foreign National Identity Document or Passport",
                "fields": ["passport_number", "full_name", "nationality"]
            },
            {
                "type": "business_cert", 
                "name": "Business Registration", 
                "description": "Certificate of Business Registration",
                "fields": ["business_name", "registration_number"]
            },
        ]
    
    async def health_check(self) -> Dict:
        """Enhanced health check with Tesseract testing"""
        services = {"api": "ok"}
        
        # Test OCR
        try:
            test_image = np.ones((100, 100), dtype=np.uint8) * 255
            pytesseract.image_to_string(test_image)
            services["ocr"] = "ok"
        except Exception as e:
            services["ocr"] = f"error: {str(e)}"
        
        # Test Redis
        if self.redis_client:
            try:
                await self.redis_client.ping()
                services["redis"] = "ok"
            except Exception:
                services["redis"] = "error"
        else:
            services["redis"] = "not_configured"
        
        # Templates
        services["templates"] = len(self.template_manager.templates)
        
        status = "healthy" if services["ocr"] == "ok" else "degraded"
        
        return {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "services": services
        }