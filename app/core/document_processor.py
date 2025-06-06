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

# Conditional imports for Railway/Docker compatibility
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
    FOREIGN_ID = "foreign_id"  # Added from vision AI
    PASSPORT = "passport"
    DRIVERS_LICENSE = "drivers_license"
    UNKNOWN = "unknown"

@dataclass
class OCRResult:
    text: str
    confidence: float
    bbox: List[float] = None

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
        
        # Vision AI enhancements - exclusion words for better name detection
        self.exclude_words = {
            'REPUBLIC', 'KENYA', 'NATIONAL', 'IDENTITY', 'CARD', 'SPECIMEN', 'FOREIGNER',
            'CERTIFICATE', 'SERIAL', 'NUMBER', 'CITY', 'HOLDER', 'SIGNATURE', 'PLACE', 
            'DATE', 'ISSUE', 'BIRTH', 'FOREIGN', 'PERMIT', 'VISA', 'TEMPORARY', 'PERMANENT',
            'ALIEN', 'RESIDENT', 'CITIZEN', 'KENYAN', 'PIN', 'AUTHORITY', 'REVENUE',
            'BUSINESS', 'REGISTRATION', 'REGISTRAR', 'COMPANY'
        }
        
    async def initialize(self):
        """Initialize processor components with enhanced error handling"""
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
                    logger.info("Redis connection established via REDIS_URL")
                except Exception as e:
                    logger.warning(f"Redis connection via REDIS_URL failed: {e}")
                    self.redis_client = None
            elif REDIS_AVAILABLE:
                # Fallback to individual parameters for local Docker
                try:
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
                    logger.info("Redis connection established via individual parameters")
                except Exception as e:
                    logger.warning(f"Redis connection via parameters failed: {e}")
                    self.redis_client = None
            else:
                logger.info("Redis not available, running without caching")
        
            # Enhanced Tesseract configuration with debugging
            tesseract_path = settings.TESSERACT_PATH
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            
            # Test Tesseract with detailed debugging
            try:
                test_image = np.ones((100, 100), dtype=np.uint8) * 255
                test_result = pytesseract.image_to_string(test_image)
                logger.info(f"✅ Tesseract test successful at {tesseract_path}")
                
                # Test image_to_data function
                test_data = pytesseract.image_to_data(test_image, output_type=pytesseract.Output.DICT)
                logger.info(f"✅ Tesseract image_to_data working, keys: {list(test_data.keys())}")
                
            except Exception as e:
                logger.error(f"❌ Tesseract test failed: {e}")
                # Try alternative paths
                alternative_paths = ['/usr/bin/tesseract', '/usr/local/bin/tesseract', 'tesseract']
                tesseract_working = False
                
                for path in alternative_paths:
                    try:
                        pytesseract.pytesseract.tesseract_cmd = path
                        pytesseract.image_to_string(test_image)
                        logger.info(f"✅ Tesseract working with alternative path: {path}")
                        tesseract_working = True
                        break
                    except:
                        continue
                
                if not tesseract_working:
                    logger.error("❌ No working Tesseract installation found!")
                    logger.error("This will cause OCR to return 0 text elements")
            
            # Load templates
            self.template_manager.load_templates()
            logger.info(f"Loaded {len(self.template_manager.templates)} templates")
            
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
        """Enhanced document processing with dual approach: templates + direct OCR"""
        try:
            logger.info(f"🔄 Starting processing for file: {filename}")
            
            # Convert file to images
            logger.info("📁 Converting to OpenCV images...")
            images = await self.file_handler.convert_to_images(file_content, filename)
            logger.info(f"📁 Converted to {len(images)} images")
            
            if not images:
                return self._create_error_result("No images could be extracted from file")
            
            # Process first image (main page)
            image = images[0]
            logger.info(f"🖼️ Processing image with shape: {image.shape}")
            
            # Try enhanced OCR extraction first (Vision AI approach)
            logger.info("🔍 Starting enhanced OCR extraction...")
            ocr_results = self._enhanced_ocr_extraction_with_debugging(image)
            logger.info(f"🔍 Enhanced OCR found {len(ocr_results)} text elements")
            
            if ocr_results:
                # Use vision AI approach for classification and extraction
                logger.info("🎯 Using Vision AI approach for classification...")
                classification = self._vision_ai_classify_document(ocr_results)
                logger.info(f"🎯 Vision AI Classification: {classification.document_type.value}, confidence: {classification.confidence}")
                
                extraction = self._vision_ai_extract_fields(ocr_results, classification.document_type)
                logger.info(f"📊 Vision AI Extraction: success={extraction.success}, data={extraction.data}")
                
                overall_confidence = (classification.confidence + extraction.confidence) / 2
                
                return {
                    "document_type": classification.document_type.value,
                    "classification_confidence": classification.confidence,
                    "extraction_success": extraction.success,
                    "extracted_data": extraction.data,
                    "extraction_confidence": extraction.confidence,
                    "extraction_errors": extraction.errors,
                    "overall_confidence": overall_confidence,
                    "processing_method": "vision_ai",
                    "ocr_text_sample": classification.extracted_features.get('text_sample', ''),
                    "debug_info": {
                        "ocr_count": len(ocr_results),
                        "image_shape": list(image.shape),
                        "tesseract_path": pytesseract.pytesseract.tesseract_cmd
                    }
                }
            else:
                # Fallback to template approach
                logger.warning("⚠️ No OCR results, falling back to template approach...")
                
                # Classify document using template approach
                classification = self.classify_document(image)
                
                if classification.document_type != DocumentType.UNKNOWN:
                    # Extract data using template
                    extraction_result = self.extract_data_with_template(image, classification.document_type)
                    
                    overall_confidence = (classification.confidence + extraction_result.confidence) / 2
                    
                    return {
                        "document_type": classification.document_type.value,
                        "classification_confidence": classification.confidence,
                        "extraction_success": extraction_result.success,
                        "extracted_data": extraction_result.data,
                        "extraction_confidence": extraction_result.confidence,
                        "extraction_errors": extraction_result.errors,
                        "overall_confidence": overall_confidence,
                        "processing_method": "template_fallback",
                        "debug_info": {"fallback_reason": "No OCR results from enhanced extraction"}
                    }
                else:
                    return self._create_error_result("Document type could not be determined and OCR failed")
            
        except Exception as e:
            logger.error(f"❌ Processing error: {str(e)}")
            logger.error(f"❌ Processing traceback: {traceback.format_exc()}")
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
        """Enhanced OCR with comprehensive debugging and multiple strategies"""
        try:
            logger.info("🔧 Starting enhanced OCR preprocessing...")
            
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                logger.info(f"🔧 Converted to grayscale. Shape: {gray.shape}")
            else:
                gray = image.copy()
                logger.info(f"🔧 Image already grayscale. Shape: {gray.shape}")
            
            # Check image properties
            logger.info(f"📊 Image stats - Min: {gray.min()}, Max: {gray.max()}, Mean: {gray.mean():.2f}")
            
            # Enhanced preprocessing
            logger.info("🔧 Applying preprocessing...")
            
            # 1. Noise reduction
            denoised = cv2.medianBlur(gray, 3)
            
            # 2. Contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            
            # 3. Try multiple OCR configurations
            configs = [
                ('--psm 6', 'Uniform block of text'),
                ('--psm 4', 'Single column'),
                ('--psm 3', 'Fully automatic'),
                ('--psm 1', 'Automatic with OSD'),
                ('--psm 11', 'Sparse text'),
                ('--psm 12', 'Sparse text with OSD'),
            ]
            
            best_results = []
            best_text_length = 0
            
            for config, description in configs:
                try:
                    logger.info(f"🔍 Trying OCR config: {config} ({description})")
                    
                    # Try image_to_data first
                    try:
                        ocr_data = pytesseract.image_to_data(enhanced, config=config, output_type=pytesseract.Output.DICT)
                        
                        if not ocr_data or 'text' not in ocr_data:
                            logger.warning(f"⚠️ No OCR data for config {config}")
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
                                # Get bounding box (optional for fallback)
                                left = ocr_data.get('left', [0])[i] if i < len(ocr_data.get('left', [])) else 0
                                top = ocr_data.get('top', [0])[i] if i < len(ocr_data.get('top', [])) else 0
                                width = ocr_data.get('width', [w])[i] if i < len(ocr_data.get('width', [])) else w
                                height = ocr_data.get('height', [h])[i] if i < len(ocr_data.get('height', [])) else h
                                
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
                                
                            except Exception as e:
                                logger.warning(f"⚠️ Skipping OCR result {i} due to bbox error: {e}")
                                # Add without bbox
                                current_results.append(OCRResult(
                                    text=text,
                                    confidence=conf / 100.0,
                                    bbox=[0, 0, 1, 1]
                                ))
                                continue
                        
                        total_text_length = sum(len(r.text) for r in current_results)
                        logger.info(f"✅ Config {config}: {len(current_results)} results, {total_text_length} chars")
                        
                        if total_text_length > best_text_length:
                            best_text_length = total_text_length
                            best_results = current_results
                    
                    except Exception as data_error:
                        logger.warning(f"⚠️ image_to_data failed for {config}: {data_error}")
                        
                        # Fallback to simple string extraction
                        try:
                            simple_text = pytesseract.image_to_string(enhanced, config=config)
                            if simple_text.strip():
                                logger.info(f"✅ Fallback simple OCR for {config}: {len(simple_text)} chars")
                                simple_results = [OCRResult(
                                    text=simple_text.strip(),
                                    confidence=0.5,
                                    bbox=[0.0, 0.0, 1.0, 1.0]
                                )]
                                if len(simple_text) > best_text_length:
                                    best_text_length = len(simple_text)
                                    best_results = simple_results
                        except Exception as simple_error:
                            logger.warning(f"⚠️ Simple OCR also failed for {config}: {simple_error}")
                            
                except Exception as e:
                    logger.warning(f"⚠️ OCR config {config} failed completely: {e}")
                    continue
            
            logger.info(f"🎯 Best OCR results: {len(best_results)} elements, {best_text_length} chars total")
            
            if best_results:
                # Log sample of detected text
                sample_texts = [r.text for r in best_results[:5]]
                logger.info(f"📝 Sample detected texts: {sample_texts}")
            else:
                logger.error("❌ All OCR configurations failed!")
            
            return best_results
            
        except Exception as e:
            logger.error(f"❌ OCR extraction error: {e}")
            logger.error(f"❌ OCR extraction traceback: {traceback.format_exc()}")
            return []
    
    def _vision_ai_classify_document(self, ocr_results: List[OCRResult]) -> ClassificationResult:
        """Vision AI classification with enhanced foreign document detection"""
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
            
            logger.info(f"🎯 Enhanced scores - KRA: {kra_score}, National: {national_score}, Foreign: {foreign_score}, Business: {business_score}")
            
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
            logger.error(f"❌ Vision AI classification error: {e}")
            return ClassificationResult(DocumentType.UNKNOWN, 0, {"error": str(e)})
    
    def _detect_languages(self, text: str) -> List[str]:
        """Simple language detection"""
        languages = []
        
        if any(word in text for word in ['REPUBLICA', 'MOLDOVA', 'ESTE', 'VALABIL', 'TOATE']):
            languages.append('romanian')
        if any(word in text for word in ['PASSEPORT', 'REPUBLIQUE', 'POUR', 'TOUS', 'LES', 'PAYS']):
            languages.append('french')
        if any(word in text for word in ['PASSPORT', 'VALID', 'ALL', 'COUNTRIES', 'REPUBLIC']):
            languages.append('english')
        if any(word in text for word in ['JAMHURI', 'YA', 'KENYA', 'KITAMBULISHO']):
            languages.append('swahili')
        
        return languages
    
    def _vision_ai_extract_fields(self, ocr_results: List[OCRResult], doc_type: DocumentType) -> ExtractionResult:
        """Vision AI field extraction with spatial analysis"""
        try:
            extracted_data = {}
            errors = []
            confidence = 0
            
            if not ocr_results:
                return ExtractionResult(False, {}, 0, ["No OCR results to extract from"])
            
            # Extract fields based on document type using Vision AI logic
            if doc_type == DocumentType.KRA_PIN:
                confidence = self._extract_kra_fields_vision_ai(extracted_data, errors, ocr_results)
            elif doc_type == DocumentType.KENYAN_ID:
                confidence = self._extract_kenyan_id_fields_vision_ai(extracted_data, errors, ocr_results)
            elif doc_type == DocumentType.FOREIGN_ID:
                confidence = self._extract_foreign_id_fields_vision_ai(extracted_data, errors, ocr_results)
            elif doc_type == DocumentType.BUSINESS_CERT:
                confidence = self._extract_business_fields_vision_ai(extracted_data, errors, ocr_results)
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
            logger.error(f"❌ Vision AI extraction error: {str(e)}")
            return ExtractionResult(False, {}, 0, [f"Extraction failed: {str(e)}"])
    
    def _extract_kra_fields_vision_ai(self, extracted_data: Dict, errors: List, ocr_results: List[OCRResult]) -> int:
        """Extract KRA PIN certificate fields using Vision AI logic"""
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
        
        # Extract taxpayer name using exclusion logic
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
    
    def _extract_kenyan_id_fields_vision_ai(self, extracted_data: Dict, errors: List, ocr_results: List[OCRResult]) -> int:
        """Extract Kenyan ID fields using Vision AI logic"""
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
        
        # Extract name using exclusion logic
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
    
    def _extract_foreign_id_fields_vision_ai(self, extracted_data: Dict, errors: List, ocr_results: List[OCRResult]) -> int:
        """Extract foreign ID fields using Vision AI logic"""
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
        
        # Extract name using exclusion logic
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
    
    def _extract_business_fields_vision_ai(self, extracted_data: Dict, errors: List, ocr_results: List[OCRResult]) -> int:
        """Extract business certificate fields using Vision AI logic"""
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

    # Keep existing template-based methods as fallback
    def classify_document(self, image: np.ndarray) -> ClassificationResult:
        """Template-based classification (fallback)"""
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
        """Template-based extraction (fallback)"""
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
        """Extract text using OCR (template fallback)"""
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
    
    # Keep all existing async methods unchanged
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
        """Get supported document types (now 4 total with foreign_id)"""
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
        """Enhanced health check with detailed Tesseract testing"""
        services = {"api": "ok"}
        
        # Test OCR with detailed information
        try:
            test_image = np.ones((100, 100), dtype=np.uint8) * 255
            test_result = pytesseract.image_to_string(test_image)
            test_data = pytesseract.image_to_data(test_image, output_type=pytesseract.Output.DICT)
            services["ocr"] = "ok"
            services["tesseract_path"] = pytesseract.pytesseract.tesseract_cmd
        except Exception as e:
            services["ocr"] = f"error: {str(e)}"
            services["tesseract_path"] = pytesseract.pytesseract.tesseract_cmd
        
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