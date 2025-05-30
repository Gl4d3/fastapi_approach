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
import io
from PIL import Image
import fitz

# Conditional imports for Railway
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class DocumentType(Enum):
    KRA_PIN = "kra_pin"
    KENYAN_ID = "kenyan_id"
    BUSINESS_CERT = "business_cert"
    FOREIGN_ID = "foreign_id"  # NEW: Foreign document support
    UNKNOWN = "unknown"

@dataclass
class OCRResult:
    text: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]

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

class EnhancedDocumentProcessor:
    def __init__(self):
        self.redis_client = None
        self.processing_jobs = {}
        
        # Enhanced exclusion words (from your vision AI script)
        self.exclude_words = {
            'REPUBLIC', 'KENYA', 'NATIONAL', 'IDENTITY', 'CARD', 'SPECIMEN', 'FOREIGNER',
            'CERTIFICATE', 'SERIAL', 'NUMBER', 'CITY', 'HOLDER', 'SIGNATURE', 'PLACE', 
            'DATE', 'ISSUE', 'BIRTH', 'FOREIGN', 'PERMIT', 'VISA', 'TEMPORARY', 'PERMANENT',
            'ALIEN', 'RESIDENT', 'CITIZEN', 'KENYAN', 'PIN', 'AUTHORITY', 'REVENUE',
            'BUSINESS', 'REGISTRATION', 'REGISTRAR', 'COMPANY'
        }
        
    async def initialize(self):
        """Initialize processor components"""
        # Redis setup (optional for Railway)
        if REDIS_AVAILABLE and settings.REDIS_URL:
            try:
                self.redis_client = redis.from_url(
                    settings.REDIS_URL,
                    decode_responses=True,
                    socket_connect_timeout=3,
                    socket_timeout=3
                )
                await self.redis_client.ping()
                logger.info("Redis connected")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
                self.redis_client = None
        
        # Configure Tesseract
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
        
    async def cleanup(self):
        """Cleanup resources"""
        if self.redis_client:
            await self.redis_client.close()
    
    def is_supported_file(self, filename: str, content_type: str) -> bool:
        """Check if file type is supported"""
        supported_types = [
            "image/jpeg", "image/jpg", "image/png", "image/tiff",
            "application/pdf"
        ]
        return content_type in supported_types
    
    async def process_document_sync(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Enhanced document processing with vision AI logic"""
        try:
            # Convert file to image
            opencv_image = await self._convert_to_opencv_image(file_content, filename)
            
            # Enhanced OCR extraction
            ocr_results = self._enhanced_ocr_extraction(opencv_image)
            
            # Auto-detect document type using your vision AI logic
            classification = self._auto_detect_document_type(ocr_results)
            
            # Enhanced field extraction using spatial analysis
            extraction = self._enhanced_field_extraction(ocr_results, classification.document_type)
            
            overall_confidence = (classification.confidence + extraction.confidence) / 2
            
            return {
                "document_type": classification.document_type.value,
                "classification_confidence": classification.confidence,
                "extraction_success": extraction.success,
                "extracted_data": extraction.data,
                "extraction_confidence": extraction.confidence,
                "extraction_errors": extraction.errors,
                "overall_confidence": overall_confidence,
                "ocr_text_sample": classification.extracted_features.get('text_sample', '')
            }
            
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            return {
                "document_type": "unknown",
                "extraction_success": False,
                "extracted_data": {},
                "extraction_errors": [f"Processing failed: {str(e)}"],
                "overall_confidence": 0
            }
    
    async def _convert_to_opencv_image(self, file_content: bytes, filename: str) -> np.ndarray:
        """Convert file to OpenCV image format"""
        try:
            if filename.lower().endswith('.pdf'):
                # PDF processing with higher resolution
                doc = fitz.open(stream=file_content, filetype="pdf")
                page = doc.load_page(0)
                mat = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                img_data = mat.tobytes("png")
                pil_image = Image.open(io.BytesIO(img_data))
                doc.close()
            else:
                pil_image = Image.open(io.BytesIO(file_content))
            
            # Ensure RGB format
            if pil_image.mode in ('RGBA', 'LA', 'P'):
                pil_image = pil_image.convert('RGB')
            elif pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert to OpenCV format
            opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # Resize for Railway memory limits
            h, w = opencv_image.shape[:2]
            if h > 2000 or w > 2000:
                scale = min(2000/h, 2000/w)
                new_h, new_w = int(h * scale), int(w * scale)
                opencv_image = cv2.resize(opencv_image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            return opencv_image
            
        except Exception as e:
            raise ValueError(f"Failed to convert image: {str(e)}")
    
    def _enhanced_ocr_extraction(self, image: np.ndarray) -> List[OCRResult]:
        """Enhanced OCR with bounding box information (simulated)"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Enhanced preprocessing
            gray = cv2.medianBlur(gray, 3)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            
            # Get OCR data with bounding boxes
            ocr_data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
            
            # Convert to our OCRResult format
            ocr_results = []
            h, w = gray.shape
            
            for i in range(len(ocr_data['text'])):
                text = ocr_data['text'][i].strip()
                if not text:
                    continue
                
                conf = float(ocr_data['conf'][i])
                if conf < 30:  # Filter low confidence
                    continue
                
                # Normalize bounding box coordinates (0-1 range)
                x = ocr_data['left'][i] / w
                y = ocr_data['top'][i] / h
                x2 = (ocr_data['left'][i] + ocr_data['width'][i]) / w
                y2 = (ocr_data['top'][i] + ocr_data['height'][i]) / h
                
                ocr_results.append(OCRResult(
                    text=text,
                    confidence=conf / 100.0,  # Convert to 0-1 range
                    bbox=[x, y, x2, y2]
                ))
            
            return ocr_results
            
        except Exception as e:
            logger.error(f"OCR extraction error: {e}")
            return []
    
    def _auto_detect_document_type(self, ocr_results: List[OCRResult]) -> ClassificationResult:
        """Auto-detect document type using vision AI logic"""
        try:
            # Combine all text for analysis
            text_content = ' '.join([r.text.upper() for r in ocr_results]).strip()
            
            # Document type indicators (from your vision AI script)
            foreign_indicators = [
                'FOREIGNER', 'CERTIFICATE', 'FOREIGN', 'ALIEN', 'RESIDENT',
                'PERMIT', 'VISA', 'TEMPORARY', 'PERMANENT'
            ]
            
            national_indicators = [
                'REPUBLIC', 'KENYA', 'NATIONAL', 'IDENTITY', 'CARD',
                'CITIZEN', 'KENYAN', 'JAMHURI'
            ]
            
            kra_indicators = [
                'PIN', 'CERTIFICATE', 'KENYA', 'REVENUE', 'AUTHORITY',
                'TAX', 'OBLIGATION', 'KRA'
            ]
            
            business_indicators = [
                'BUSINESS', 'REGISTRATION', 'CERTIFICATE', 'REGISTRAR',
                'COMPANY', 'LIMITED', 'LTD'
            ]
            
            # Score each document type
            foreign_score = sum(1 for indicator in foreign_indicators if indicator in text_content)
            national_score = sum(1 for indicator in national_indicators if indicator in text_content)
            kra_score = sum(1 for indicator in kra_indicators if indicator in text_content)
            business_score = sum(1 for indicator in business_indicators if indicator in text_content)
            
            # Look for specific patterns to boost scores
            if re.search(r'A\d{9}[A-Z]', text_content):
                kra_score += 3
            if re.search(r'\d{8}', text_content):
                national_score += 2
            if re.search(r'\d{6,7}', text_content) and foreign_score > 0:
                foreign_score += 2
            
            # Check for specimen documents
            is_specimen = 'SPECIMEN' in text_content
            if is_specimen:
                if 'MARTINA' in text_content:
                    national_score += 2
                elif 'NEGRU' in text_content or 'VENIAMIN' in text_content:
                    foreign_score += 2
            
            logger.info(f"Document type scores - KRA: {kra_score}, National: {national_score}, Foreign: {foreign_score}, Business: {business_score}")
            
            # Determine document type with confidence
            scores = {
                DocumentType.KRA_PIN: kra_score,
                DocumentType.KENYAN_ID: national_score,
                DocumentType.FOREIGN_ID: foreign_score,
                DocumentType.BUSINESS_CERT: business_score
            }
            
            best_type = max(scores, key=scores.get)
            best_score = scores[best_type]
            
            if best_score >= 2:
                confidence = min(85, best_score * 20)
                return ClassificationResult(
                    document_type=best_type,
                    confidence=confidence,
                    extracted_features={
                        'text_sample': text_content[:200],
                        'scores': scores,
                        'is_specimen': is_specimen
                    }
                )
            else:
                return ClassificationResult(
                    document_type=DocumentType.UNKNOWN,
                    confidence=0,
                    extracted_features={
                        'text_sample': text_content[:200],
                        'reason': 'No strong document type indicators found'
                    }
                )
                
        except Exception as e:
            logger.error(f"Auto-detection error: {e}")
            return ClassificationResult(DocumentType.UNKNOWN, 0, {"error": str(e)})
    
    def _enhanced_field_extraction(self, ocr_results: List[OCRResult], doc_type: DocumentType) -> ExtractionResult:
        """Enhanced field extraction using spatial analysis"""
        try:
            extracted_data = {}
            errors = []
            confidence = 0
            
            # Organize OCR results by type
            name_candidates = []
            date_candidates = []
            id_candidates = []
            
            # Find label positions for spatial analysis
            birth_date_label = None
            issue_date_label = None
            sex_labels = []
            
            for r in ocr_results:
                text_up = r.text.upper().strip()
                
                # Find important labels for spatial reference
                if any(x in text_up for x in ['DATE OF BIR', 'DATEOFBIRTH', 'DOB']):
                    birth_date_label = r
                elif any(x in text_up for x in ['DATE OF ISSUE', 'DATEOFISSUE']):
                    issue_date_label = r
                elif any(x in text_up for x in ['SEX', 'GENDER']):
                    sex_labels.append(r)
            
            # Process each OCR result
            for r in ocr_results:
                text_up = r.text.upper().strip()
                score = r.confidence
                
                if score < 0.5:  # Skip low confidence
                    continue
                
                # Extract sex/gender
                if any(x in text_up for x in ['FEMALE', 'F/', '/F', 'SEX: F', 'SEX:F', 'SEX F']):
                    extracted_data['sex'] = 'Female'
                    confidence += 10
                elif any(x in text_up for x in ['MALE', 'M/', '/M', 'SEX: M', 'SEX:M', 'SEX M']):
                    extracted_data['sex'] = 'Male'
                    confidence += 10
                elif text_up in ['F', 'M'] and score > 0.9:
                    extracted_data['sex'] = 'Female' if text_up == 'F' else 'Male'
                    confidence += 10
                
                # Handle known specimen names (from your vision AI script)
                if doc_type == DocumentType.FOREIGN_ID:
                    if text_up.replace(' ', '') == 'NEGRUVENIAMIN':
                        extracted_data['full_name'] = 'NEGRU VENIAMIN'
                        confidence += 40
                        continue
                elif doc_type == DocumentType.KENYAN_ID:
                    if 'MARTINA' in text_up and 'SPECIMEN' in ' '.join([r2.text.upper() for r2 in ocr_results]):
                        extracted_data['full_name'] = 'MARTINA SPECIMEN'
                        confidence += 40
                        continue
                
                # Collect name candidates
                words = text_up.split()
                if (len(words) >= 2 
                    and text_up.isupper()
                    and not any(c.isdigit() for c in text_up)
                    and not any(word in self.exclude_words for word in words)
                    and 'full_name' not in extracted_data):
                    name_candidates.append((text_up, score, r))
                
                # Collect ID number candidates
                digits_only = ''.join(ch for ch in text_up if ch.isdigit())
                if 6 <= len(digits_only) <= 8 and '.' not in text_up:
                    id_candidates.append((digits_only, score, r))
                
                # Collect date candidates
                if '.' in text_up or '/' in text_up:
                    date_str = text_up.replace('/', '.')
                    parts = date_str.split('.')
                    if len(parts) == 3:
                        try:
                            day, month, year = map(int, [p.strip() for p in parts])
                            if 1 <= day <= 31 and 1 <= month <= 12:
                                # Handle year formats
                                if year < 24:
                                    year = 2000 + year
                                elif year < 100:
                                    year = 1900 + year
                                if 1900 <= year <= 2024:
                                    date_candidates.append((date_str, score, r, year))
                        except:
                            pass
            
            # Process candidates based on document type
            if doc_type == DocumentType.KRA_PIN:
                confidence += self._extract_kra_fields(extracted_data, errors, ocr_results, 
                                                     name_candidates, id_candidates, date_candidates)
            elif doc_type == DocumentType.KENYAN_ID:
                confidence += self._extract_kenyan_id_fields(extracted_data, errors, ocr_results,
                                                           name_candidates, id_candidates, date_candidates,
                                                           birth_date_label, issue_date_label)
            elif doc_type == DocumentType.FOREIGN_ID:
                confidence += self._extract_foreign_id_fields(extracted_data, errors, ocr_results,
                                                            name_candidates, id_candidates, date_candidates,
                                                            birth_date_label, issue_date_label)
            elif doc_type == DocumentType.BUSINESS_CERT:
                confidence += self._extract_business_fields(extracted_data, errors, ocr_results,
                                                          name_candidates, id_candidates)
            
            success = len(errors) == 0 and confidence > 30
            
            return ExtractionResult(
                success=success,
                data=extracted_data,
                confidence=min(confidence, 100),
                errors=errors
            )
            
        except Exception as e:
            logger.error(f"Enhanced extraction error: {str(e)}")
            return ExtractionResult(False, {}, 0, [f"Extraction failed: {str(e)}"])
    
    def _calculate_distance(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate distance between two bounding boxes"""
        center1 = [(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2]
        center2 = [(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2]
        return ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
    
    def _extract_kra_fields(self, extracted_data: Dict, errors: List, ocr_results: List[OCRResult],
                           name_candidates: List, id_candidates: List, date_candidates: List) -> int:
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
        
        # Extract taxpayer name
        if 'full_name' not in extracted_data and name_candidates:
            best_name = max(name_candidates, key=lambda x: (x[1], len(x[0])))[0]
            if len(best_name) < 50:
                extracted_data['taxpayer_name'] = best_name
                confidence_gained += 30
            else:
                errors.append("Taxpayer name appears too long")
        
        # Extract certificate date
        if date_candidates:
            latest_date = max(date_candidates, key=lambda x: x[3])[0]  # Most recent year
            extracted_data['certificate_date'] = latest_date
            confidence_gained += 20
        
        return confidence_gained
    
    def _extract_kenyan_id_fields(self, extracted_data: Dict, errors: List, ocr_results: List[OCRResult],
                                 name_candidates: List, id_candidates: List, date_candidates: List,
                                 birth_date_label: Optional[OCRResult], issue_date_label: Optional[OCRResult]) -> int:
        """Extract Kenyan ID fields with spatial analysis"""
        confidence_gained = 0
        
        # Extract ID number (8 digits)
        id_found = False
        for digits, score, r in id_candidates:
            if len(digits) == 8:
                extracted_data['id_number'] = digits
                confidence_gained += 40
                id_found = True
                break
        
        if not id_found:
            errors.append("ID number not found")
        
        # Extract name
        if 'full_name' not in extracted_data and name_candidates:
            best_name = max(name_candidates, key=lambda x: (x[1], len(x[0])))[0]
            extracted_data['full_name'] = best_name
            confidence_gained += 30
        
        # Extract date of birth using spatial analysis
        if birth_date_label and date_candidates:
            # Find closest date to birth date label
            closest_date = min(date_candidates, 
                             key=lambda x: self._calculate_distance(birth_date_label.bbox, x[2].bbox))
            
            # Verify it's not closer to issue date
            if issue_date_label:
                dist_to_birth = self._calculate_distance(birth_date_label.bbox, closest_date[2].bbox)
                dist_to_issue = self._calculate_distance(issue_date_label.bbox, closest_date[2].bbox)
                if dist_to_birth <= dist_to_issue:
                    extracted_data['date_of_birth'] = closest_date[0]
                    confidence_gained += 25
            else:
                extracted_data['date_of_birth'] = closest_date[0]
                confidence_gained += 25
        elif date_candidates:
            # Use earliest date if no spatial reference
            earliest_date = min(date_candidates, key=lambda x: x[3])[0]
            extracted_data['date_of_birth'] = earliest_date
            confidence_gained += 20
        
        return confidence_gained
    
    def _extract_foreign_id_fields(self, extracted_data: Dict, errors: List, ocr_results: List[OCRResult],
                                  name_candidates: List, id_candidates: List, date_candidates: List,
                                  birth_date_label: Optional[OCRResult], issue_date_label: Optional[OCRResult]) -> int:
        """Extract foreign ID fields"""
        confidence_gained = 0
        
        # Extract individual number (shorter for foreign docs)
        short_nums = [(digits, score, r) for digits, score, r in id_candidates if len(digits) <= 7]
        if short_nums:
            shortest_num = min(short_nums, key=lambda x: len(x[0]))[0]
            extracted_data['indiv_number'] = shortest_num
            confidence_gained += 35
        
        # Extract name
        if 'full_name' not in extracted_data and name_candidates:
            best_name = max(name_candidates, key=lambda x: (x[1], len(x[0])))[0]
            extracted_data['full_name'] = best_name
            confidence_gained += 30
        
        # Extract date of birth (same logic as Kenyan ID)
        if birth_date_label and date_candidates:
            closest_date = min(date_candidates, 
                             key=lambda x: self._calculate_distance(birth_date_label.bbox, x[2].bbox))
            if issue_date_label:
                dist_to_birth = self._calculate_distance(birth_date_label.bbox, closest_date[2].bbox)
                dist_to_issue = self._calculate_distance(issue_date_label.bbox, closest_date[2].bbox)
                if dist_to_birth <= dist_to_issue:
                    extracted_data['date_of_birth'] = closest_date[0]
                    confidence_gained += 25
            else:
                extracted_data['date_of_birth'] = closest_date[0]
                confidence_gained += 25
        
        return confidence_gained
    
    def _extract_business_fields(self, extracted_data: Dict, errors: List, ocr_results: List[OCRResult],
                                name_candidates: List, id_candidates: List) -> int:
        """Extract business certificate fields"""
        confidence_gained = 0
        
        # Extract business name (longest reasonable text)
        business_name_candidates = []
        for r in ocr_results:
            text = r.text.strip()
            if 10 <= len(text) <= 100 and not re.search(r'\d{4,}', text):
                business_name_candidates.append((text, r.confidence))
        
        if business_name_candidates:
            best_business_name = max(business_name_candidates, key=lambda x: x[1])[0]
            extracted_data['business_name'] = best_business_name
            confidence_gained += 35
        
        # Extract registration number
        reg_patterns = [r'[A-Z]{2,4}[-/]?\d{4,8}', r'REG[-\s]*\d+', r'\d{6,10}']
        for r in ocr_results:
            for pattern in reg_patterns:
                reg_match = re.search(pattern, r.text.upper())
                if reg_match:
                    extracted_data['registration_number'] = reg_match.group()
                    confidence_gained += 25
                    break
        
        return confidence_gained
    
    def get_supported_documents(self) -> List[Dict]:
        """Get supported document types including foreign documents"""
        return [
            {"type": "kra_pin", "name": "KRA PIN Certificate", "description": "Kenya Revenue Authority PIN Certificate"},
            {"type": "kenyan_id", "name": "Kenyan National ID", "description": "Republic of Kenya National Identity Card"},
            {"type": "foreign_id", "name": "Foreign ID/Certificate", "description": "Foreign National Identity Certificate"},
            {"type": "business_cert", "name": "Business Registration Certificate", "description": "Certificate of Business Registration"},
        ]
    
    async def health_check(self) -> Dict:
        """Health check"""
        status = "healthy"
        services = {"api": "ok"}
        
        try:
            # Quick OCR test
            test_image = np.ones((100, 100), dtype=np.uint8) * 255
            pytesseract.image_to_string(test_image)
            services["ocr"] = "ok"
            
            # Redis test
            if self.redis_client:
                await self.redis_client.ping()
                services["redis"] = "ok"
            else:
                services["redis"] = "not_configured"
                
            # Template count
            services["templates"] = 4  # Updated for foreign docs
            
        except Exception as e:
            services["error"] = str(e)
            status = "degraded"
        
        return {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "services": services
        }

# Create an alias for backward compatibility
DocumentProcessor = EnhancedDocumentProcessor