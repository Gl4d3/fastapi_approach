import cv2
import numpy as np
import pytesseract
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
# import uuid
import traceback
import io
import os
from PIL import Image
import fitz
import json

# Multiple OCR imports
try:
    import paddleocr
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    logging.warning("PaddleOCR not available")

try:
    from vision_agent.tools import load_image, ocr as vision_ocr, overlay_bounding_boxes, save_image
    from pillow_heif import register_heif_opener
    register_heif_opener()
    VISION_AGENT_AVAILABLE = True
except ImportError:
    VISION_AGENT_AVAILABLE = False
    logging.warning("Vision Agent not available")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logging.warning("EasyOCR not available")

from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class DocumentType(Enum):
    NATIONAL_ID = "national_id"
    FOREIGN_CERTIFICATE = "foreign_certificate"
    PASSPORT = "passport"
    UNKNOWN = "unknown"

@dataclass
class OCRResult:
    text: str
    confidence: float
    bbox: List[float] = None
    engine: str = "unknown"

@dataclass
class DocumentClassification:
    document_type: DocumentType
    confidence: float
    reasoning: str

@dataclass
class ExtractedFields:
    document_type: str
    full_name: Optional[str] = None
    name: Optional[str] = None  # For foreign cert and passport
    id_number: Optional[str] = None
    indiv_number: Optional[str] = None  # For foreign cert
    passport_number: Optional[str] = None  # For passport
    sex: Optional[str] = None
    date_of_birth: Optional[str] = None
    date_of_expiry: Optional[str] = None  # For foreign cert

class BackupDocumentProcessor:
    def __init__(self):
        self.output_dir = "ocr_comparison_outputs"
        self.setup_output_directory()
        
        # Initialize OCR engines
        self.paddle_ocr = None
        self.easy_ocr = None
        self.initialize_ocr_engines()
        
        # Enhanced exclusion words based on your perfect logic
        self.exclude_words = {
            'republic', 'kenya', 'identity', 'card', 'certificate', 'holder',
            'signature', 'serial', 'number', 'date', 'birth', 'place', 'issue',
            'nationality', 'sex', 'male', 'female', 'specimen', 'foreigner', 'skon',
            'national', 'passport', 'passaport', 'passaporte', 'document', 'valid',
            'expires', 'issued', 'authority', 'government', 'official'
        }
    
    def setup_output_directory(self):
        """Create output directory for comparison results"""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(f"{self.output_dir}/originals", exist_ok=True)
            os.makedirs(f"{self.output_dir}/tesseract_annotated", exist_ok=True)
            os.makedirs(f"{self.output_dir}/paddle_annotated", exist_ok=True)
            os.makedirs(f"{self.output_dir}/vision_agent_annotated", exist_ok=True)
            os.makedirs(f"{self.output_dir}/easyocr_annotated", exist_ok=True)
            logger.info(f"✅ Output directories created at {self.output_dir}")
        except Exception as e:
            logger.error(f"❌ Failed to create output directories: {e}")
    
    def initialize_ocr_engines(self):
        """Initialize all available OCR engines"""
        try:
            # Initialize PaddleOCR
            if PADDLE_AVAILABLE:
                self.paddle_ocr = paddleocr.PaddleOCR(
                    use_angle_cls=True,  # Enable orientation detection
                    lang='en',
                    show_log=False,
                    use_gpu=False  # Set to True if GPU available
                )
                logger.info("✅ PaddleOCR initialized with orientation detection")
            
            # Initialize EasyOCR
            if EASYOCR_AVAILABLE:
                self.easy_ocr = easyocr.Reader(['en'], gpu=False)
                logger.info("✅ EasyOCR initialized")
                
            logger.info("🔧 OCR engines initialization completed")
            
        except Exception as e:
            logger.error(f"❌ OCR engines initialization failed: {e}")
    
    async def process_document_with_all_ocr(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Process document with all available OCR engines and compare results"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"{timestamp}_{filename.replace('.', '_')}"
            
            logger.info(f"🚀 Starting multi-OCR processing for: {filename}")
            
            # Convert file to image
            image = self._convert_to_opencv_image(file_content, filename)
            
            # Save original image
            original_path = f"{self.output_dir}/originals/{base_filename}_original.jpg"
            cv2.imwrite(original_path, image)
            logger.info(f"💾 Original image saved: {original_path}")
            
            # Process with all OCR engines
            ocr_results = {}
            
            # 1. Tesseract OCR
            logger.info("🔍 Processing with Tesseract OCR...")
            tesseract_results = self._process_with_tesseract(image, base_filename)
            ocr_results['tesseract'] = tesseract_results
            
            # 2. PaddleOCR
            if PADDLE_AVAILABLE and self.paddle_ocr:
                logger.info("🔍 Processing with PaddleOCR...")
                paddle_results = self._process_with_paddle(image, base_filename)
                ocr_results['paddleocr'] = paddle_results
            
            # 3. Vision Agent OCR
            if VISION_AGENT_AVAILABLE:
                logger.info("🔍 Processing with Vision Agent OCR...")
                vision_results = self._process_with_vision_agent(image, base_filename)
                ocr_results['vision_agent'] = vision_results
            
            # 4. EasyOCR
            if EASYOCR_AVAILABLE and self.easy_ocr:
                logger.info("🔍 Processing with EasyOCR...")
                easy_results = self._process_with_easyocr(image, base_filename)
                ocr_results['easyocr'] = easy_results
            
            # Analyze and compare results
            comparison_results = self._analyze_ocr_comparison(ocr_results, base_filename)
            
            logger.info(f"✅ Multi-OCR processing completed for: {filename}")
            
            return comparison_results
            
        except Exception as e:
            logger.error(f"❌ Multi-OCR processing failed: {str(e)}")
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _convert_to_opencv_image(self, file_content: bytes, filename: str) -> np.ndarray:
        """Convert file to OpenCV image format"""
        try:
            if filename.lower().endswith('.pdf'):
                # PDF processing
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
            
            # Resize if too large
            h, w = opencv_image.shape[:2]
            if h > 2000 or w > 2000:
                scale = min(2000/h, 2000/w)
                new_h, new_w = int(h * scale), int(w * scale)
                opencv_image = cv2.resize(opencv_image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            return opencv_image
            
        except Exception as e:
            raise ValueError(f"Failed to convert image: {str(e)}")
    
    def _process_with_tesseract(self, image: np.ndarray, base_filename: str) -> Dict[str, Any]:
        """Process with Tesseract OCR and save annotated image"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
            
            # Enhanced preprocessing
            denoised = cv2.medianBlur(gray, 3)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            
            # Get OCR data with bounding boxes
            ocr_data = pytesseract.image_to_data(enhanced, output_type=pytesseract.Output.DICT)
            
            # Convert to our format
            ocr_results = []
            h, w = enhanced.shape
            
            for i in range(len(ocr_data['text'])):
                text = ocr_data['text'][i].strip()
                if not text:
                    continue
                
                conf = float(ocr_data['conf'][i])
                if conf < 10:
                    continue
                
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
                
                ocr_results.append(OCRResult(
                    text=text,
                    confidence=conf / 100.0,
                    bbox=[x, y, x2, y2],
                    engine="tesseract"
                ))
            
            # Create annotated image
            annotated_image = self._create_annotated_image(image, ocr_results)
            annotated_path = f"{self.output_dir}/tesseract_annotated/{base_filename}_tesseract.jpg"
            cv2.imwrite(annotated_path, annotated_image)
            
            # Classify and extract using perfect logic
            classification = self._classify_document_perfect_logic(ocr_results)
            extracted_fields = self._extract_fields_perfect_logic(ocr_results, classification.document_type)
            
            return {
                "engine": "tesseract",
                "ocr_results_count": len(ocr_results),
                "classification": {
                    "document_type": classification.document_type.value,
                    "confidence": classification.confidence,
                    "reasoning": classification.reasoning
                },
                "extracted_fields": extracted_fields.__dict__,
                "annotated_image_path": annotated_path,
                "raw_ocr_sample": [r.text for r in ocr_results[:10]]  # First 10 results
            }
            
        except Exception as e:
            logger.error(f"❌ Tesseract processing failed: {e}")
            return {
                "engine": "tesseract",
                "error": str(e),
                "ocr_results_count": 0
            }
        
    async def cleanup(self):
        """Cleanup resources"""
        if self.redis_client:
            await self.redis_client.close()
            
    def _process_with_paddle(self, image: np.ndarray, base_filename: str) -> Dict[str, Any]:
        """Process with PaddleOCR and save annotated image"""
        try:
            # PaddleOCR works with PIL images or numpy arrays
            results = self.paddle_ocr.ocr(image, cls=True)
            
            # Convert PaddleOCR results to our format
            ocr_results = []
            h, w = image.shape[:2]
            
            for line in results[0]:  # PaddleOCR returns nested list
                if line is None:
                    continue
                    
                bbox_coords, (text, confidence) = line
                
                # Normalize bounding box coordinates
                x_coords = [point[0] for point in bbox_coords]
                y_coords = [point[1] for point in bbox_coords]
                
                x1, y1 = min(x_coords) / w, min(y_coords) / h
                x2, y2 = max(x_coords) / w, max(y_coords) / h
                
                ocr_results.append(OCRResult(
                    text=text,
                    confidence=confidence,
                    bbox=[x1, y1, x2, y2],
                    engine="paddleocr"
                ))
            
            # Create annotated image
            annotated_image = self._create_annotated_image(image, ocr_results)
            annotated_path = f"{self.output_dir}/paddle_annotated/{base_filename}_paddle.jpg"
            cv2.imwrite(annotated_path, annotated_image)
            
            # Classify and extract using perfect logic
            classification = self._classify_document_perfect_logic(ocr_results)
            extracted_fields = self._extract_fields_perfect_logic(ocr_results, classification.document_type)
            
            return {
                "engine": "paddleocr",
                "ocr_results_count": len(ocr_results),
                "classification": {
                    "document_type": classification.document_type.value,
                    "confidence": classification.confidence,
                    "reasoning": classification.reasoning
                },
                "extracted_fields": extracted_fields.__dict__,
                "annotated_image_path": annotated_path,
                "raw_ocr_sample": [r.text for r in ocr_results[:10]]
            }
            
        except Exception as e:
            logger.error(f"❌ PaddleOCR processing failed: {e}")
            return {
                "engine": "paddleocr",
                "error": str(e),
                "ocr_results_count": 0
            }
    
    def _process_with_vision_agent(self, image: np.ndarray, base_filename: str) -> Dict[str, Any]:
        """Process with Vision Agent OCR and save annotated image"""
        try:
            # Convert OpenCV image to PIL for vision agent
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # Save temporarily for vision agent
            temp_path = f"{self.output_dir}/temp_{base_filename}.jpg"
            pil_image.save(temp_path)
            
            # Load with vision agent
            va_image = load_image(temp_path)
            
            # Run vision agent OCR
            va_results = vision_ocr(va_image)
            
            # Convert to our format
            ocr_results = []
            for result in va_results:
                # Vision agent returns different format
                text = result.get('label', '')
                confidence = result.get('score', 0.0)
                bbox = result.get('bbox', [0, 0, 1, 1])
                
                ocr_results.append(OCRResult(
                    text=text,
                    confidence=confidence,
                    bbox=bbox,
                    engine="vision_agent"
                ))
            
            # Create annotated image using vision agent's overlay
            try:
                annotated_va_image = overlay_bounding_boxes(va_image, va_results)
                annotated_path = f"{self.output_dir}/vision_agent_annotated/{base_filename}_vision_agent.jpg"
                save_image(annotated_va_image, annotated_path)
            except:
                # Fallback to our annotation method
                annotated_image = self._create_annotated_image(image, ocr_results)
                annotated_path = f"{self.output_dir}/vision_agent_annotated/{base_filename}_vision_agent.jpg"
                cv2.imwrite(annotated_path, annotated_image)
            
            # Clean up temp file
            os.remove(temp_path)
            
            # Classify and extract using perfect logic
            classification = self._classify_document_perfect_logic(ocr_results)
            extracted_fields = self._extract_fields_perfect_logic(ocr_results, classification.document_type)
            
            return {
                "engine": "vision_agent",
                "ocr_results_count": len(ocr_results),
                "classification": {
                    "document_type": classification.document_type.value,
                    "confidence": classification.confidence,
                    "reasoning": classification.reasoning
                },
                "extracted_fields": extracted_fields.__dict__,
                "annotated_image_path": annotated_path,
                "raw_ocr_sample": [r.text for r in ocr_results[:10]]
            }
            
        except Exception as e:
            logger.error(f"❌ Vision Agent processing failed: {e}")
            return {
                "engine": "vision_agent",
                "error": str(e),
                "ocr_results_count": 0
            }
    
    def _process_with_easyocr(self, image: np.ndarray, base_filename: str) -> Dict[str, Any]:
        """Process with EasyOCR and save annotated image"""
        try:
            # EasyOCR processing
            results = self.easy_ocr.readtext(image)
            
            # Convert to our format
            ocr_results = []
            h, w = image.shape[:2]
            
            for bbox, text, confidence in results:
                # Normalize bounding box
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                
                x1, y1 = min(x_coords) / w, min(y_coords) / h
                x2, y2 = max(x_coords) / w, max(y_coords) / h
                
                ocr_results.append(OCRResult(
                    text=text,
                    confidence=confidence,
                    bbox=[x1, y1, x2, y2],
                    engine="easyocr"
                ))
            
            # Create annotated image
            annotated_image = self._create_annotated_image(image, ocr_results)
            annotated_path = f"{self.output_dir}/easyocr_annotated/{base_filename}_easyocr.jpg"
            cv2.imwrite(annotated_path, annotated_image)
            
            # Classify and extract using perfect logic
            classification = self._classify_document_perfect_logic(ocr_results)
            extracted_fields = self._extract_fields_perfect_logic(ocr_results, classification.document_type)
            
            return {
                "engine": "easyocr",
                "ocr_results_count": len(ocr_results),
                "classification": {
                    "document_type": classification.document_type.value,
                    "confidence": classification.confidence,
                    "reasoning": classification.reasoning
                },
                "extracted_fields": extracted_fields.__dict__,
                "annotated_image_path": annotated_path,
                "raw_ocr_sample": [r.text for r in ocr_results[:10]]
            }
            
        except Exception as e:
            logger.error(f"❌ EasyOCR processing failed: {e}")
            return {
                "engine": "easyocr",
                "error": str(e),
                "ocr_results_count": 0
            }
    
    def _create_annotated_image(self, image: np.ndarray, ocr_results: List[OCRResult]) -> np.ndarray:
        """Create annotated image with bounding boxes"""
        annotated = image.copy()
        h, w = image.shape[:2]
        
        for result in ocr_results:
            if result.bbox:
                x1, y1, x2, y2 = result.bbox
                # Convert normalized coordinates back to pixels
                x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
                
                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add text label
                label = f"{result.text[:20]}... ({result.confidence:.2f})"
                cv2.putText(annotated, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return annotated
    
    def _classify_document_perfect_logic(self, ocr_results: List[OCRResult]) -> DocumentClassification:
        """Classify document using the perfect logic you specified"""
        # Combine all text
        text_joined = ' '.join([r.text.lower() for r in ocr_results])
        
        # Perfect classification logic as specified
        if any(foreign in text_joined for foreign in ['foreigner certificate', 'foreign certificate', 'foreign', 'foreigner']):
            # Check for 'foreigner certificate' first
            return DocumentClassification(
                document_type=DocumentType.FOREIGN_CERTIFICATE,
                confidence=95.0,
                reasoning="Contains 'foreigner certificate' - highest priority classification"
            )
        elif any(passport_word in text_joined for passport_word in ['passport', 'passaport', 'passaporte']):
            return DocumentClassification(
                document_type=DocumentType.PASSPORT,
                confidence=90.0,
                reasoning="Contains passport-related words in multiple languages"
            )
        elif 'republic of kenya' in text_joined and 'foreigner certificate' not in text_joined:
            return DocumentClassification(
                document_type=DocumentType.NATIONAL_ID,
                confidence=85.0,
                reasoning="Contains 'republic of kenya' without 'foreigner certificate'"
            )
        else:
            return DocumentClassification(
                document_type=DocumentType.UNKNOWN,
                confidence=0.0,
                reasoning="No clear document type indicators found"
            )
    
    def _extract_fields_perfect_logic(self, ocr_results: List[OCRResult], doc_type: DocumentType) -> ExtractedFields:
        """Extract fields using the perfect logic from your vision agent code"""
        fields = ExtractedFields(document_type=doc_type.value)
        
        for r in ocr_results:
            text = r.text.strip()
            text_lower = text.lower()
            
            # Extract name using perfect logic
            words_spl = text.split()
            if (
                2 <= len(words_spl) <= 3
                and not any(w.lower() in self.exclude_words for w in words_spl)
                and not any(c.isdigit() for c in text)
                and all(len(w) > 1 and w[0].isupper() for w in words_spl)
            ):
                # Set name based on document type
                if doc_type == DocumentType.NATIONAL_ID and fields.full_name is None:
                    fields.full_name = text
                elif doc_type in [DocumentType.FOREIGN_CERTIFICATE, DocumentType.PASSPORT] and fields.name is None:
                    fields.name = text
            
            # Extract sex
            if text_lower in ['male', 'female']:
                fields.sex = text.title()
            
            # Extract numbers based on document type
            numeric_part = ''.join(c for c in text if c.isdigit())
            if numeric_part:
                if doc_type == DocumentType.FOREIGN_CERTIFICATE:
                    if len(numeric_part) < 8 and fields.indiv_number is None:
                        fields.indiv_number = numeric_part
                elif doc_type == DocumentType.PASSPORT:
                    if 6 <= len(numeric_part) <= 10 and fields.passport_number is None:
                        fields.passport_number = numeric_part
                elif doc_type == DocumentType.NATIONAL_ID:
                    if len(numeric_part) == 8 and fields.id_number is None:
                        fields.id_number = numeric_part
            
            # Extract dates using perfect logic
            try_formats = ['%d.%m.%Y', '%d.%m.%y', '%d/%m/%Y', '%d/%m/%y']
            for fmt in try_formats:
                try:
                    dt = datetime.strptime(text, fmt)
                    # Handle 2-digit years
                    if len(text.split('.')[-1]) == 2 or len(text.split('/')[-1]) == 2:
                        if dt.year > 50:
                            dt = dt.replace(year=dt.year + 1900)
                        else:
                            dt = dt.replace(year=dt.year + 2000)
                    
                    # Date logic based on document type and year
                    if dt.year < 2015 and fields.date_of_birth is None:
                        fields.date_of_birth = text
                    elif dt.year >= 2015 and doc_type == DocumentType.FOREIGN_CERTIFICATE and fields.date_of_expiry is None:
                        fields.date_of_expiry = text
                except ValueError:
                    pass
        
        return fields
    
    def _analyze_ocr_comparison(self, ocr_results: Dict[str, Any], base_filename: str) -> Dict[str, Any]:
        """Analyze and compare results from all OCR engines"""
        
        # Create comprehensive comparison
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "filename": base_filename,
            "total_engines_used": len([r for r in ocr_results.values() if 'error' not in r]),
            "engines_with_errors": [engine for engine, result in ocr_results.items() if 'error' in result],
            "ocr_engine_results": ocr_results,
            "comparison_analysis": {},
            "best_result": None,
            "output_directory": self.output_dir
        }
        
        # Find best result based on OCR count and classification confidence
        best_engine = None
        best_score = 0
        
        for engine, result in ocr_results.items():
            if 'error' in result:
                continue
                
            # Score based on OCR results count and classification confidence
            ocr_count = result.get('ocr_results_count', 0)
            classification_conf = result.get('classification', {}).get('confidence', 0)
            
            score = (ocr_count * 0.3) + (classification_conf * 0.7)
            
            if score > best_score:
                best_score = score
                best_engine = engine
        
        if best_engine:
            comparison['best_result'] = {
                "engine": best_engine,
                "score": best_score,
                "result": ocr_results[best_engine]
            }
        
        # Analyze differences between engines
        classification_agreement = {}
        for engine, result in ocr_results.items():
            if 'error' not in result:
                doc_type = result.get('classification', {}).get('document_type', 'unknown')
                if doc_type in classification_agreement:
                    classification_agreement[doc_type] += 1
                else:
                    classification_agreement[doc_type] = 1
        
        comparison['comparison_analysis'] = {
            "classification_agreement": classification_agreement,
            "most_agreed_document_type": max(classification_agreement, key=classification_agreement.get) if classification_agreement else "unknown",
            "engines_agreement_rate": max(classification_agreement.values()) / len([r for r in ocr_results.values() if 'error' not in r]) if classification_agreement else 0
        }
        
        # Save comparison results to file
        comparison_file = f"{self.output_dir}/{base_filename}_comparison.json"
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 Comparison results saved: {comparison_file}")
        
        return comparison

# Create a simple endpoint to use this backup processor
class BackupProcessorManager:
    def __init__(self):
        self.processor = BackupDocumentProcessor()
    
    async def process_document(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Process document with backup processor"""
        return await self.processor.process_document_with_all_ocr(file_content, filename)
    
    def get_available_engines(self) -> List[str]:
        """Get list of available OCR engines"""
        engines = ["tesseract"]  # Always available
        
        if PADDLE_AVAILABLE:
            engines.append("paddleocr")
        if VISION_AGENT_AVAILABLE:
            engines.append("vision_agent")
        if EASYOCR_AVAILABLE:
            engines.append("easyocr")
            
        return engines
    
    def get_output_directory(self) -> str:
        """Get output directory path"""
        return self.processor.output_dir