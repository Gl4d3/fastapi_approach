import cv2
import numpy as np
from PIL import Image
import io
import fitz  # PyMuPDF
from typing import List, Union, Dict, Any  # Add Dict and Any here
import logging
import asyncio
from pathlib import Path

logger = logging.getLogger(__name__)

class FileHandler:
    """Handle different file types and convert to images for processing"""
    
    SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.webp', '.gif'}
    SUPPORTED_PDF_EXTENSIONS = {'.pdf'}
    
    def __init__(self):
        self.max_image_size = (2000, 2000)  # Max dimensions for processing
        self.pdf_dpi = 200  # DPI for PDF to image conversion
    
    def is_supported_extension(self, filename: str) -> bool:
        """Check if file extension is supported"""
        ext = Path(filename).suffix.lower()
        return ext in self.SUPPORTED_IMAGE_EXTENSIONS or ext in self.SUPPORTED_PDF_EXTENSIONS
    
    async def convert_to_images(self, file_content: bytes, filename: str) -> List[np.ndarray]:
        """Convert file content to list of OpenCV images"""
        ext = Path(filename).suffix.lower()
        
        if ext in self.SUPPORTED_PDF_EXTENSIONS:
            return await self._convert_pdf_to_images(file_content)
        elif ext in self.SUPPORTED_IMAGE_EXTENSIONS:
            return await self._convert_image_to_opencv(file_content)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    
    async def _convert_pdf_to_images(self, pdf_content: bytes) -> List[np.ndarray]:
        """Convert PDF to list of images"""
        try:
            # Run PDF processing in executor to avoid blocking
            loop = asyncio.get_event_loop()
            images = await loop.run_in_executor(None, self._process_pdf, pdf_content)
            return images
        except Exception as e:
            logger.error(f"PDF conversion error: {str(e)}")
            raise ValueError(f"Failed to convert PDF: {str(e)}")
    
    def _process_pdf(self, pdf_content: bytes) -> List[np.ndarray]:
        """Process PDF content (runs in executor)"""
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        images = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Convert to image
            mat = page.get_pixmap(matrix=fitz.Matrix(self.pdf_dpi/72, self.pdf_dpi/72))
            img_data = mat.tobytes("png")
            
            # Convert to OpenCV format
            pil_image = Image.open(io.BytesIO(img_data))
            opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # Resize if too large
            opencv_image = self._resize_if_needed(opencv_image)
            images.append(opencv_image)
        
        doc.close()
        return images
    
    async def _convert_image_to_opencv(self, image_content: bytes) -> List[np.ndarray]:
        """Convert image to OpenCV format"""
        try:
            # Run image processing in executor
            loop = asyncio.get_event_loop()
            image = await loop.run_in_executor(None, self._process_image, image_content)
            return [image]
        except Exception as e:
            logger.error(f"Image conversion error: {str(e)}")
            raise ValueError(f"Failed to convert image: {str(e)}")
    
    def _process_image(self, image_content: bytes) -> np.ndarray:
        """Process image content (runs in executor)"""
        # Open with PIL first to handle various formats
        pil_image = Image.open(io.BytesIO(image_content))
        
        # Convert to RGB if needed
        if pil_image.mode in ('RGBA', 'LA', 'P'):
            pil_image = pil_image.convert('RGB')
        
        # Convert to OpenCV format
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Resize if needed
        return self._resize_if_needed(opencv_image)
    
    def _resize_if_needed(self, image: np.ndarray) -> np.ndarray:
        """Resize image if it's too large"""
        h, w = image.shape[:2]
        max_h, max_w = self.max_image_size
        
        if h > max_h or w > max_w:
            # Calculate scaling factor
            scale = min(max_h/h, max_w/w)
            new_h, new_w = int(h * scale), int(w * scale)
            
            # Resize using high-quality interpolation
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        return image
    
    def validate_image_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """Validate image quality for OCR processing"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Check image size
        h, w = gray.shape
        if h < 100 or w < 100:
            return {"valid": False, "reason": "Image too small"}
        
        # Check if image is too blurry
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 100:
            return {"valid": False, "reason": "Image too blurry"}
        
        # Check brightness
        mean_brightness = gray.mean()
        if mean_brightness < 50 or mean_brightness > 200:
            return {"valid": False, "reason": "Poor lighting conditions"}
        
        return {"valid": True, "laplacian_variance": laplacian_var, "brightness": mean_brightness}