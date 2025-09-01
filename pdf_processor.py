"""
PDF Processing Module for RAG Chatbot
Handles PDF text extraction, image extraction, and OCR processing
"""

import fitz  # PyMuPDF
import logging
import io
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import json
import hashlib

# OCR imports
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

from config import Config
from text_utils import TextChunker

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Handles PDF processing including text extraction, image extraction, and OCR"""
    
    def __init__(self):
        self.config = Config()
        self.text_chunker = TextChunker()
        self.ocr_reader = None
        self._initialize_ocr()
    
    def _initialize_ocr(self):
        """Initialize OCR engine based on configuration"""
        if self.config.OCR_ENGINE == "easyocr" and EASYOCR_AVAILABLE:
            try:
                self.ocr_reader = easyocr.Reader(self.config.OCR_LANGUAGES)
                logger.info("✅ EasyOCR initialized successfully")
            except Exception as e:
                logger.warning(f"⚠️ EasyOCR initialization failed: {e}")
                if TESSERACT_AVAILABLE:
                    logger.info("🔄 Falling back to Tesseract")
                    self.config.OCR_ENGINE = "tesseract"
        
        elif self.config.OCR_ENGINE == "tesseract" and not TESSERACT_AVAILABLE:
            logger.warning("⚠️ Tesseract not available, trying EasyOCR")
            if EASYOCR_AVAILABLE:
                self.config.OCR_ENGINE = "easyocr"
                self._initialize_ocr()
        
        if not EASYOCR_AVAILABLE and not TESSERACT_AVAILABLE:
            logger.warning("⚠️ No OCR engine available. Image text extraction will be skipped.")
    
    def process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Main method to process PDF and return chunks with metadata
        
        Returns:
            List of chunks with metadata including text, source, type, etc.
        """
        logger.info(f"📄 Processing PDF: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            # Open PDF document
            doc = fitz.open(pdf_path)
            logger.info(f"📊 PDF has {len(doc)} pages")
            
            all_chunks = []
            
            for page_num in range(len(doc)):
                logger.info(f"🔄 Processing page {page_num + 1}/{len(doc)}")
                
                page = doc[page_num]
                
                # Extract text from page
                text_chunks = self._extract_text_from_page(page, page_num + 1, pdf_path)
                all_chunks.extend(text_chunks)
                
                # Extract and process images if enabled
                if self.config.EXTRACT_IMAGES:
                    image_chunks = self._extract_images_from_page(page, page_num + 1, pdf_path)
                    all_chunks.extend(image_chunks)
            
            doc.close()
            
            logger.info(f"✅ PDF processing complete. Generated {len(all_chunks)} chunks")
            
            # Save processing results
            self._save_processing_metadata(all_chunks, pdf_path)
            
            return all_chunks
            
        except Exception as e:
            logger.error(f"❌ Error processing PDF: {str(e)}")
            raise
    
    def _extract_text_from_page(self, page, page_num: int, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract and chunk text from a PDF page"""
        try:
            # Get text with formatting
            text = page.get_text()
            
            if not text or len(text.strip()) < self.config.MIN_TEXT_LENGTH:
                return []
            
            # Clean and preprocess text
            cleaned_text = self._clean_text(text)
            
            # Split into chunks
            chunks = self.text_chunker.chunk_text(cleaned_text)
            
            # Create chunk objects with metadata
            text_chunks = []
            for i, chunk_text in enumerate(chunks):
                if len(chunk_text.strip()) >= self.config.MIN_TEXT_LENGTH:
                    chunk = {
                        'content': chunk_text.strip(),
                        'type': 'text',
                        'source': pdf_path,
                        'page': page_num,
                        'chunk_id': f"text_{page_num}_{i}",
                        'metadata': {
                            'file_name': Path(pdf_path).name,
                            'extraction_method': 'pymupdf'
                        }
                    }
                    text_chunks.append(chunk)
            
            return text_chunks
            
        except Exception as e:
            logger.error(f"❌ Error extracting text from page {page_num}: {str(e)}")
            return []
    
    def _extract_images_from_page(self, page, page_num: int, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract images from PDF page and perform OCR"""
        try:
            image_chunks = []
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    # Get image data
                    xref = img[0]
                    pix = fitz.Pixmap(page.parent, xref)
                    
                    if pix.n - pix.alpha < 4:  # Skip non-RGB images
                        # Convert to PIL Image
                        img_data = pix.tobytes("ppm")
                        pil_image = Image.open(io.BytesIO(img_data))
                        
                        # Check image size
                        if (pil_image.size[0] >= self.config.MIN_IMAGE_SIZE[0] and 
                            pil_image.size[1] >= self.config.MIN_IMAGE_SIZE[1]):
                            
                            # Save image
                            image_filename = f"page_{page_num}_img_{img_index}.png"
                            image_path = Path(self.config.IMAGES_DIR) / image_filename
                            pil_image.save(image_path)
                            
                            # Perform OCR
                            ocr_text = self._perform_ocr(pil_image)
                            
                            if ocr_text and len(ocr_text.strip()) >= self.config.MIN_TEXT_LENGTH:
                                chunk = {
                                    'content': f"[IMAGE CONTENT] {ocr_text.strip()}",
                                    'type': 'image_ocr',
                                    'source': pdf_path,
                                    'page': page_num,
                                    'chunk_id': f"img_{page_num}_{img_index}",
                                    'image_path': str(image_path),
                                    'metadata': {
                                        'file_name': Path(pdf_path).name,
                                        'image_filename': image_filename,
                                        'image_size': pil_image.size,
                                        'extraction_method': 'pymupdf_ocr'
                                    }
                                }
                                image_chunks.append(chunk)
                    
                    pix = None  # Clean up
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error processing image {img_index} on page {page_num}: {str(e)}")
                    continue
            
            return image_chunks
            
        except Exception as e:
            logger.error(f"❌ Error extracting images from page {page_num}: {str(e)}")
            return []
    
    def _perform_ocr(self, image: Image.Image) -> str:
        """Perform OCR on image using configured OCR engine"""
        try:
            if self.config.OCR_ENGINE == "easyocr" and self.ocr_reader:
                # Use EasyOCR
                results = self.ocr_reader.readtext(image, detail=1)
                text_parts = []
                
                for (bbox, text, confidence) in results:
                    if confidence >= self.config.OCR_CONFIDENCE_THRESHOLD:
                        text_parts.append(text)
                
                return " ".join(text_parts)
            
            elif self.config.OCR_ENGINE == "tesseract" and TESSERACT_AVAILABLE:
                # Use Tesseract
                try:
                    text = pytesseract.image_to_string(
                        image, 
                        config=self.config.TESSERACT_CONFIG
                    )
                    return text.strip()
                except Exception as e:
                    logger.warning(f"⚠️ Tesseract OCR failed: {str(e)}")
                    return ""
            
            else:
                logger.warning("⚠️ No OCR engine available")
                return ""
                
        except Exception as e:
            logger.error(f"❌ OCR processing failed: {str(e)}")
            return ""
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace and newlines
        text = ' '.join(text.split())
        
        # Remove common PDF artifacts
        text = text.replace('\x00', '')  # Remove null characters
        text = text.replace('\ufeff', '')  # Remove BOM
        
        # Remove headers/footers if configured
        if self.config.IGNORE_HEADERS_FOOTERS:
            lines = text.split('\n')
            # Simple heuristic: remove very short lines at start/end
            while lines and len(lines[0].strip()) < 20:
                lines.pop(0)
            while lines and len(lines[-1].strip()) < 20:
                lines.pop()
            text = '\n'.join(lines)
        
        return text.strip()
    
    def _save_processing_metadata(self, chunks: List[Dict[str, Any]], pdf_path: str):
        """Save processing metadata for debugging and analysis"""
        try:
            metadata = {
                'pdf_path': pdf_path,
                'total_chunks': len(chunks),
                'text_chunks': len([c for c in chunks if c['type'] == 'text']),
                'image_chunks': len([c for c in chunks if c['type'] == 'image_ocr']),
                'processing_config': {
                    'chunk_size': self.config.CHUNK_SIZE,
                    'chunk_overlap': self.config.CHUNK_OVERLAP,
                    'ocr_engine': self.config.OCR_ENGINE,
                    'extract_images': self.config.EXTRACT_IMAGES
                },
                'file_hash': self._get_file_hash(pdf_path)
            }
            
            metadata_path = Path(self.config.DATA_DIR) / 'processing_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"💾 Processing metadata saved to {metadata_path}")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not save processing metadata: {str(e)}")
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate MD5 hash of file for change detection"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return "unknown"