"""
Image Processor Module for Multimodal RAG Chatbot
Handles image captioning, processing, and integration with text responses
"""

import logging
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import hashlib
from PIL import Image

# Vision model imports
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    BLIP_AVAILABLE = True
except ImportError:
    BLIP_AVAILABLE = False

try:
    import clip
    import torch
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

from config import Config

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Handles image processing and captioning for multimodal responses"""
    
    def __init__(self):
        self.config = Config()
        self.vision_model = None
        self.processor = None
        self.device = "cpu"  # Use CPU for compatibility
        
        # Cache for generated captions
        self.caption_cache_path = Path(self.config.DATA_DIR) / "image_captions.json"
        self.caption_cache = self._load_caption_cache()
        
        # Initialize vision model
        if self.config.ENABLE_IMAGE_RESPONSES:
            self._initialize_vision_model()
    
    def _initialize_vision_model(self):
        """Initialize the vision model for image captioning"""
        try:
            if self.config.VISION_MODEL == "blip2" and BLIP_AVAILABLE:
                logger.info("🔄 Loading BLIP-2 vision model...")
                self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.vision_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                logger.info("✅ BLIP-2 model loaded successfully")
                
            elif self.config.VISION_MODEL == "clip" and CLIP_AVAILABLE:
                logger.info("🔄 Loading CLIP vision model...")
                self.vision_model, self.processor = clip.load("ViT-B/32", device=self.device)
                logger.info("✅ CLIP model loaded successfully")
                
            else:
                logger.warning("⚠️ No vision model available. Image captioning will be limited.")
                
        except Exception as e:
            logger.error(f"❌ Error loading vision model: {str(e)}")
            logger.info("📄 Will use OCR text only for image descriptions")
    
    def _load_caption_cache(self) -> Dict[str, str]:
        """Load cached image captions"""
        try:
            if self.caption_cache_path.exists():
                with open(self.caption_cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"⚠️ Could not load caption cache: {str(e)}")
        
        return {}
    
    def _save_caption_cache(self):
        """Save caption cache to disk"""
        try:
            with open(self.caption_cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.caption_cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"⚠️ Could not save caption cache: {str(e)}")
    
    def _get_image_hash(self, image_path: str) -> str:
        """Generate hash for image file"""
        try:
            with open(image_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return str(hash(image_path))
    
    def generate_image_caption(self, image_path: str) -> str:
        """
        Generate caption for an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Generated caption or description
        """
        if not os.path.exists(image_path):
            return "Image not found"
        
        # Check cache first
        image_hash = self._get_image_hash(image_path)
        if self.config.USE_CACHED_CAPTIONS and image_hash in self.caption_cache:
            return self.caption_cache[image_hash]
        
        caption = self._generate_caption_with_model(image_path)
        
        # Cache the result
        if caption:
            self.caption_cache[image_hash] = caption
            self._save_caption_cache()
        
        return caption or "Unable to generate image description"
    
    def _generate_caption_with_model(self, image_path: str) -> Optional[str]:
        """Generate caption using the loaded vision model"""
        try:
            # Load and preprocess image
            image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Generate caption based on model type
            if self.config.VISION_MODEL == "blip2" and self.vision_model and self.processor:
                return self._generate_blip_caption(image)
            
            elif self.config.VISION_MODEL == "clip" and self.vision_model:
                return self._generate_clip_description(image)
            
            else:
                # Fallback: use a simple description based on image properties
                return self._generate_simple_description(image)
                
        except Exception as e:
            logger.error(f"❌ Error generating caption for {image_path}: {str(e)}")
            return None
    
    def _generate_blip_caption(self, image: Image.Image) -> str:
        """Generate caption using BLIP model"""
        try:
            # Process image
            inputs = self.processor(image, return_tensors="pt")
            
            # Generate caption
            out = self.vision_model.generate(**inputs, max_length=50)
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            
            return caption.strip()
            
        except Exception as e:
            logger.error(f"❌ BLIP caption generation failed: {str(e)}")
            return "Technical diagram or illustration"
    
    def _generate_clip_description(self, image: Image.Image) -> str:
        """Generate description using CLIP model"""
        try:
            # Prepare image
            image_input = self.processor(image).unsqueeze(0).to(self.device)
            
            # Common descriptions for technical documents
            text_options = [
                "a technical diagram",
                "a flowchart or process diagram", 
                "a schematic or blueprint",
                "an instructional illustration",
                "a chart or graph",
                "equipment or machinery photo",
                "a safety procedure illustration",
                "a maintenance guide diagram"
            ]
            
            # Tokenize text options
            text_tokens = clip.tokenize(text_options).to(self.device)
            
            # Calculate similarities
            with torch.no_grad():
                image_features = self.vision_model.encode_image(image_input)
                text_features = self.vision_model.encode_text(text_tokens)
                
                # Calculate cosine similarity
                similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                
                # Get best match
                best_match_idx = similarities[0].argmax().item()
                confidence = similarities[0][best_match_idx].item()
                
                if confidence > 0.3:
                    return text_options[best_match_idx]
                else:
                    return "technical illustration or diagram"
                    
        except Exception as e:
            logger.error(f"❌ CLIP description generation failed: {str(e)}")
            return "technical illustration or diagram"
    
    def _generate_simple_description(self, image: Image.Image) -> str:
        """Generate simple description based on image properties"""
        try:
            width, height = image.size
            aspect_ratio = width / height
            
            # Determine likely content type based on image properties
            if aspect_ratio > 1.5:
                return "wide technical diagram or flowchart"
            elif aspect_ratio < 0.7:
                return "tall diagram or vertical process illustration"
            elif width > 800 and height > 600:
                return "detailed technical illustration or photograph"
            else:
                return "technical diagram or illustration"
                
        except Exception:
            return "technical image or diagram"
    
    def find_relevant_images(self, chunks: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Find images relevant to the query from retrieved chunks
        
        Args:
            chunks: Retrieved text chunks that may reference images
            query: User's query
            
        Returns:
            List of relevant image information
        """
        relevant_images = []
        query_lower = query.lower()
        
        for chunk in chunks:
            # Check if chunk is from image OCR
            if chunk.get('type') == 'image_ocr' and 'image_path' in chunk:
                image_path = chunk['image_path']
                
                if os.path.exists(image_path):
                    # Generate or retrieve caption
                    caption = self.generate_image_caption(image_path)
                    
                    # Calculate relevance score
                    relevance = self._calculate_image_relevance(chunk['content'], caption, query_lower)
                    
                    image_info = {
                        'path': image_path,
                        'caption': caption,
                        'ocr_text': chunk['content'].replace('[IMAGE CONTENT] ', ''),
                        'page': chunk.get('page', 'Unknown'),
                        'relevance_score': relevance,
                        'similarity_score': chunk.get('similarity_score', 0.0)
                    }
                    
                    relevant_images.append(image_info)
        
        # Sort by relevance and similarity
        relevant_images.sort(key=lambda x: (x['relevance_score'], x['similarity_score']), reverse=True)
        
        # Limit number of images
        return relevant_images[:self.config.MAX_IMAGES_PER_RESPONSE]
    
    def _calculate_image_relevance(self, ocr_text: str, caption: str, query: str) -> float:
        """Calculate relevance score between image content and query"""
        relevance = 0.0
        
        # Check OCR text relevance
        ocr_words = set(ocr_text.lower().split())
        query_words = set(query.split())
        ocr_overlap = len(ocr_words.intersection(query_words))
        
        if ocr_words:
            relevance += (ocr_overlap / len(ocr_words)) * 0.6
        
        # Check caption relevance
        caption_words = set(caption.lower().split())
        caption_overlap = len(caption_words.intersection(query_words))
        
        if caption_words:
            relevance += (caption_overlap / len(caption_words)) * 0.4
        
        # Boost for specific image keywords in query
        for keyword in self.config.IMAGE_KEYWORDS:
            if keyword in query:
                relevance += 0.2
                break
        
        return min(relevance, 1.0)
    
    def format_images_for_response(self, images: List[Dict[str, Any]]) -> str:
        """
        Format images for inclusion in response
        
        Args:
            images: List of relevant image information
            
        Returns:
            Formatted markdown string with images and descriptions
        """
        if not images:
            return ""
        
        formatted_parts = []
        
        for i, img_info in enumerate(images, 1):
            # Create relative path for display
            image_path = img_info['path']
            relative_path = os.path.relpath(image_path)
            
            # Format image with description
            image_section = f"""
**📸 Image {i} (Page {img_info['page']})**

![Image {i}]({relative_path})

*{img_info['caption']}*
"""
            
            # Add OCR text if available and different from caption
            ocr_text = img_info.get('ocr_text', '').strip()
            if ocr_text and ocr_text.lower() not in img_info['caption'].lower():
                image_section += f"\n**Text in image:** {ocr_text}\n"
            
            formatted_parts.append(image_section)
        
        return "\n".join(formatted_parts)
    
    def get_image_context_for_llm(self, images: List[Dict[str, Any]]) -> str:
        """
        Get image context information for LLM processing
        
        Args:
            images: List of relevant image information
            
        Returns:
            Text description of images for LLM context
        """
        if not images:
            return ""
        
        context_parts = []
        
        for i, img_info in enumerate(images, 1):
            context = f"Image {i} (Page {img_info['page']}): {img_info['caption']}"
            
            ocr_text = img_info.get('ocr_text', '').strip()
            if ocr_text:
                context += f" - Text in image: {ocr_text}"
            
            context_parts.append(context)
        
        return "\n".join(context_parts)
    
    def cleanup_old_captions(self, days_old: int = 30):
        """Clean up old cached captions"""
        try:
            import time
            current_time = time.time()
            cutoff_time = current_time - (days_old * 24 * 60 * 60)
            
            # This is a simple cleanup - in practice, you'd want to track creation times
            logger.info(f"🧹 Caption cache cleanup not implemented (would clean entries older than {days_old} days)")
            
        except Exception as e:
            logger.error(f"❌ Error during caption cleanup: {str(e)}")


class VisionModelManager:
    """Manages vision model loading and switching"""
    
    def __init__(self):
        self.available_models = self._check_available_models()
        
    def _check_available_models(self) -> Dict[str, bool]:
        """Check which vision models are available"""
        return {
            'blip2': BLIP_AVAILABLE,
            'clip': CLIP_AVAILABLE
        }
    
    def get_recommended_model(self) -> str:
        """Get recommended vision model based on availability"""
        if self.available_models['blip2']:
            return 'blip2'
        elif self.available_models['clip']:
            return 'clip'
        else:
            return 'none'
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models"""
        return {
            'available_models': self.available_models,
            'recommended': self.get_recommended_model(),
            'descriptions': {
                'blip2': 'BLIP-2: Best for natural image captioning',
                'clip': 'CLIP: Good for technical image classification',
                'none': 'No vision model available - OCR text only'
            }
        }