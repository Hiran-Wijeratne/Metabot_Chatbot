"""
Configuration settings for RAG Chatbot
Modify these settings according to your needs
"""

import os
from pathlib import Path

class Config:
    """Configuration class for RAG Chatbot"""
    
    # =============================================================================
    # MULTIMODAL FEATURES
    # =============================================================================
    
    ENABLE_IMAGE_RESPONSES = True
    MAX_IMAGES_PER_RESPONSE = 3
    
    VISION_MODEL = "blip2"  # Options: blip2, clip
    USE_CACHED_CAPTIONS = True
    
    IMAGE_KEYWORDS = [
        'image', 'images', 'picture', 'pictures', 'diagram', 'diagrams',
        'figure', 'figures', 'chart', 'charts', 'graph', 'graphs',
        'illustration', 'illustrations', 'photo', 'photos', 'visual',
        'screenshot', 'screenshots', 'drawing', 'drawings'
    ]
    
    SUMMARY_KEYWORDS = [
        'summary', 'summarize', 'overview', 'outline', 'main points',
        'key points', 'highlights', 'brief', 'all about', 'everything about',
        'entire', 'whole', 'complete', 'comprehensive', 'general information'
    ]
    
    # =============================================================================
    # FILE PATHS
    # =============================================================================
    
    PDF_PATH = "documents/META-SCRUB 60_User Manual.pdf"
    DATA_DIR = "data"
    IMAGES_DIR = "images"
    DOCUMENTS_DIR = "documents"
    
    # =============================================================================
    # OPENAI SETTINGS
    # =============================================================================
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MODEL_NAME = "gpt-3.5-turbo"
    EMBEDDING_MODEL = "text-embedding-3-small"
    MAX_TOKENS = 1000
    TEMPERATURE = 0.1
    
    # =============================================================================
    # TEXT PROCESSING SETTINGS
    # =============================================================================
    
    CHUNK_SIZE = 250
    CHUNK_OVERLAP = 50
    MAX_CHUNK_CHARS = CHUNK_SIZE * 4
    
    # =============================================================================
    # VECTOR SEARCH SETTINGS
    # =============================================================================
    
    TOP_K = 5
    TOP_K_SUMMARY = 15
    SIMILARITY_THRESHOLD = 0.3
    EMBEDDING_DIMENSION = 1536
    
    # =============================================================================
    # OCR SETTINGS
    # =============================================================================
    
    OCR_ENGINE = "easyocr"
    OCR_LANGUAGES = ['en']
    TESSERACT_CONFIG = '--oem 3 --psm 6'
    OCR_CONFIDENCE_THRESHOLD = 0.5
    
    # =============================================================================
    # PDF PROCESSING SETTINGS
    # =============================================================================
    
    EXTRACT_IMAGES = True
    MIN_IMAGE_SIZE = (100, 100)
    IMAGE_DPI = 150
    IGNORE_HEADERS_FOOTERS = True
    MIN_TEXT_LENGTH = 10
    
    # =============================================================================
    # UI SETTINGS
    # =============================================================================
    
    GRADIO_THEME = "soft"
    GRADIO_PORT = 7860
    GRADIO_HOST = "127.0.0.1"
    MAX_HISTORY_LENGTH = 50
    
    # =============================================================================
    # SYSTEM PROMPTS
    # =============================================================================
    
    SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on a training manual. 

IMPORTANT RULES:
1. Only answer based on the provided context from the training manual
2. If the context doesn't contain relevant information, say "I don't have information about that in the training manual"
3. Always cite which section or page the information comes from when possible
4. Be concise but comprehensive in your answers
5. If there are related images or diagrams mentioned, reference them in your response

Format your responses clearly with:
- Direct answer to the question
- Supporting details from the manual
- Source references where applicable
"""
    
    RAG_PROMPT_TEMPLATE = """Based on the following context from the training manual, please answer the user's question.

Context:
{context}

Question: {question}
"""
    
    SUMMARY_PROMPT_TEMPLATE = """Based on the following sections from the training manual, provide a comprehensive summary that covers the main points and key information.

Sections from manual:
{context}

Question: {question}

Please provide a well-structured summary that:
1. Covers the main topics and key points
2. Is organized logically
3. Includes important details without being overwhelming
4. References the source sections when relevant

Summary:
"""
    
    IMAGE_CONTEXT_PROMPT = """Based on the following context from the training manual and the image description, please answer the user's question.

Text Context:
{context}

Image Information:
{image_info}

Question: {question}

Answer based on both the text context and image information provided:
"""
    
    # =============================================================================
    # LOGGING SETTINGS
    # =============================================================================
    
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # =============================================================================
    # PERFORMANCE SETTINGS
    # =============================================================================
    
    EMBEDDING_BATCH_SIZE = 100
    USE_MEMORY_MAPPING = True
    N_JOBS = None
    
    # =============================================================================
    # INIT
    # =============================================================================
    
    def __init__(self):
        self.validate_config()
        self.ensure_directories()
    
    def validate_config(self):
        if not self.OPENAI_API_KEY:
            print("⚠️  WARNING: OPENAI_API_KEY not found in environment variables")
            print("   Set it with: export OPENAI_API_KEY='your-key-here'")
        
        if self.CHUNK_SIZE < 50:
            raise ValueError("CHUNK_SIZE must be at least 50 tokens")
        
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP must be less than CHUNK_SIZE")
        
        if not 0 <= self.SIMILARITY_THRESHOLD <= 1:
            raise ValueError("SIMILARITY_THRESHOLD must be between 0 and 1")
    
    def ensure_directories(self):
        for directory in [self.DATA_DIR, self.IMAGES_DIR, self.DOCUMENTS_DIR]:
            Path(directory).mkdir(exist_ok=True)
    
    def get_embedding_settings(self):
        return {
            'model': self.EMBEDDING_MODEL,
            'dimension': self.EMBEDDING_DIMENSION,
            'batch_size': self.EMBEDDING_BATCH_SIZE
        }
    
    def get_ocr_settings(self):
        return {
            'engine': self.OCR_ENGINE,
            'languages': self.OCR_LANGUAGES,
            'confidence_threshold': self.OCR_CONFIDENCE_THRESHOLD,
            'tesseract_config': self.TESSERACT_CONFIG
        }
    
    def get_pdf_settings(self):
        return {
            'extract_images': self.EXTRACT_IMAGES,
            'min_image_size': self.MIN_IMAGE_SIZE,
            'image_dpi': self.IMAGE_DPI,
            'ignore_headers_footers': self.IGNORE_HEADERS_FOOTERS,
            'min_text_length': self.MIN_TEXT_LENGTH
        }

# Create global config instance
config = Config()
