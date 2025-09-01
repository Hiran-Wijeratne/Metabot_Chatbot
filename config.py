"""
Configuration settings for RAG Chatbot
Modify these settings according to your needs
"""

import os
from pathlib import Path

class Config:
    """Configuration class for RAG Chatbot"""
    
    # =============================================================================
    # FILE PATHS
    # =============================================================================
    
    # Path to your PDF training manual
    # Update this to point to your actual PDF file
    PDF_PATH = "documents/META-SCRUB 60_User Manual.pdf"
    
    # Data directories
    DATA_DIR = "data"
    IMAGES_DIR = "images" 
    DOCUMENTS_DIR = "documents"
    
    # =============================================================================
    # OPENAI SETTINGS
    # =============================================================================
    
    # OpenAI API settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Model for chat responses (cost-optimized)
    MODEL_NAME = "gpt-3.5-turbo"
    
    # Embedding model (cheapest option)
    EMBEDDING_MODEL = "text-embedding-3-small"
    
    # Maximum tokens for responses
    MAX_TOKENS = 1000
    
    # Temperature for response generation (0.0 = deterministic, 1.0 = creative)
    TEMPERATURE = 0.1
    
    # =============================================================================
    # TEXT PROCESSING SETTINGS
    # =============================================================================
    
    # Text chunking parameters
    CHUNK_SIZE = 250          # Target tokens per chunk (200-300 as requested)
    CHUNK_OVERLAP = 50        # Overlap between chunks to maintain context
    
    # Maximum characters per chunk (rough estimate: 4 chars per token)
    MAX_CHUNK_CHARS = CHUNK_SIZE * 4
    
    # =============================================================================
    # VECTOR SEARCH SETTINGS  
    # =============================================================================
    
    # Number of relevant chunks to retrieve for each query
    TOP_K = 5
    
    # Similarity threshold (0.0 to 1.0, higher = more similar)
    SIMILARITY_THRESHOLD = 0.3
    
    # Vector database settings
    EMBEDDING_DIMENSION = 1536  # Dimension for text-embedding-3-small
    
    # =============================================================================
    # OCR SETTINGS
    # =============================================================================
    
    # OCR engine preference: 'tesseract' or 'easyocr'
    OCR_ENGINE = "easyocr"  # easyocr is more accurate but slower
    
    # OCR languages (for EasyOCR)
    OCR_LANGUAGES = ['en']
    
    # Tesseract configuration (if using tesseract)
    TESSERACT_CONFIG = '--oem 3 --psm 6'
    
    # Minimum confidence threshold for OCR text
    OCR_CONFIDENCE_THRESHOLD = 0.5
    
    # =============================================================================
    # PDF PROCESSING SETTINGS
    # =============================================================================
    
    # Image extraction settings
    EXTRACT_IMAGES = True
    MIN_IMAGE_SIZE = (100, 100)  # Minimum width, height to process
    IMAGE_DPI = 150              # DPI for image extraction
    
    # Text extraction settings
    IGNORE_HEADERS_FOOTERS = True
    MIN_TEXT_LENGTH = 10         # Minimum characters for a text chunk
    
    # =============================================================================
    # UI SETTINGS
    # =============================================================================
    
    # Gradio interface settings
    GRADIO_THEME = "soft"
    GRADIO_PORT = 7860
    GRADIO_HOST = "127.0.0.1"
    
    # Chat settings
    MAX_HISTORY_LENGTH = 50  # Maximum chat history to maintain
    
    # =============================================================================
    # SYSTEM PROMPTS
    # =============================================================================
    
    SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on a user training manual. 

IMPORTANT RULES:
1. Only answer based on the provided context from the training manual
2. If the context doesn't contain relevant information, say "I don't have information about that in the user training manual"
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

Answer based only on the provided context. If the context doesn't contain relevant information, say so clearly."""
    
    # =============================================================================
    # LOGGING SETTINGS
    # =============================================================================
    
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # =============================================================================
    # PERFORMANCE SETTINGS
    # =============================================================================
    
    # Batch processing settings
    EMBEDDING_BATCH_SIZE = 100
    
    # Memory optimization
    USE_MEMORY_MAPPING = True
    
    # Parallel processing (set to None to use all CPU cores)
    N_JOBS = None
    
    def __init__(self):
        """Initialize configuration and validate settings"""
        self.validate_config()
        self.ensure_directories()
    
    def validate_config(self):
        """Validate configuration settings"""
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
        """Create necessary directories"""
        for directory in [self.DATA_DIR, self.IMAGES_DIR, self.DOCUMENTS_DIR]:
            Path(directory).mkdir(exist_ok=True)
    
    def get_embedding_settings(self):
        """Get embedding-related settings as a dictionary"""
        return {
            'model': self.EMBEDDING_MODEL,
            'dimension': self.EMBEDDING_DIMENSION,
            'batch_size': self.EMBEDDING_BATCH_SIZE
        }
    
    def get_ocr_settings(self):
        """Get OCR-related settings as a dictionary"""
        return {
            'engine': self.OCR_ENGINE,
            'languages': self.OCR_LANGUAGES,
            'confidence_threshold': self.OCR_CONFIDENCE_THRESHOLD,
            'tesseract_config': self.TESSERACT_CONFIG
        }
    
    def get_pdf_settings(self):
        """Get PDF processing settings as a dictionary"""
        return {
            'extract_images': self.EXTRACT_IMAGES,
            'min_image_size': self.MIN_IMAGE_SIZE,
            'image_dpi': self.IMAGE_DPI,
            'ignore_headers_footers': self.IGNORE_HEADERS_FOOTERS,
            'min_text_length': self.MIN_TEXT_LENGTH
        }

# Create global config instance
config = Config()