#!/usr/bin/env python3
"""
RAG Chatbot for PDF Training Manual
Main application entry point

Requirements:
1. Set OPENAI_API_KEY environment variable
2. Install requirements: pip install -r requirements.txt
3. Run: python app.py
4. Place your PDF in the 'documents' folder and update PDF_PATH in config
"""

import os
import logging
from pathlib import Path
import gradio as gr
from typing import List, Tuple, Optional
import json

from pdf_processor import PDFProcessor
from vector_store import VectorStore
from chat_engine import ChatEngine
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGChatbot:
    """Main RAG Chatbot class that orchestrates all components"""
    
    def __init__(self):
        self.config = Config()
        self.pdf_processor = None
        self.vector_store = None
        self.chat_engine = None
        self.is_initialized = False
        
    def initialize(self, pdf_path: str = None) -> str:
        """Initialize the RAG system with PDF processing"""
        try:
            # Use default PDF path if none provided
            if pdf_path is None:
                pdf_path = self.config.PDF_PATH
                
            if not os.path.exists(pdf_path):
                return f"❌ Error: PDF file not found at {pdf_path}. Please check the path in config.py"
            
            logger.info("🚀 Initializing RAG Chatbot...")
            
            # Initialize components
            self.pdf_processor = PDFProcessor()
            self.vector_store = VectorStore()
            self.chat_engine = ChatEngine(self.vector_store)
            
            # Check if processed data already exists
            embeddings_path = Path("data/embeddings.faiss")
            metadata_path = Path("data/chunks_metadata.json")
            
            if embeddings_path.exists() and metadata_path.exists():
                logger.info("📂 Loading existing embeddings...")
                self.vector_store.load_embeddings()
                status = "✅ Loaded existing embeddings from cache"
            else:
                logger.info("📄 Processing PDF...")
                # Process PDF and extract content
                chunks = self.pdf_processor.process_pdf(pdf_path)
                
                if not chunks:
                    return "❌ Error: No content extracted from PDF"
                
                logger.info(f"📝 Extracted {len(chunks)} chunks from PDF")
                
                # Create embeddings and build vector store
                logger.info("🔄 Creating embeddings (this may take a few minutes)...")
                self.vector_store.build_index(chunks)
                status = f"✅ Successfully processed PDF and created {len(chunks)} embeddings"
            
            self.is_initialized = True
            logger.info("✅ RAG Chatbot initialization complete!")
            return status
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {str(e)}")
            return f"❌ Initialization failed: {str(e)}"
    
    def chat(self, message: str, history: List[List[str]]) -> Tuple[str, List[List[str]]]:
        """Process chat message and return response"""
        if not self.is_initialized:
            init_status = self.initialize()
            if "❌" in init_status:
                return init_status, history
        
        try:
            if not message.strip():
                return "Please enter a question.", history
            
            # Generate response using chat engine
            response = self.chat_engine.generate_response(message)
            
            # Add to history
            history.append([message, response])
            
            return "", history
            
        except Exception as e:
            error_msg = f"❌ Error processing your question: {str(e)}"
            logger.error(error_msg)
            history.append([message, error_msg])
            return "", history
    
    def get_system_info(self) -> str:
        """Get system information and status"""
        info = []
        info.append("🤖 **RAG Chatbot System Information**\n")
        
        # API Key status
        if os.getenv("OPENAI_API_KEY"):
            info.append("✅ OpenAI API Key: Set")
        else:
            info.append("❌ OpenAI API Key: Not set (required for operation)")
        
        # PDF status
        if os.path.exists(self.config.PDF_PATH):
            info.append(f"✅ PDF File: Found at {self.config.PDF_PATH}")
        else:
            info.append(f"❌ PDF File: Not found at {self.config.PDF_PATH}")
        
        # Embeddings status
        if Path("data/embeddings.faiss").exists():
            info.append("✅ Embeddings: Available")
            
            # Load metadata to get chunk count
            try:
                with open("data/chunks_metadata.json", 'r') as f:
                    metadata = json.load(f)
                info.append(f"📊 Total Chunks: {len(metadata)}")
            except:
                pass
        else:
            info.append("❌ Embeddings: Not created yet")
        
        # Configuration
        info.append(f"\n⚙️ **Configuration:**")
        info.append(f"- Model: {self.config.MODEL_NAME}")
        info.append(f"- Embedding Model: {self.config.EMBEDDING_MODEL}")
        info.append(f"- Chunk Size: {self.config.CHUNK_SIZE} tokens")
        info.append(f"- Chunk Overlap: {self.config.CHUNK_OVERLAP} tokens")
        info.append(f"- Top-K Results: {self.config.TOP_K}")
        
        return "\n".join(info)

def create_ui() -> gr.Interface:
    """Create Gradio interface"""
    
    chatbot_instance = RAGChatbot()
    
    with gr.Blocks(title="RAG Chatbot - PDF Training Manual", theme=gr.themes.Soft()) as demo:
        gr.HTML("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h1>🤖 RAG Chatbot - PDF Training Manual</h1>
            <p>Ask questions about your training manual and get AI-powered answers with source context!</p>
        </div>
        """)
        
        with gr.Tab("💬 Chat"):
            chatbot = gr.Chatbot(
                height=500,
                bubble_full_width=False,
                show_copy_button=True
            )
            
            msg = gr.Textbox(
                placeholder="Ask a question about your training manual...",
                label="Your Question",
                lines=2
            )
            
            with gr.Row():
                submit_btn = gr.Button("Send", variant="primary", scale=1)
                clear_btn = gr.Button("Clear Chat", scale=1)
            
            # Initialize on first load
            demo.load(
                fn=chatbot_instance.initialize,
                outputs=gr.Textbox(visible=False)
            )
            
            # Chat functionality
            msg.submit(
                fn=chatbot_instance.chat,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot]
            )
            
            submit_btn.click(
                fn=chatbot_instance.chat,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot]
            )
            
            clear_btn.click(
                fn=lambda: ([], ""),
                outputs=[chatbot, msg]
            )
        
        with gr.Tab("ℹ️ System Info"):
            info_output = gr.Markdown()
            refresh_btn = gr.Button("Refresh Info", variant="secondary")
            
            # Load info on tab open
            demo.load(
                fn=chatbot_instance.get_system_info,
                outputs=info_output
            )
            
            refresh_btn.click(
                fn=chatbot_instance.get_system_info,
                outputs=info_output
            )
        
        '''
        with gr.Tab("📖 Instructions"):
            gr.Markdown("""
            ## 🚀 Quick Start Guide
            
            ### 1. Setup (First Time Only)
            ```bash
            # Install dependencies
            pip install -r requirements.txt
            
            # Set your OpenAI API key
            export OPENAI_API_KEY="your-api-key-here"
            # On Windows: set OPENAI_API_KEY=your-api-key-here
            ```
            
            ### 2. Add Your PDF
            - Place your PDF file in the `documents/` folder
            - Update the `PDF_PATH` in `config.py` to point to your file
            - Or use the default: `documents/training_manual.pdf`
            
            ### 3. Start Chatting
            - The system will automatically process your PDF on first run
            - This creates embeddings (may take a few minutes for large PDFs)
            - Once complete, start asking questions!
            
            ### 📝 Example Questions
            - "What are the safety procedures mentioned in chapter 3?"
            - "How do I troubleshoot equipment failures?"
            - "Summarize the maintenance schedule"
            - "What are the key points about quality control?"
            
            ### 🔧 Features
            - ✅ **Smart Context Retrieval**: Finds relevant sections automatically
            - ✅ **Image OCR**: Extracts text from images and diagrams  
            - ✅ **Source Attribution**: Shows which parts of the manual were used
            - ✅ **Cost Optimized**: Uses GPT-3.5-turbo and efficient embeddings
            - ✅ **Persistent Storage**: Processes PDF once, reuses embeddings
            
            ### 💡 Tips
            - Be specific in your questions for better results
            - The system only answers based on your PDF content
            - Check the System Info tab to verify everything is set up correctly
            """)
       '''
    return demo

def main():
    """Main entry point"""
    print("🚀 Starting RAG Chatbot...")
    
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ WARNING: OPENAI_API_KEY environment variable not set!")
        print("Please set it before running:")
        print("   Linux/Mac: export OPENAI_API_KEY='your-key-here'")
        print("   Windows: set OPENAI_API_KEY=your-key-here")
        print("\nThe app will still start, but won't work without the API key.\n")
    
    # Ensure required directories exist
    Path("documents").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    Path("images").mkdir(exist_ok=True)
    
    # Create and launch UI
    demo = create_ui()
    
    print("✅ RAG Chatbot is ready!")
    print("📖 Open your browser and start asking questions about your training manual!")
    
    # Launch with sharing disabled for security (enable if needed)
  # Launch with LAN access + optional public share
    demo.launch(
        server_name="0.0.0.0",   # Bind to all network interfaces
        server_port=7860,
        share=True,              # Generate a public URL via Gradio
        show_error=True
    )


if __name__ == "__main__":
    main()