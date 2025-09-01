# 🤖 RAG Chatbot for PDF Training Manual

A sophisticated Retrieval-Augmented Generation (RAG) chatbot that can answer questions about your PDF training manual using OpenAI's GPT models and vector search.

## ✨ Features

- **📄 Comprehensive PDF Processing**: Extract text and images with OCR support
- **🔍 Smart Vector Search**: Uses FAISS for fast similarity search with OpenAI embeddings
- **🤖 AI-Powered Responses**: GPT-3.5-turbo for cost-effective, accurate answers
- **🖼️ Image Text Extraction**: OCR support for diagrams and images in PDFs
- **💬 User-Friendly Interface**: Clean Gradio web interface
- **💾 Persistent Storage**: Process PDF once, reuse embeddings
- **📊 Source Attribution**: Shows which pages/sections were used for answers
- **⚡ Optimized for Cost**: Uses most economical OpenAI models

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Git (optional)

### 2. Installation

```bash
# Clone or download the project
git clone <repository-url>
cd rag-chatbot

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

1. **Set your OpenAI API Key**:
   ```bash
   # Linux/Mac
   export OPENAI_API_KEY="your-openai-api-key-here"
   
   # Windows Command Prompt
   set OPENAI_API_KEY=your-openai-api-key-here
   
   # Windows PowerShell
   $env:OPENAI_API_KEY="your-openai-api-key-here"
   ```

2. **Add your PDF**:
   - Create a `documents/` folder in the project directory
   - Place your PDF training manual in the `documents/` folder
   - Update the `PDF_PATH` in `config.py` to point to your file

### 4. Run the Application

```bash
python app.py
```

The chatbot will:
1. Start processing your PDF (first time only)
2. Create embeddings and build the search index
3. Launch a web interface at `http://127.0.0.1:7860`

## 📁 Project Structure

```
rag-chatbot/
├── app.py                 # Main application entry point
├── config.py             # Configuration settings
├── pdf_processor.py      # PDF text and image extraction
├── vector_store.py       # FAISS vector database management
├── chat_engine.py        # Query processing and response generation
├── text_utils.py         # Text chunking and preprocessing utilities
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── documents/           # Place your PDF files here
├── data/               # Generated embeddings and metadata
└── images/             # Extracted images from PDFs
```

## ⚙️ Configuration Options

Edit `config.py` to customize:

- **PDF_PATH**: Path to your training manual
- **CHUNK_SIZE**: Text chunk size in tokens (default: 250)
- **TOP_K**: Number of relevant chunks to retrieve (default: 5)
- **MODEL_NAME**: OpenAI model to use (default: gpt-3.5-turbo)
- **OCR_ENGINE**: Choose between 'easyocr' or 'tesseract'

## 💡 Usage Tips

### Asking Effective Questions

✅ **Good questions**:
- "What are the safety procedures for equipment maintenance?"
- "How do I troubleshoot network connectivity issues?"
- "What is the quality control checklist for product testing?"
- "Summarize the emergency response procedures"

❌ **Less effective questions**:
- "Tell me everything"
- "What's in chapter 1?"
- Questions about topics not in your PDF

### Features

- **Source References**: Each answer shows which pages were used
- **Image Content**: OCR extracts text from diagrams and images
- **Context Awareness**: Related information is automatically included
- **Conversation History**: Previous questions are remembered during the session

## 🔧 Advanced Setup

### OCR Configuration

The system supports two OCR engines:

1. **EasyOCR** (recommended):
   - More accurate
   - Automatic model download
   - No additional setup required

2. **Tesseract**:
   - Faster processing
   - Requires separate installation:
     ```bash
     # Ubuntu/Debian
     sudo apt install tesseract-ocr
     
     # macOS
     brew install tesseract
     
     # Windows
     # Download from: https://github.com/UB-Mannheim/tesseract/wiki
     ```

### GPU Acceleration (Optional)

For faster processing with large documents:

```bash
# Replace faiss-cpu with faiss-gpu in requirements.txt
pip uninstall faiss-cpu
pip install faiss-gpu
```

### Environment Variables

Create a `.env` file for persistent configuration:

```env
OPENAI_API_KEY=your-api-key-here
PDF_PATH=documents/my_manual.pdf
MODEL_NAME=gpt-3.5-turbo
CHUNK_SIZE=250
```

## 📊 System Requirements

- **RAM**: 4GB minimum (8GB+ recommended for large PDFs)
- **Storage**: 1-2GB for embeddings (depends on PDF size)
- **Network**: Internet connection for OpenAI API calls

## 🔍 Troubleshooting

### Common Issues

1. **"No module named 'X'" Error**:
   ```bash
   pip install -r requirements.txt --upgrade
   ```

2. **OpenAI API Error**:
   - Verify your API key is set correctly
   - Check your OpenAI account has credits
   - Ensure API key has correct permissions

3. **PDF Processing Fails**:
   - Verify PDF file exists and is readable
   - Try a different PDF to test
   - Check PDF isn't password protected

4. **OCR Not Working**:
   - Install system OCR dependencies
   - Switch between EasyOCR and Tesseract in config
   - Check image quality in PDF

5. **Out of Memory**:
   - Reduce `CHUNK_SIZE` in config
   - Process smaller PDF sections
   - Increase system RAM

### Performance Optimization

- **Large PDFs**: Increase `EMBEDDING_BATCH_SIZE` if you have good internet
- **Slow OCR**: Disable image extraction by setting `EXTRACT_IMAGES = False`
- **Better Accuracy**: Increase `TOP_K` to retrieve more context

## 💰 Cost Optimization

The system is designed to be cost-effective:

- **Embeddings**: Uses `text-embedding-3-small` (cheapest option)
- **Chat Model**: Uses `gpt-3.5-turbo` instead of GPT-4
- **Caching**: Embeddings are created once and reused
- **Batch Processing**: Efficient API usage

Estimated costs for a 90-page manual:
- Initial processing: ~$0.50-2.00 (one-time)
- Per query: ~$0.001-0.01

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review the system logs for error messages
3. Ensure all requirements are installed correctly
4. Verify your OpenAI API key is valid

## 🔄 Updates

To update the system:

```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

## 🎯 Roadmap

Future enhancements:
- [ ] Support for multiple PDF files
- [ ] Advanced query expansion
- [ ] Conversation memory across sessions
- [ ] Export functionality for Q&A pairs
- [ ] Integration with other LLM providers
- [ ] Mobile-responsive interface

---

Made with ❤️ by Hiran Wijeratne for efficient knowledge retrieval from training manuals.
