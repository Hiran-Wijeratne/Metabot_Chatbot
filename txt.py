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
            
            ## 🆕 New Enhanced Features
            
            ### 📊 Summary Queries
            The chatbot now handles broad summary questions:
            - **"Summarize the entire manual"**
            - **"Give me an overview of safety procedures"**
            - **"What are the main points in section 3?"**
            - **"Brief summary of maintenance requirements"**
            
            ### 🖼️ Multimodal Support
            Ask about images and diagrams:
            - **"Show me the diagrams about troubleshooting"**
            - **"What images are related to installation?"**
            - **"Display figure 5 and explain it"**
            - **"Are there any charts about quality control?"**
            
            ### 🌟 Combined Queries
            Get comprehensive responses with both text and images:
            - **"Summarize safety procedures and show related diagrams"**
            - **"Overview of equipment setup with images"**
            
            ### 📝 Example Questions by Type
            
            **Standard Q&A:**
            - "How do I troubleshoot network issues?"
            - "What is the quality control process?"
            - "When should maintenance be performed?"
            
            **Summary Queries:**
            - "Summarize chapter 2"
            - "Give me an overview of all safety protocols"
            - "What are the key points about installation?"
            
            **Image Queries:**
            - "Show me diagrams related to wiring"
            - "Are there images of the control panel?"
            - "Display troubleshooting flowcharts"
            
            ### 🔧 Features
            - ✅ **Smart Query Analysis**: Automatically detects question type
            - ✅ **Context Retrieval**: Finds relevant sections automatically
            - ✅ **Image Integration**: Shows images with captions and OCR text
            - ✅ **Comprehensive Summaries**: Synthesizes information from multiple sections  
            - ✅ **Source Attribution**: Shows which parts of the manual were used
            - ✅ **Cost Optimized**: Uses GPT-3.5-turbo and efficient embeddings
            - ✅ **Persistent Storage**: Processes PDF once, reuses embeddings
            
            ### 💡 Tips for Best Results
            - **Be specific** in your questions for better results
            - **Use keywords** like "summarize", "overview", "show me", "images"
            - **Ask about sections** by name or number if known
            - **Combine requests** like "summarize X and show related images"
            - Check the System Info tab to verify everything is set up correctly
            
            ### 🎯 Query Types Detected
            - **📝 Standard Q&A**: Specific questions with targeted answers
            - **📊 Summary**: Broad overviews and comprehensive summaries  
            - **🖼️ Multimodal**: Image-related queries with visual content
            - **🌟 Comprehensive**: Combined text summaries with relevant images
            """)
    
    return demo