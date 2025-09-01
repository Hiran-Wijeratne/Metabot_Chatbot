"""
Chat Engine Module for RAG Chatbot
Handles query processing, context retrieval, and response generation
"""

import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI
import json
from pathlib import Path

from config import Config
from vector_store import VectorStore
from text_utils import TextPreprocessor

logger = logging.getLogger(__name__)

class ChatEngine:
    """Main chat engine that orchestrates retrieval and generation"""
    
    def __init__(self, vector_store: VectorStore):
        self.config = Config()
        self.client = OpenAI(api_key=self.config.OPENAI_API_KEY)
        self.vector_store = vector_store
        self.conversation_history = []
        
    def generate_response(self, query: str) -> str:
        """
        Generate response for user query using RAG approach
        
        Args:
            query: User's question
            
        Returns:
            Generated response with sources
        """
        try:
            logger.info(f"🤔 Processing query: {query[:100]}...")
            
            # Clean and preprocess query
            cleaned_query = TextPreprocessor.clean_query(query)
            
            # Retrieve relevant context
            relevant_chunks = self.vector_store.search(cleaned_query, self.config.TOP_K)
            
            if not relevant_chunks:
                return "I don't have information about that in the training manual. Could you try rephrasing your question or asking about a different topic?"
            
            # Re-rank results for better relevance
            relevant_chunks = self.vector_store.rerank_results(relevant_chunks, cleaned_query)
            
            # Build context from retrieved chunks
            context = self._build_context(relevant_chunks)
            
            # Generate response using OpenAI
            response = self._generate_openai_response(cleaned_query, context)
            
            # Add source information
            response_with_sources = self._add_source_information(response, relevant_chunks)
            
            # Update conversation history
            self._update_history(query, response_with_sources)
            
            logger.info("✅ Response generated successfully")
            
            return response_with_sources
            
        except Exception as e:
            logger.error(f"❌ Error generating response: {str(e)}")
            return f"I apologize, but I encountered an error while processing your question. Please try again. Error: {str(e)}"
    
    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved chunks"""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            content = chunk['content']
            page = chunk.get('page', 'Unknown')
            chunk_type = chunk.get('type', 'text')
            
            # Format context with source information
            context_part = f"[Source {i} - Page {page}]:\n{content}"
            
            # Add image indicator if it's an OCR chunk
            if chunk_type == 'image_ocr':
                context_part = f"[Image Content - Page {page}]:\n{content}"
            
            context_parts.append(context_part)
        
        return "\n\n".join(context_parts)
    
    def _generate_openai_response(self, query: str, context: str) -> str:
        """Generate response using OpenAI GPT"""
        
        # Prepare messages for chat completion
        messages = [
            {
                "role": "system",
                "content": self.config.SYSTEM_PROMPT
            },
            {
                "role": "user", 
                "content": self.config.RAG_PROMPT_TEMPLATE.format(
                    context=context,
                    question=query
                )
            }
        ]
        
        try:
            # Generate response
            response = self.client.chat.completions.create(
                model=self.config.MODEL_NAME,
                messages=messages,
                max_tokens=self.config.MAX_TOKENS,
                temperature=self.config.TEMPERATURE,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"❌ OpenAI API error: {str(e)}")
            raise
    
    def _add_source_information(self, response: str, chunks: List[Dict[str, Any]]) -> str:
        """Add source references to the response"""
        if not chunks:
            return response
        
        # Build sources section
        sources = []
        unique_pages = set()
        
        for chunk in chunks:
            page = chunk.get('page', 'Unknown')
            chunk_type = chunk.get('type', 'text')
            file_name = chunk.get('metadata', {}).get('file_name', 'Training Manual')
            
            if page not in unique_pages:
                unique_pages.add(page)
                source_type = "📄" if chunk_type == 'text' else "🖼️"
                sources.append(f"{source_type} {file_name}, Page {page}")
        
        if sources:
            sources_text = "\n\n📚 **Sources:**\n" + "\n".join(f"• {source}" for source in sources[:5])
            return f"{response}{sources_text}"
        
        return response
    
    def _update_history(self, query: str, response: str) -> None:
        """Update conversation history"""
        self.conversation_history.append({
            'query': query,
            'response': response,
            'timestamp': self._get_timestamp()
        })
        
        # Keep history within limits
        if len(self.conversation_history) > self.config.MAX_HISTORY_LENGTH:
            self.conversation_history.pop(0)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history"""
        return self.conversation_history.copy()
    
    def clear_history(self) -> None:
        """Clear conversation history"""
        self.conversation_history.clear()
        logger.info("🗑️ Conversation history cleared")
    
    def save_history(self, file_path: str = None) -> None:
        """Save conversation history to file"""
        if file_path is None:
            file_path = Path(self.config.DATA_DIR) / "conversation_history.json"
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Conversation history saved to {file_path}")
        except Exception as e:
            logger.error(f"❌ Error saving history: {str(e)}")
    
    def load_history(self, file_path: str = None) -> bool:
        """Load conversation history from file"""
        if file_path is None:
            file_path = Path(self.config.DATA_DIR) / "conversation_history.json"
        
        try:
            if Path(file_path).exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.conversation_history = json.load(f)
                logger.info(f"📂 Conversation history loaded from {file_path}")
                return True
        except Exception as e:
            logger.error(f"❌ Error loading history: {str(e)}")
        
        return False


class QueryProcessor:
    """Advanced query processing and enhancement"""
    
    def __init__(self):
        self.config = Config()
    
    def preprocess_query(self, query: str) -> str:
        """Preprocess and enhance user query"""
        if not query:
            return ""
        
        # Basic cleaning
        cleaned = TextPreprocessor.clean_query(query)
        
        # Expand abbreviations (domain-specific)
        cleaned = self._expand_abbreviations(cleaned)
        
        # Add context keywords if needed
        enhanced = self._add_context_keywords(cleaned)
        
        return enhanced
    
    def _expand_abbreviations(self, query: str) -> str:
        """Expand common abbreviations in the domain"""
        # This would be customized based on your specific training manual
        abbreviations = {
            'qa': 'quality assurance',
            'qc': 'quality control',
            'sop': 'standard operating procedure',
            'faq': 'frequently asked questions',
            'ops': 'operations',
            'mgmt': 'management',
            'maint': 'maintenance',
            'equip': 'equipment',
            'proc': 'procedure',
            'std': 'standard'
        }
        
        words = query.split()
        expanded_words = []
        
        for word in words:
            word_lower = word.lower().strip('.,!?')
            if word_lower in abbreviations:
                expanded_words.append(abbreviations[word_lower])
            else:
                expanded_words.append(word)
        
        return ' '.join(expanded_words)
    
    def _add_context_keywords(self, query: str) -> str:
        """Add relevant context keywords to improve retrieval"""
        # This is a simple implementation - could be enhanced with ML models
        context_mappings = {
            'safety': ['procedure', 'protocol', 'guidelines'],
            'troubleshoot': ['problem', 'issue', 'error', 'fix'],
            'maintenance': ['schedule', 'procedure', 'checklist'],
            'quality': ['control', 'assurance', 'standards'],
            'installation': ['setup', 'configuration', 'instructions']
        }
        
        query_lower = query.lower()
        additional_keywords = []
        
        for trigger, keywords in context_mappings.items():
            if trigger in query_lower:
                additional_keywords.extend(keywords)
        
        if additional_keywords:
            # Add keywords that aren't already in the query
            new_keywords = [kw for kw in additional_keywords if kw not in query_lower]
            if new_keywords:
                return f"{query} {' '.join(new_keywords[:2])}"  # Add max 2 keywords
        
        return query
    
    def extract_intent(self, query: str) -> str:
        """Extract user intent from query (simple rule-based)"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['how to', 'how do', 'steps', 'procedure']):
            return 'how_to'
        elif any(word in query_lower for word in ['what is', 'define', 'definition', 'meaning']):
            return 'definition'
        elif any(word in query_lower for word in ['troubleshoot', 'problem', 'error', 'issue', 'fix']):
            return 'troubleshooting'
        elif any(word in query_lower for word in ['list', 'show', 'all', 'types']):
            return 'listing'
        elif any(word in query_lower for word in ['where', 'location', 'find']):
            return 'location'
        elif any(word in query_lower for word in ['when', 'schedule', 'frequency']):
            return 'timing'
        else:
            return 'general'


class ResponseFormatter:
    """Format responses for better readability"""
    
    def __init__(self):
        self.config = Config()
    
    def format_response(self, response: str, intent: str = 'general') -> str:
        """Format response based on intent and content"""
        if not response:
            return response
        
        # Add appropriate formatting based on intent
        if intent == 'how_to':
            return self._format_procedure(response)
        elif intent == 'listing':
            return self._format_list(response)
        elif intent == 'troubleshooting':
            return self._format_troubleshooting(response)
        else:
            return self._format_general(response)
    
    def _format_procedure(self, response: str) -> str:
        """Format procedural responses with clear steps"""
        # Look for numbered steps or bullet points
        if any(marker in response for marker in ['1.', '2.', '•', '-']):
            return f"📋 **Procedure:**\n\n{response}"
        else:
            return f"📋 **Instructions:**\n\n{response}"
    
    def _format_list(self, response: str) -> str:
        """Format list responses"""
        return f"📝 **List:**\n\n{response}"
    
    def _format_troubleshooting(self, response: str) -> str:
        """Format troubleshooting responses"""
        return f"🔧 **Troubleshooting:**\n\n{response}"
    
    def _format_general(self, response: str) -> str:
        """Format general responses"""
        return response
    
    def add_helpful_tips(self, response: str, chunks: List[Dict[str, Any]]) -> str:
        """Add helpful tips based on retrieved content"""
        tips = []
        
        # Analyze chunks for additional helpful information
        for chunk in chunks:
            content = chunk['content'].lower()
            
            if 'warning' in content or 'caution' in content:
                tips.append("⚠️ Please note any warnings or cautions mentioned in the manual.")
            
            if 'see also' in content or 'refer to' in content:
                tips.append("📖 Check for cross-references to related sections.")
            
            if any(word in content for word in ['image', 'figure', 'diagram', 'chart']):
                tips.append("🖼️ Visual aids may be available - check the referenced pages.")
        
        # Remove duplicates and limit tips
        unique_tips = list(set(tips))[:2]
        
        if unique_tips:
            tips_text = "\n\n💡 **Tips:**\n" + "\n".join(f"• {tip}" for tip in unique_tips)
            return f"{response}{tips_text}"
        
        return response