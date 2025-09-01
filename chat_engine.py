"""
Chat Engine Module for RAG Chatbot
Handles query processing, context retrieval, and response generation
Enhanced with summary processing and multimodal support
"""

import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI
import json
from pathlib import Path

from config import Config
from vector_store import VectorStore
from text_utils import TextPreprocessor
from query_analyzer import QueryAnalyzer, QueryType, QueryEnhancer
from summary_processor import SummaryProcessor, SummaryFormatter
from image_processor import ImageProcessor

logger = logging.getLogger(__name__)

class ChatEngine:
    """Enhanced chat engine with summary and multimodal capabilities"""
    
    def __init__(self, vector_store: VectorStore):
        self.config = Config()
        self.client = OpenAI(api_key=self.config.OPENAI_API_KEY)
        self.vector_store = vector_store
        self.conversation_history = []
        
        # Initialize enhanced components
        self.query_analyzer = QueryAnalyzer()
        self.query_enhancer = QueryEnhancer()
        self.summary_processor = SummaryProcessor()
        self.summary_formatter = SummaryFormatter()
        self.image_processor = ImageProcessor()
        
    def generate_response(self, query: str) -> str:
        """
        Enhanced response generation with summary and multimodal support
        
        Args:
            query: User's question
            
        Returns:
            Generated response with sources and possibly images
        """
        try:
            logger.info(f"🤔 Processing enhanced query: {query[:100]}...")
            
            # Analyze query to determine response strategy
            query_analysis = self.query_analyzer.analyze_query(query)
            logger.info(f"🔍 Query type detected: {query_analysis['type'].value} (confidence: {query_analysis['confidence']:.2f})")
            
            # Enhance query based on analysis
            enhanced_query = self._enhance_query(query, query_analysis)
            
            # Get retrieval parameters based on query type
            retrieval_params = self.query_analyzer.get_retrieval_params(query_analysis)
            
            # Retrieve relevant context
            relevant_chunks = self.vector_store.search(enhanced_query, retrieval_params['top_k'])
            
            if not relevant_chunks:
                return self._handle_no_results(query_analysis)
            
            # Re-rank results for better relevance
            relevant_chunks = self.vector_store.rerank_results(relevant_chunks, enhanced_query)
            
            # Generate response based on query type
            response = self._generate_response_by_type(query, query_analysis, relevant_chunks)
            
            # Update conversation history
            self._update_history(query, response)
            
            logger.info("✅ Enhanced response generated successfully")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error generating response: {str(e)}")
            return f"I apologize, but I encountered an error while processing your question. Please try again. Error: {str(e)}"
    
    def _enhance_query(self, query: str, analysis: Dict[str, Any]) -> str:
        """Enhance query based on analysis results"""
        query_type = analysis['type']
        
        if query_type == QueryType.SUMMARY:
            return self.query_enhancer.enhance_query_for_summary(query, analysis['metadata'])
        elif query_type in [QueryType.MULTIMODAL, QueryType.SUMMARY_WITH_IMAGES]:
            return self.query_enhancer.enhance_query_for_images(query, analysis['metadata'])
        else:
            return TextPreprocessor.clean_query(query)
    
    def _generate_response_by_type(self, query: str, analysis: Dict[str, Any], 
                                 chunks: List[Dict[str, Any]]) -> str:
        """Generate response based on detected query type"""
        query_type = analysis['type']
        
        if query_type == QueryType.SUMMARY:
            return self._generate_summary_response(query, analysis, chunks)
        
        elif query_type == QueryType.MULTIMODAL:
            return self._generate_multimodal_response(query, analysis, chunks)
        
        elif query_type == QueryType.SUMMARY_WITH_IMAGES:
            return self._generate_comprehensive_response(query, analysis, chunks)
        
        else:  # QueryType.QA
            return self._generate_standard_response(query, chunks)
    
    def _generate_summary_response(self, query: str, analysis: Dict[str, Any], 
                                 chunks: List[Dict[str, Any]]) -> str:
        """Generate response for summary queries"""
        logger.info("📊 Generating summary response")
        
        # Process summary using the summary processor
        summary = self.summary_processor.process_summary_query(query, chunks)
        
        # Format the summary
        formatted_summary = self.summary_formatter.format_summary_response(summary, analysis)
        
        return formatted_summary
    
    def _generate_multimodal_response(self, query: str, analysis: Dict[str, Any], 
                                    chunks: List[Dict[str, Any]]) -> str:
        """Generate response for image-related queries"""
        logger.info("🖼️ Generating multimodal response")
        
        # Find relevant images
        relevant_images = self.image_processor.find_relevant_images(chunks, query)
        
        if not relevant_images:
            # No images found, fall back to standard response
            return self._generate_standard_response(query, chunks)
        
        # Build context from text chunks
        text_context = self._build_context(chunks)
        
        # Get image context for LLM
        image_context = self.image_processor.get_image_context_for_llm(relevant_images)
        
        # Generate response with both text and image context
        response = self._generate_openai_response_with_images(query, text_context, image_context)
        
        # Format images for display
        formatted_images = self.image_processor.format_images_for_response(relevant_images)
        
        # Combine response with images
        if formatted_images:
            response = f"{response}\n\n{formatted_images}"
        
        # Add source information
        response_with_sources = self._add_source_information(response, chunks)
        
        return response_with_sources
    
    def _generate_comprehensive_response(self, query: str, analysis: Dict[str, Any], 
                                       chunks: List[Dict[str, Any]]) -> str:
        """Generate comprehensive response with both summary and images"""
        logger.info("🌟 Generating comprehensive response with summary and images")
        
        # Generate summary
        summary = self.summary_processor.process_summary_query(query, chunks)
        
        # Find relevant images
        relevant_images = self.image_processor.find_relevant_images(chunks, query)
        
        # If images are found, integrate them
        if relevant_images:
            image_context = self.image_processor.get_image_context_for_llm(relevant_images)
            
            # Enhance summary with image information
            enhanced_prompt = f"""Based on the summary below and the additional image information, provide a comprehensive response to: {query}

Summary:
{summary}

Image Information:
{image_context}

Please integrate the visual information with the text summary where relevant:"""

            try:
                enhanced_response = self._call_openai_simple(enhanced_prompt)
                
                # Format images for display
                formatted_images = self.image_processor.format_images_for_response(relevant_images)
                
                if formatted_images:
                    enhanced_response = f"{enhanced_response}\n\n{formatted_images}"
                
                return enhanced_response
                
            except Exception as e:
                logger.error(f"❌ Error enhancing summary with images: {str(e)}")
                # Fall back to summary only
        
        # Format the summary
        formatted_summary = self.summary_formatter.format_summary_response(summary, analysis)
        
        return formatted_summary
    
    def _generate_standard_response(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """Generate standard QA response (original functionality)"""
        # Build context from retrieved chunks
        context = self._build_context(chunks)
        
        # Generate response using OpenAI
        response = self._generate_openai_response(query, context)
        
        # Add source information
        response_with_sources = self._add_source_information(response, chunks)
        
        return response_with_sources
    
    def _generate_openai_response_with_images(self, query: str, text_context: str, image_context: str) -> str:
        """Generate response using both text and image context"""
        
        # Use the image context prompt template
        prompt = self.config.IMAGE_CONTEXT_PROMPT.format(
            context=text_context,
            image_info=image_context,
            question=query
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.MODEL_NAME,
                messages=[
                    {"role": "system", "content": self.config.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.MAX_TOKENS,
                temperature=self.config.TEMPERATURE
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"❌ OpenAI API error in multimodal response: {str(e)}")
            raise
    
    def _call_openai_simple(self, prompt: str) -> str:
        """Simple OpenAI call for enhancement tasks"""
        try:
            response = self.client.chat.completions.create(
                model=self.config.MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides clear, accurate information."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.MAX_TOKENS,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"❌ Error in simple OpenAI call: {str(e)}")
            raise
    
    def _handle_no_results(self, analysis: Dict[str, Any]) -> str:
        """Handle cases where no relevant chunks are found"""
        query_type = analysis['type']
        
        if query_type == QueryType.SUMMARY:
            return "I don't have enough information in the training manual to provide a comprehensive summary on this topic. Could you try asking about a more specific aspect or section?"
        
        elif query_type in [QueryType.MULTIMODAL, QueryType.SUMMARY_WITH_IMAGES]:
            return "I don't have relevant images or information about that topic in the training manual. Could you try asking about a different topic or be more specific?"
        
        else:
            return "I don't have information about that in the training manual. Could you try rephrasing your question or asking about a different topic?"    
    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved chunks (original functionality)"""
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
        """Generate response using OpenAI GPT (original functionality)"""
        
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
        """Add source references to the response (original functionality)"""
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
        """Update conversation history (original functionality)"""
        self.conversation_history.append({
            'query': query,
            'response': response,
            'timestamp': self._get_timestamp()
        })
        
        # Keep history within limits
        if len(self.conversation_history) > self.config.MAX_HISTORY_LENGTH:
            self.conversation_history.pop(0)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp (original functionality)"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    # Original functionality methods (preserved)
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
    
    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get enhanced statistics about chat engine capabilities"""
        stats = {
            'conversation_length': len(self.conversation_history),
            'query_analyzer': {
                'available': True,
                'supported_types': [qt.value for qt in QueryType]
            },
            'summary_processor': {
                'available': True,
                'max_chunks': self.config.TOP_K_SUMMARY
            },
            'image_processor': {
                'available': self.config.ENABLE_IMAGE_RESPONSES,
                'max_images': self.config.MAX_IMAGES_PER_RESPONSE,
                'vision_model': self.config.VISION_MODEL
            }
        }
        
        return stats


class QueryProcessor:
    """Advanced query processing and enhancement (original functionality preserved)"""
    
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
    """Format responses for better readability (original functionality preserved)"""
    
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
        
        return response.   