"""
Text utilities for RAG Chatbot
Handles text chunking, cleaning, and tokenization
"""

import re
import logging
from typing import List, Optional
from config import Config

logger = logging.getLogger(__name__)

class TextChunker:
    """Handles intelligent text chunking for better retrieval"""
    
    def __init__(self):
        self.config = Config()
        
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks optimized for retrieval
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []
        
        # Clean text first
        text = self._clean_text_for_chunking(text)
        
        # Try semantic chunking first (by paragraphs/sections)
        semantic_chunks = self._semantic_chunk(text)
        
        # If semantic chunking produces good results, use it
        if semantic_chunks and all(self._is_good_chunk_size(chunk) for chunk in semantic_chunks):
            return semantic_chunks
        
        # Otherwise, fall back to sliding window chunking
        return self._sliding_window_chunk(text)
    
    def _clean_text_for_chunking(self, text: str) -> str:
        """Clean text before chunking"""
        if not text:
            return ""
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive punctuation
        text = re.sub(r'([.!?]){2,}', r'\1', text)
        
        # Fix common OCR errors
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between words
        text = re.sub(r'(\d+)([A-Za-z])', r'\1 \2', text)  # Space between numbers and letters
        
        return text.strip()
    
    def _semantic_chunk(self, text: str) -> List[str]:
        """
        Chunk text based on semantic boundaries (paragraphs, sections)
        """
        chunks = []
        
        # Split by double newlines (paragraphs) first
        paragraphs = re.split(r'\n\s*\n', text)
        
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # Check if adding this paragraph would exceed chunk size
            potential_chunk = f"{current_chunk}\n\n{paragraph}".strip()
            
            if self._estimate_tokens(potential_chunk) <= self.config.CHUNK_SIZE:
                # Add paragraph to current chunk
                current_chunk = potential_chunk
            else:
                # Save current chunk if it's not empty
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Start new chunk with current paragraph
                if self._estimate_tokens(paragraph) <= self.config.CHUNK_SIZE:
                    current_chunk = paragraph
                else:
                    # Paragraph is too long, need to split it
                    para_chunks = self._sliding_window_chunk(paragraph)
                    chunks.extend(para_chunks[:-1])  # Add all but last
                    current_chunk = para_chunks[-1] if para_chunks else ""
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _sliding_window_chunk(self, text: str) -> List[str]:
        """
        Chunk text using sliding window approach with overlap
        """
        if not text:
            return []
        
        chunks = []
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return [text]  # Return original if no sentences found
        
        current_chunk = ""
        sentence_buffer = []
        
        for sentence in sentences:
            # Add sentence to buffer
            sentence_buffer.append(sentence)
            potential_chunk = " ".join(sentence_buffer)
            
            # Check if chunk is getting too large
            if self._estimate_tokens(potential_chunk) > self.config.CHUNK_SIZE:
                # Save current chunk (without the last sentence)
                if len(sentence_buffer) > 1:
                    chunk_text = " ".join(sentence_buffer[:-1])
                    if chunk_text.strip():
                        chunks.append(chunk_text.strip())
                    
                    # Calculate overlap
                    overlap_sentences = self._calculate_overlap_sentences(sentence_buffer[:-1])
                    sentence_buffer = overlap_sentences + [sentence]
                else:
                    # Single sentence is too long, split it by words
                    word_chunks = self._chunk_by_words(sentence)
                    chunks.extend(word_chunks[:-1])
                    sentence_buffer = [word_chunks[-1]] if word_chunks else []
        
        # Add final chunk
        if sentence_buffer:
            final_chunk = " ".join(sentence_buffer).strip()
            if final_chunk:
                chunks.append(final_chunk)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting (can be improved with NLTK or spaCy)
        sentence_endings = r'[.!?]+(?:\s|$)'
        sentences = re.split(sentence_endings, text)
        
        # Clean and filter sentences
        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Minimum sentence length
                clean_sentences.append(sentence)
        
        return clean_sentences
    
    def _calculate_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Calculate which sentences to include in overlap"""
        if not sentences:
            return []
        
        overlap_text = ""
        overlap_sentences = []
        
        # Work backwards from the end to get recent context
        for sentence in reversed(sentences):
            potential_overlap = f"{sentence} {overlap_text}".strip()
            if self._estimate_tokens(potential_overlap) <= self.config.CHUNK_OVERLAP:
                overlap_text = potential_overlap
                overlap_sentences.insert(0, sentence)
            else:
                break
        
        return overlap_sentences
    
    def _chunk_by_words(self, text: str) -> List[str]:
        """Chunk text by words when sentences are too long"""
        words = text.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            chunk_text = " ".join(current_chunk)
            
            if self._estimate_tokens(chunk_text) > self.config.CHUNK_SIZE:
                # Save current chunk (without last word)
                if len(current_chunk) > 1:
                    chunks.append(" ".join(current_chunk[:-1]))
                    
                    # Calculate word overlap
                    overlap_size = max(1, self.config.CHUNK_OVERLAP // 4)  # Rough word estimate
                    overlap_start = max(0, len(current_chunk) - overlap_size - 1)
                    current_chunk = current_chunk[overlap_start:]
                else:
                    # Single word is somehow too long, just add it
                    chunks.append(word)
                    current_chunk = []
        
        # Add final chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Rough estimate of token count
        OpenAI typically uses ~4 characters per token for English
        """
        if not text:
            return 0
        
        # More accurate estimation considering:
        # - Whitespace doesn't count as much
        # - Common words are often single tokens
        # - Punctuation is usually separate tokens
        
        # Remove extra whitespace for counting
        clean_text = re.sub(r'\s+', ' ', text.strip())
        
        # Rough estimation: 4 characters per token, but adjust for common patterns
        base_estimate = len(clean_text) / 4
        
        # Adjust for punctuation (each punct mark ~ 1 token)
        punct_count = len(re.findall(r'[^\w\s]', clean_text))
        
        # Adjust for word boundaries
        word_count = len(clean_text.split())
        
        # Weighted average of different estimation methods
        final_estimate = (base_estimate * 0.6) + (word_count * 0.3) + (punct_count * 0.1)
        
        return max(1, int(final_estimate))
    
    def _is_good_chunk_size(self, chunk: str) -> bool:
        """Check if chunk is within acceptable size range"""
        token_count = self._estimate_tokens(chunk)
        min_size = max(50, self.config.CHUNK_SIZE * 0.5)  # At least 50% of target size
        max_size = self.config.CHUNK_SIZE * 1.2  # Allow 20% over target
        
        return min_size <= token_count <= max_size


class TextPreprocessor:
    """Additional text preprocessing utilities"""
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for better matching"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive punctuation
        text = re.sub(r'([.!?]){2,}', r'\1', text)
        
        # Fix common formatting issues
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)  # Fix hyphenated line breaks
        
        return text.strip()
    
    @staticmethod
    def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
        """Extract key terms from text (simple implementation)"""
        if not text:
            return []
        
        # Simple keyword extraction
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
            'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
        }
        
        # Extract words (alphanumeric sequences)
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out stop words and short words
        keywords = [word for word in words 
                   if word not in stop_words and len(word) > 3]
        
        # Count frequency
        word_counts = {}
        for word in keywords:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        sorted_keywords = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [word for word, count in sorted_keywords[:max_keywords]]
    
    @staticmethod
    def clean_query(query: str) -> str:
        """Clean user query for better retrieval"""
        if not query:
            return ""
        
        # Basic cleaning
        query = query.strip()
        
        # Remove excessive punctuation
        query = re.sub(r'([.!?]){2,}', r'\1', query)
        
        # Normalize whitespace
        query = re.sub(r'\s+', ' ', query)
        
        return query