"""
Query Analyzer Module for Enhanced RAG Chatbot
Analyzes user queries to determine the type of response needed (QA, Summary, or Multimodal)
"""

import logging
import re
from typing import Dict, Any, List
from enum import Enum

from config import Config

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Enum for different types of queries"""
    QA = "qa"                    # Standard question-answer
    SUMMARY = "summary"          # Broad summary questions
    MULTIMODAL = "multimodal"    # Image-related queries
    SUMMARY_WITH_IMAGES = "summary_with_images"  # Summary that might include images

class QueryAnalyzer:
    """Analyzes user queries to determine appropriate response strategy"""
    
    def __init__(self):
        self.config = Config()
        
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze user query and return query information
        
        Args:
            query: User's question/query
            
        Returns:
            Dictionary containing query type, confidence, and metadata
        """
        if not query or not query.strip():
            return {
                'type': QueryType.QA,
                'confidence': 0.0,
                'metadata': {}
            }
        
        query_lower = query.lower().strip()
        
        # Check for summary indicators
        is_summary = self._detect_summary_query(query_lower)
        summary_confidence = self._calculate_summary_confidence(query_lower)
        
        # Check for image/visual indicators
        is_image_query = self._detect_image_query(query_lower)
        image_confidence = self._calculate_image_confidence(query_lower)
        
        # Determine primary query type
        query_type, confidence = self._determine_query_type(
            is_summary, summary_confidence,
            is_image_query, image_confidence
        )
        
        # Extract additional metadata
        metadata = self._extract_query_metadata(query_lower, query_type)
        
        return {
            'type': query_type,
            'confidence': confidence,
            'metadata': metadata,
            'original_query': query,
            'normalized_query': query_lower
        }
    
    def _detect_summary_query(self, query: str) -> bool:
        """Detect if query is asking for a summary/overview"""
        summary_patterns = [
            # Direct summary requests
            r'\b(summarize|summary|overview|outline)\b',
            r'\b(main points|key points|highlights)\b',
            r'\b(brief|briefly)\b',
            
            # Broad scope indicators
            r'\b(entire|whole|complete|all about|everything about)\b',
            r'\b(comprehensive|general information)\b',
            
            # Section/chapter summaries
            r'\b(section|chapter|part).*\b(summary|overview)\b',
            r'\bsummarize.*\b(section|chapter|part)\b',
            
            # "What is X about" patterns
            r'\bwhat is.*about\b',
            r'\bwhat does.*cover\b',
            r'\bwhat are.*main\b',
        ]
        
        for pattern in summary_patterns:
            if re.search(pattern, query):
                return True
        
        return False
    
    def _calculate_summary_confidence(self, query: str) -> float:
        """Calculate confidence score for summary query detection"""
        confidence = 0.0
        
        # Strong indicators
        strong_keywords = ['summarize', 'summary', 'overview', 'outline', 'entire', 'whole']
        for keyword in strong_keywords:
            if keyword in query:
                confidence += 0.3
        
        # Medium indicators
        medium_keywords = ['main points', 'key points', 'highlights', 'brief', 'all about']
        for keyword in medium_keywords:
            if keyword in query:
                confidence += 0.2
        
        # Weak indicators
        weak_keywords = ['everything', 'complete', 'comprehensive', 'general']
        for keyword in weak_keywords:
            if keyword in query:
                confidence += 0.1
        
        # Broad scope patterns
        if re.search(r'\bwhat (is|are|does).*\b(about|cover|include)\b', query):
            confidence += 0.2
        
        # Question length (longer questions often ask for summaries)
        if len(query.split()) > 8:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _detect_image_query(self, query: str) -> bool:
        """Detect if query is asking about images/visual content"""
        # Check for image-related keywords
        for keyword in self.config.IMAGE_KEYWORDS:
            if keyword in query:
                return True
        
        # Check for visual reference patterns
        visual_patterns = [
            r'\b(show me|display|view)\b',
            r'\b(visual|visually)\b',
            r'\b(see|look at|examine)\b.*\b(image|picture|diagram)\b',
            r'\b(figure|fig\.?)\s*\d+\b',
            r'\b(page|p\.?)\s*\d+.*\b(image|picture|diagram)\b'
        ]
        
        for pattern in visual_patterns:
            if re.search(pattern, query):
                return True
        
        return False
    
    def _calculate_image_confidence(self, query: str) -> float:
        """Calculate confidence score for image query detection"""
        confidence = 0.0
        
        # Count image keywords
        keyword_count = sum(1 for keyword in self.config.IMAGE_KEYWORDS if keyword in query)
        confidence += keyword_count * 0.2
        
        # Visual action words
        visual_actions = ['show', 'display', 'view', 'see', 'look', 'examine']
        for action in visual_actions:
            if action in query:
                confidence += 0.15
        
        # Figure/page references
        if re.search(r'\b(figure|fig\.?|page|p\.?)\s*\d+\b', query):
            confidence += 0.3
        
        return min(confidence, 1.0)
    
    def _determine_query_type(self, is_summary: bool, summary_conf: float, 
                            is_image: bool, image_conf: float) -> tuple:
        """Determine the primary query type and confidence"""
        
        # If both summary and image indicators are present
        if is_summary and is_image and summary_conf > 0.3 and image_conf > 0.3:
            return QueryType.SUMMARY_WITH_IMAGES, max(summary_conf, image_conf)
        
        # If strong summary indicators
        if is_summary and summary_conf > 0.5:
            return QueryType.SUMMARY, summary_conf
        
        # If strong image indicators
        if is_image and image_conf > 0.4:
            return QueryType.MULTIMODAL, image_conf
        
        # If moderate summary indicators
        if is_summary and summary_conf > 0.3:
            return QueryType.SUMMARY, summary_conf
        
        # If moderate image indicators
        if is_image and image_conf > 0.2:
            return QueryType.MULTIMODAL, image_conf
        
        # Default to standard QA
        return QueryType.QA, 1.0 - max(summary_conf, image_conf)
    
    def _extract_query_metadata(self, query: str, query_type: QueryType) -> Dict[str, Any]:
        """Extract additional metadata from the query"""
        metadata = {
            'word_count': len(query.split()),
            'has_question_mark': '?' in query,
            'detected_keywords': []
        }
        
        # Extract detected keywords based on query type
        if query_type in [QueryType.SUMMARY, QueryType.SUMMARY_WITH_IMAGES]:
            metadata['detected_keywords'] = [
                kw for kw in self.config.SUMMARY_KEYWORDS if kw in query
            ]
            
            # Try to extract what to summarize
            section_match = re.search(r'\b(section|chapter|part)\s*(\d+|\w+)\b', query)
            if section_match:
                metadata['target_section'] = section_match.group(0)
        
        if query_type in [QueryType.MULTIMODAL, QueryType.SUMMARY_WITH_IMAGES]:
            metadata['detected_keywords'].extend([
                kw for kw in self.config.IMAGE_KEYWORDS if kw in query
            ])
            
            # Try to extract figure/page references
            fig_match = re.search(r'\b(figure|fig\.?)\s*(\d+)\b', query)
            if fig_match:
                metadata['figure_reference'] = fig_match.group(0)
                
            page_match = re.search(r'\b(page|p\.?)\s*(\d+)\b', query)
            if page_match:
                metadata['page_reference'] = page_match.group(0)
        
        return metadata
    
    def should_use_summary_approach(self, analysis: Dict[str, Any]) -> bool:
        """Determine if query should use summary approach"""
        return analysis['type'] in [QueryType.SUMMARY, QueryType.SUMMARY_WITH_IMAGES]
    
    def should_include_images(self, analysis: Dict[str, Any]) -> bool:
        """Determine if response should include images"""
        return analysis['type'] in [QueryType.MULTIMODAL, QueryType.SUMMARY_WITH_IMAGES]
    
    def get_retrieval_params(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get retrieval parameters based on query analysis"""
        base_params = {
            'top_k': self.config.TOP_K,
            'include_images': False,
            'strategy': 'standard'
        }
        
        query_type = analysis['type']
        
        if query_type == QueryType.SUMMARY:
            base_params.update({
                'top_k': self.config.TOP_K_SUMMARY,
                'strategy': 'summary'
            })
        
        elif query_type == QueryType.MULTIMODAL:
            base_params.update({
                'include_images': True,
                'strategy': 'multimodal'
            })
        
        elif query_type == QueryType.SUMMARY_WITH_IMAGES:
            base_params.update({
                'top_k': self.config.TOP_K_SUMMARY,
                'include_images': True,
                'strategy': 'comprehensive'
            })
        
        return base_params


class QueryEnhancer:
    """Enhances queries for better retrieval"""
    
    def __init__(self):
        self.config = Config()
    
    def enhance_query_for_summary(self, query: str, metadata: Dict[str, Any]) -> str:
        """Enhance query for better summary retrieval"""
        enhanced = query
        
        # Add broad context terms for summary queries
        summary_terms = ['overview', 'main topics', 'key information', 'important points']
        
        # Add terms that aren't already in the query
        new_terms = [term for term in summary_terms if term not in query.lower()][:2]
        
        if new_terms:
            enhanced = f"{query} {' '.join(new_terms)}"
        
        return enhanced
    
    def enhance_query_for_images(self, query: str, metadata: Dict[str, Any]) -> str:
        """Enhance query for better image retrieval"""
        enhanced = query
        
        # Add visual context terms
        if 'figure_reference' not in metadata and 'page_reference' not in metadata:
            visual_terms = ['diagram', 'illustration', 'visual', 'figure']
            new_terms = [term for term in visual_terms if term not in query.lower()][:1]
            
            if new_terms:
                enhanced = f"{query} {' '.join(new_terms)}"
        
        return enhanced