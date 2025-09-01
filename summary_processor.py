"""
Summary Processor Module for Enhanced RAG Chatbot
Handles broad summary questions by retrieving and synthesizing multiple chunks
"""

import logging
from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter
import re

from config import Config
from openai import OpenAI

logger = logging.getLogger(__name__)

class SummaryProcessor:
    """Processes broad summary queries by retrieving and synthesizing multiple chunks"""
    
    def __init__(self):
        self.config = Config()
        self.client = OpenAI(api_key=self.config.OPENAI_API_KEY)
    
    def process_summary_query(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Process a summary query using multiple retrieved chunks
        
        Args:
            query: User's summary question
            chunks: Retrieved chunks from vector store
            
        Returns:
            Generated summary response
        """
        if not chunks:
            return "I don't have enough information to provide a summary. Could you try a more specific question?"
        
        logger.info(f"📊 Processing summary query with {len(chunks)} chunks")
        
        # Organize and prepare chunks for summarization
        organized_content = self._organize_chunks_for_summary(chunks)
        
        # Detect summary scope and type
        summary_info = self._analyze_summary_request(query, chunks)
        
        # Generate summary using different strategies based on content amount
        if len(chunks) > 10:
            summary = self._generate_hierarchical_summary(query, organized_content, summary_info)
        else:
            summary = self._generate_direct_summary(query, organized_content, summary_info)
        
        # Add source information
        summary_with_sources = self._add_comprehensive_sources(summary, chunks)
        
        return summary_with_sources
    
    def _organize_chunks_for_summary(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Organize chunks by page, type, and content for better summarization"""
        organized = {
            'by_page': defaultdict(list),
            'by_type': defaultdict(list),
            'all_content': [],
            'page_range': {'min': float('inf'), 'max': 0}
        }
        
        for chunk in chunks:
            page = chunk.get('page', 'Unknown')
            chunk_type = chunk.get('type', 'text')
            content = chunk.get('content', '')
            
            # Group by page
            organized['by_page'][page].append(chunk)
            
            # Group by type
            organized['by_type'][chunk_type].append(chunk)
            
            # All content for overall processing
            organized['all_content'].append({
                'content': content,
                'page': page,
                'type': chunk_type,
                'similarity': chunk.get('similarity_score', 0.0)
            })
            
            # Track page range
            if isinstance(page, (int, float)) or (isinstance(page, str) and page.isdigit()):
                page_num = int(page)
                organized['page_range']['min'] = min(organized['page_range']['min'], page_num)
                organized['page_range']['max'] = max(organized['page_range']['max'], page_num)
        
        # Clean up page range
        if organized['page_range']['min'] == float('inf'):
            organized['page_range'] = {'min': 1, 'max': 1}
        
        return organized
    
    def _analyze_summary_request(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze what kind of summary is being requested"""
        query_lower = query.lower()
        
        summary_info = {
            'scope': 'general',  # general, section, topic-specific
            'detail_level': 'medium',  # brief, medium, comprehensive
            'focus_areas': [],
            'requested_sections': []
        }
        
        # Determine scope
        if any(term in query_lower for term in ['entire', 'whole', 'complete', 'all']):
            summary_info['scope'] = 'comprehensive'
        elif any(term in query_lower for term in ['section', 'chapter', 'part']):
            summary_info['scope'] = 'section'
            # Try to extract section references
            section_matches = re.findall(r'\b(section|chapter|part)\s*(\d+|\w+)', query_lower)
            summary_info['requested_sections'] = [match[1] for match in section_matches]
        
        # Determine detail level
        if any(term in query_lower for term in ['brief', 'quickly', 'short']):
            summary_info['detail_level'] = 'brief'
        elif any(term in query_lower for term in ['detailed', 'comprehensive', 'thorough']):
            summary_info['detail_level'] = 'comprehensive'
        
        # Extract focus areas (topics mentioned in query)
        focus_keywords = self._extract_focus_keywords(query_lower, chunks)
        summary_info['focus_areas'] = focus_keywords
        
        return summary_info
    
    def _extract_focus_keywords(self, query: str, chunks: List[Dict[str, Any]]) -> List[str]:
        """Extract key topics/keywords that should be focused on in the summary"""
        # Common technical topics that might be in training manuals
        topic_keywords = [
            'safety', 'security', 'maintenance', 'installation', 'operation',
            'troubleshooting', 'configuration', 'setup', 'procedures', 'guidelines',
            'quality', 'testing', 'inspection', 'compliance', 'requirements',
            'equipment', 'tools', 'software', 'hardware', 'system'
        ]
        
        focus_areas = []
        query_words = query.split()
        
        # Find topic keywords in query
        for keyword in topic_keywords:
            if keyword in query:
                focus_areas.append(keyword)
        
        # Also look for domain-specific terms that appear frequently in chunks
        content_text = ' '.join([chunk.get('content', '') for chunk in chunks[:5]])  # Sample from top chunks
        content_words = content_text.lower().split()
        word_freq = Counter(content_words)
        
        # Find words that appear in both query and frequently in content
        for word in query_words:
            if len(word) > 4 and word_freq.get(word.lower(), 0) > 2:
                focus_areas.append(word.lower())
        
        return list(set(focus_areas))[:5]  # Limit to top 5 focus areas
    
    def _generate_direct_summary(self, query: str, organized_content: Dict[str, Any], 
                                summary_info: Dict[str, Any]) -> str:
        """Generate summary directly from chunks (for smaller content volumes)"""
        # Build context from organized content
        context_parts = []
        
        # Sort content by page for logical flow
        sorted_content = sorted(organized_content['all_content'], 
                               key=lambda x: (x['page'] if isinstance(x['page'], (int, float)) else 999, x['similarity']), 
                               reverse=True)
        
        for i, item in enumerate(sorted_content[:15], 1):  # Limit context size
            context_part = f"[Section {i} - Page {item['page']}]:\n{item['content']}"
            context_parts.append(context_part)
        
        context = '\n\n'.join(context_parts)
        
        # Customize prompt based on summary info
        prompt = self._build_summary_prompt(query, context, summary_info)
        
        # Generate summary
        return self._call_openai_for_summary(prompt)
    
    def _generate_hierarchical_summary(self, query: str, organized_content: Dict[str, Any], 
                                     summary_info: Dict[str, Any]) -> str:
        """Generate summary using hierarchical approach (for large content volumes)"""
        logger.info("📊 Using hierarchical summary approach for large content")
        
        # Step 1: Create mini-summaries for each page/section
        page_summaries = {}
        
        for page, page_chunks in organized_content['by_page'].items():
            if len(page_chunks) > 0:
                page_content = '\n\n'.join([chunk['content'] for chunk in page_chunks])
                page_summary = self._summarize_page_content(page_content, page, summary_info)
                page_summaries[page] = page_summary
        
        # Step 2: Combine page summaries into final summary
        combined_context = '\n\n'.join([
            f"[Page {page}]: {summary}" 
            for page, summary in page_summaries.items()
        ])
        
        # Generate final comprehensive summary
        final_prompt = f"""Based on the following page summaries from a training manual, provide a comprehensive summary that answers: {query}

Page Summaries:
{combined_context}

Please provide a well-organized summary that:
1. Covers the main topics and themes
2. Highlights key procedures and important information
3. Maintains logical flow between sections
4. Includes specific details where relevant

Comprehensive Summary:"""

        return self._call_openai_for_summary(final_prompt)
    
    def _summarize_page_content(self, content: str, page: Any, summary_info: Dict[str, Any]) -> str:
        """Create a mini-summary for a single page's content"""
        prompt = f"""Summarize the following content from page {page} of a training manual. Focus on key points, procedures, and important information.

Content:
{content[:2000]}  # Limit content size

Provide a concise summary (2-3 sentences) of the main points:"""

        try:
            response = self.client.chat.completions.create(
                model=self.config.MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates concise, accurate summaries."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,  # Keep page summaries short
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"❌ Error creating page summary: {str(e)}")
            return f"Content from page {page} covers important information related to the topic."
    
    def _build_summary_prompt(self, query: str, context: str, summary_info: Dict[str, Any]) -> str:
        """Build customized prompt for summary generation"""
        # Base prompt
        base_prompt = self.config.SUMMARY_PROMPT_TEMPLATE.format(
            context=context,
            question=query
        )
        
        # Add customizations based on summary info
        customizations = []
        
        if summary_info['detail_level'] == 'brief':
            customizations.append("Keep the summary concise and focus on the most important points only.")
        elif summary_info['detail_level'] == 'comprehensive':
            customizations.append("Provide a detailed summary that covers all important aspects and includes specific procedures and requirements.")
        
        if summary_info['focus_areas']:
            focus_list = ', '.join(summary_info['focus_areas'])
            customizations.append(f"Pay special attention to information about: {focus_list}")
        
        if summary_info['scope'] == 'section' and summary_info['requested_sections']:
            sections = ', '.join(summary_info['requested_sections'])
            customizations.append(f"Focus specifically on section(s): {sections}")
        
        if customizations:
            base_prompt += f"\n\nAdditional Instructions:\n" + '\n'.join(f"- {custom}" for custom in customizations)
        
        return base_prompt
    
    def _call_openai_for_summary(self, prompt: str) -> str:
        """Call OpenAI API to generate summary"""
        try:
            response = self.client.chat.completions.create(
                model=self.config.MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are an expert at creating clear, comprehensive summaries of technical documentation."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.MAX_TOKENS,
                temperature=0.1  # Low temperature for consistent, factual summaries
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"❌ Error generating summary: {str(e)}")
            return "I encountered an error while generating the summary. Please try a more specific question."
    
    def _add_comprehensive_sources(self, summary: str, chunks: List[Dict[str, Any]]) -> str:
        """Add comprehensive source information for summary responses"""
        if not chunks:
            return summary
        
        # Collect source information
        page_info = defaultdict(set)
        section_info = defaultdict(int)
        
        for chunk in chunks:
            page = chunk.get('page', 'Unknown')
            chunk_type = chunk.get('type', 'text')
            file_name = chunk.get('metadata', {}).get('file_name', 'Training Manual')
            
            page_info[file_name].add(str(page))
            section_info[chunk_type] += 1
        
        # Build sources section
        sources_parts = []
        
        # Pages covered
        for file_name, pages in page_info.items():
            sorted_pages = sorted(pages, key=lambda x: int(x) if x.isdigit() else 999)
            if len(sorted_pages) > 5:
                page_range = f"{sorted_pages[0]}-{sorted_pages[-1]}"
                sources_parts.append(f"📄 {file_name}, Pages {page_range} (and others)")
            else:
                page_list = ", ".join(sorted_pages)
                sources_parts.append(f"📄 {file_name}, Pages {page_list}")
        
        # Content types
        type_descriptions = {
            'text': 'text content',
            'image_ocr': 'images and diagrams'
        }
        
        content_types = []
        for chunk_type, count in section_info.items():
            type_desc = type_descriptions.get(chunk_type, chunk_type)
            content_types.append(f"{count} sections of {type_desc}")
        
        if content_types:
            sources_parts.append(f"📊 Content types: {', '.join(content_types)}")
        
        # Add coverage information
        total_chunks = len(chunks)
        unique_pages = len(set(chunk.get('page', 'Unknown') for chunk in chunks))
        sources_parts.append(f"📈 Coverage: {total_chunks} sections from {unique_pages} pages")
        
        sources_text = f"\n\n📚 **Summary Sources:**\n" + "\n".join(f"• {source}" for source in sources_parts)
        
        return f"{summary}{sources_text}"
    
    def get_summary_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about the content being summarized"""
        if not chunks:
            return {}
        
        stats = {
            'total_chunks': len(chunks),
            'unique_pages': len(set(chunk.get('page', 'Unknown') for chunk in chunks)),
            'content_types': Counter(chunk.get('type', 'text') for chunk in chunks),
            'avg_similarity': sum(chunk.get('similarity_score', 0) for chunk in chunks) / len(chunks),
            'page_range': {
                'min': min((chunk.get('page', 0) for chunk in chunks if isinstance(chunk.get('page'), (int, float))), default=0),
                'max': max((chunk.get('page', 0) for chunk in chunks if isinstance(chunk.get('page'), (int, float))), default=0)
            }
        }
        
        return stats


class SummaryFormatter:
    """Formats summary responses for better readability"""
    
    def __init__(self):
        self.config = Config()
    
    def format_summary_response(self, summary: str, query_analysis: Dict[str, Any]) -> str:
        """Format summary response based on query analysis"""
        if not summary:
            return summary
        
        # Add appropriate header
        summary_type = self._determine_summary_type(query_analysis)
        header = self._get_summary_header(summary_type)
        
        # Format the content
        formatted_summary = self._format_summary_content(summary)
        
        return f"{header}\n\n{formatted_summary}"
    
    def _determine_summary_type(self, query_analysis: Dict[str, Any]) -> str:
        """Determine the type of summary being provided"""
        query = query_analysis.get('original_query', '').lower()
        
        if 'entire' in query or 'whole' in query:
            return 'comprehensive'
        elif 'section' in query or 'chapter' in query:
            return 'section'
        elif 'overview' in query:
            return 'overview'
        else:
            return 'general'
    
    def _get_summary_header(self, summary_type: str) -> str:
        """Get appropriate header for summary type"""
        headers = {
            'comprehensive': '📋 **Comprehensive Manual Summary**',
            'section': '📑 **Section Summary**',
            'overview': '📖 **Overview**',
            'general': '📝 **Summary**'
        }
        
        return headers.get(summary_type, '📝 **Summary**')
    
    def _format_summary_content(self, summary: str) -> str:
        """Format the summary content for better readability"""
        # Add some basic formatting improvements
        # This is a simple implementation - could be enhanced further
        
        # Ensure proper paragraph breaks
        formatted = re.sub(r'\n{3,}', '\n\n', summary)
        
        # Add emphasis to key sections if they exist
        formatted = re.sub(r'\b(Key points?|Important|Note|Warning|Caution):', r'**\1:**', formatted)
        
        return formatted.strip()