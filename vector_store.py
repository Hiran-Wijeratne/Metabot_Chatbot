"""
Vector Store Module for RAG Chatbot
Handles embeddings creation, storage, and similarity search using FAISS
"""

import numpy as np
import faiss
import logging
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import openai
from openai import OpenAI
import time

from config import Config
from text_utils import TextPreprocessor

logger = logging.getLogger(__name__)

class VectorStore:
    """Handles vector embeddings and similarity search using FAISS"""
    
    def __init__(self):
        self.config = Config()
        self.client = OpenAI(api_key=self.config.OPENAI_API_KEY)
        self.index = None
        self.chunks_metadata = []
        self.embeddings_path = Path(self.config.DATA_DIR) / "embeddings.faiss"
        self.metadata_path = Path(self.config.DATA_DIR) / "chunks_metadata.json"
        
    def build_index(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Build FAISS index from document chunks
        
        Args:
            chunks: List of document chunks with content and metadata
        """
        logger.info("🔨 Building vector index...")
        
        if not chunks:
            raise ValueError("No chunks provided for indexing")
        
        # Store metadata
        self.chunks_metadata = chunks
        
        # Extract text content for embedding
        texts = [chunk['content'] for chunk in chunks]
        
        # Create embeddings
        embeddings = self._create_embeddings(texts)
        
        # Build FAISS index
        self._build_faiss_index(embeddings)
        
        # Save index and metadata
        self._save_index()
        
        logger.info(f"✅ Vector index built with {len(chunks)} chunks")
    
    def _create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for a list of texts"""
        logger.info(f"🔄 Creating embeddings for {len(texts)} chunks...")
        
        all_embeddings = []
        batch_size = self.config.EMBEDDING_BATCH_SIZE
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            logger.info(f"📊 Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            try:
                # Create embeddings using OpenAI API
                response = self.client.embeddings.create(
                    model=self.config.EMBEDDING_MODEL,
                    input=batch_texts
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Small delay to respect rate limits
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"❌ Error creating embeddings for batch {i//batch_size + 1}: {str(e)}")
                raise
        
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        logger.info(f"✅ Created {embeddings_array.shape[0]} embeddings of dimension {embeddings_array.shape[1]}")
        
        return embeddings_array
    
    def _build_faiss_index(self, embeddings: np.ndarray) -> None:
        """Build FAISS index from embeddings"""
        logger.info("🏗️ Building FAISS index...")
        
        dimension = embeddings.shape[1]
        
        # Use IndexFlatIP for cosine similarity (inner product)
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(dimension)
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        logger.info(f"✅ FAISS index built with {self.index.ntotal} vectors")
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search for similar chunks given a query
        
        Args:
            query: Search query text
            top_k: Number of results to return (default: config.TOP_K)
            
        Returns:
            List of similar chunks with similarity scores
        """
        if not self.index:
            raise ValueError("Index not built. Call build_index() or load_embeddings() first.")
        
        if top_k is None:
            top_k = self.config.TOP_K
        
        # Clean query
        cleaned_query = TextPreprocessor.clean_query(query)
        
        # Create query embedding
        query_embedding = self._create_embeddings([cleaned_query])
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search in index
        similarities, indices = self.index.search(query_embedding, top_k)
        
        # Prepare results
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if similarity >= self.config.SIMILARITY_THRESHOLD:
                chunk = self.chunks_metadata[idx].copy()
                chunk['similarity_score'] = float(similarity)
                chunk['rank'] = i + 1
                results.append(chunk)
        
        logger.info(f"🔍 Found {len(results)} relevant chunks for query: '{query[:50]}...'")
        
        return results
    
    def _save_index(self) -> None:
        """Save FAISS index and metadata to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(self.embeddings_path))
            
            # Save chunks metadata
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.chunks_metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Index saved to {self.embeddings_path}")
            logger.info(f"💾 Metadata saved to {self.metadata_path}")
            
        except Exception as e:
            logger.error(f"❌ Error saving index: {str(e)}")
            raise
    
    def load_embeddings(self) -> bool:
        """
        Load existing FAISS index and metadata from disk
        
        Returns:
            True if loading successful, False otherwise
        """
        try:
            if not self.embeddings_path.exists() or not self.metadata_path.exists():
                logger.warning("📂 No existing embeddings found")
                return False
            
            logger.info("📂 Loading existing embeddings...")
            
            # Load FAISS index
            self.index = faiss.read_index(str(self.embeddings_path))
            
            # Load metadata
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.chunks_metadata = json.load(f)
            
            logger.info(f"✅ Loaded index with {self.index.ntotal} vectors")
            logger.info(f"✅ Loaded metadata for {len(self.chunks_metadata)} chunks")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading embeddings: {str(e)}")
            return False
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get chunk by its ID"""
        for chunk in self.chunks_metadata:
            if chunk.get('chunk_id') == chunk_id:
                return chunk
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        if not self.index:
            return {"status": "not_initialized"}
        
        stats = {
            "status": "ready",
            "total_vectors": self.index.ntotal,
            "vector_dimension": self.index.d,
            "total_chunks": len(self.chunks_metadata),
            "chunk_types": {},
            "sources": set()
        }
        
        # Analyze chunks
        for chunk in self.chunks_metadata:
            chunk_type = chunk.get('type', 'unknown')
            stats["chunk_types"][chunk_type] = stats["chunk_types"].get(chunk_type, 0) + 1
            stats["sources"].add(chunk.get('source', 'unknown'))
        
        stats["sources"] = list(stats["sources"])
        
        return stats
    
    def delete_index(self) -> None:
        """Delete stored index files"""
        try:
            if self.embeddings_path.exists():
                self.embeddings_path.unlink()
                logger.info("🗑️ Deleted embeddings file")
            
            if self.metadata_path.exists():
                self.metadata_path.unlink()
                logger.info("🗑️ Deleted metadata file")
            
            self.index = None
            self.chunks_metadata = []
            
        except Exception as e:
            logger.error(f"❌ Error deleting index: {str(e)}")
    
    def rerank_results(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Simple re-ranking of results based on query relevance
        This is a basic implementation - can be enhanced with more sophisticated methods
        """
        if not results:
            return results
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for result in results:
            content_lower = result['content'].lower()
            content_words = set(content_lower.split())
            
            # Calculate additional relevance factors
            word_overlap = len(query_words.intersection(content_words))
            word_overlap_ratio = word_overlap / len(query_words) if query_words else 0
            
            # Boost score based on exact phrase matches
            exact_matches = sum(1 for word in query_words if word in content_lower)
            
            # Calculate boosted score
            base_score = result['similarity_score']
            boost = (word_overlap_ratio * 0.1) + (exact_matches * 0.05)
            result['boosted_score'] = base_score + boost
        
        # Re-sort by boosted score
        results.sort(key=lambda x: x.get('boosted_score', x['similarity_score']), reverse=True)
        
        return results


class EmbeddingCache:
    """Cache for embeddings to avoid re-computation"""
    
    def __init__(self, cache_dir: str = "data/embedding_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        import hashlib
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding if available"""
        cache_key = self.get_cache_key(text)
        cache_file = self.cache_dir / f"{cache_key}.npy"
        
        if cache_file.exists():
            try:
                return np.load(cache_file)
            except Exception:
                pass
        
        return None
    
    def save_embedding(self, text: str, embedding: np.ndarray) -> None:
        """Save embedding to cache"""
        cache_key = self.get_cache_key(text)
        cache_file = self.cache_dir / f"{cache_key}.npy"
        
        try:
            np.save(cache_file, embedding)
        except Exception as e:
            logger.warning(f"⚠️ Could not cache embedding: {str(e)}")
    
    def clear_cache(self) -> None:
        """Clear all cached embeddings"""
        try:
            for cache_file in self.cache_dir.glob("*.npy"):
                cache_file.unlink()
            logger.info("🗑️ Embedding cache cleared")
        except Exception as e:
            logger.error(f"❌ Error clearing cache: {str(e)}")