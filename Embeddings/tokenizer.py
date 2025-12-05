"""
RAG System: Chunking, Embedding, and Vector Storage for HuggingFace Articles

Architecture:
- Chunk articles with overlap WITHIN each article
- Store chunks in vector DB with metadata
- Each chunk links back to parent article
- Efficient retrieval with proper metadata structure
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib

# For embeddings - you'll need to install: pip install sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("⚠️  Please install: pip install sentence-transformers")
    SentenceTransformer = None


# ------------------ DATA STRUCTURES ------------------

@dataclass
class ChunkMetadata:
    """Metadata for a single chunk"""
    # Chunk identification
    chunk_id: str           # Unique ID for this chunk
    chunk_index: int        # Position in article (0, 1, 2, ...)
    
    # Parent article reference
    article_id: str         # Link or unique ID of parent article
    article_title: str      # Title of parent article
    article_link: str       # URL to original article
    
    # Chunk positioning
    chunk_start: int        # Character position in original article
    chunk_end: int          # Character position in original article
    total_chunks: int       # Total chunks in this article
    
    # Article metadata (copied for quick access)
    authors: List[str]      # Article authors
    publish_date: str       # Publication date
    date_extracted: str     # Extracted date from text
    
    # Optional fields
    keywords: List[str]     # Article keywords
    summary: str            # Article summary
    
    # Chunk metadata
    chunk_text: str         # The actual chunk text
    word_count: int         # Words in this chunk
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


# ------------------ CHUNKING STRATEGY ------------------

class ArticleChunker:
    """Smart chunking with overlap within articles"""
    
    def __init__(
        self,
        chunk_size: int = 512,      # Characters per chunk
        overlap_size: int = 50,      # Character overlap between chunks
        min_chunk_size: int = 100    # Minimum chunk size
    ):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.min_chunk_size = min_chunk_size
    
    def chunk_article(self, article: Dict) -> List[Tuple[str, int, int]]:
        """
        Chunk a single article into overlapping segments.
        
        Returns: List of (chunk_text, start_pos, end_pos)
        """
        text = article.get("Text", "")
        if not text or len(text) < self.min_chunk_size:
            return [(text, 0, len(text))] if text else []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # Define chunk end
            end = min(start + self.chunk_size, text_length)
            
            # Try to break at sentence boundary if not at end
            if end < text_length:
                # Look for sentence endings: . ! ? followed by space or newline
                last_sentence = max(
                    text.rfind('. ', start, end),
                    text.rfind('! ', start, end),
                    text.rfind('? ', start, end),
                    text.rfind('.\n', start, end),
                )
                
                # If found a good break point, use it
                if last_sentence > start + self.min_chunk_size:
                    end = last_sentence + 1
            
            # Extract chunk
            chunk_text = text[start:end].strip()
            
            # Only add non-empty chunks
            if chunk_text and len(chunk_text) >= self.min_chunk_size:
                chunks.append((chunk_text, start, end))
            
            # Move to next chunk with overlap
            start = end - self.overlap_size
            
            # Prevent infinite loop
            if start >= text_length:
                break
        
        return chunks
    
    def process_article_to_chunks(
        self,
        article: Dict,
        article_index: int = 0
    ) -> List[ChunkMetadata]:
        """
        Process article into chunks with full metadata.
        
        Args:
            article: Article dictionary
            article_index: Index in dataset (for generating IDs)
            
        Returns: List of ChunkMetadata objects
        """
        # Skip articles with errors or no text
        if article.get("_error") or not article.get("Text"):
            return []
        
        # Get chunks
        chunks = self.chunk_article(article)
        if not chunks:
            return []
        
        # Generate article ID (use link as primary identifier)
        article_link = article.get("link", f"article_{article_index}")
        article_id = hashlib.md5(article_link.encode()).hexdigest()[:12]
        
        # Build chunk metadata
        chunk_metadata_list = []
        
        for chunk_idx, (chunk_text, start_pos, end_pos) in enumerate(chunks):
            # Generate unique chunk ID
            chunk_id = f"{article_id}_chunk_{chunk_idx}"
            
            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                chunk_index=chunk_idx,
                article_id=article_id,
                article_title=article.get("Title", "Untitled"),
                article_link=article_link,
                chunk_start=start_pos,
                chunk_end=end_pos,
                total_chunks=len(chunks),
                authors=article.get("Authors", []),
                publish_date=article.get("Publish Date", ""),
                date_extracted=article.get("Date_Extracted", ""),
                keywords=article.get("Keywords", []),
                summary=article.get("Summary", ""),
                chunk_text=chunk_text,
                word_count=len(chunk_text.split())
            )
            
            chunk_metadata_list.append(metadata)
        
        return chunk_metadata_list


# ------------------ EMBEDDING GENERATOR ------------------

class EmbeddingGenerator:
    """Generate embeddings for text chunks"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embedding model.
        
        Popular models:
        - all-MiniLM-L6-v2: Fast, 384 dim, good for general use
        - all-mpnet-base-v2: Better quality, 768 dim, slower
        - multi-qa-mpnet-base-dot-v1: Optimized for Q&A
        """
        if SentenceTransformer is None:
            raise ImportError("Please install: pip install sentence-transformers")
        
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"✓ Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def embed_chunks(
        self,
        chunks: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for list of text chunks.
        
        Args:
            chunks: List of text strings
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns: numpy array of shape (num_chunks, embedding_dim)
        """
        embeddings = self.model.encode(
            chunks,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        return embeddings


# ------------------ COMPLETE PIPELINE ------------------

class RAGPipeline:
    """Complete pipeline: chunk articles → generate embeddings → prepare for storage"""
    
    def __init__(
        self,
        chunk_size: int = 512,
        overlap_size: int = 50,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.chunker = ArticleChunker(
            chunk_size=chunk_size,
            overlap_size=overlap_size
        )
        self.embedding_generator = EmbeddingGenerator(embedding_model)
    
    def process_dataset(
        self,
        input_file: str,
        output_embeddings_file: str = "embeddings.npy",
        output_metadata_file: str = "metadata.json"
    ) -> Dict:
        """
        Complete pipeline: Load → Chunk → Embed → Save
        
        Saves:
        1. embeddings.npy: numpy array of embeddings
        2. metadata.json: list of metadata dicts (same order as embeddings)
        
        Returns: Statistics dict
        """
        # Load dataset
        print(f"Loading dataset from {input_file}...")
        with open(input_file, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        
        print(f"Loaded {len(articles)} articles")
        
        # Process all articles into chunks
        print("\nChunking articles...")
        all_chunks_metadata = []
        
        for idx, article in enumerate(articles):
            chunk_metadata_list = self.chunker.process_article_to_chunks(article, idx)
            all_chunks_metadata.extend(chunk_metadata_list)
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(articles)} articles...")
        
        print(f"✓ Generated {len(all_chunks_metadata)} chunks from {len(articles)} articles")
        
        if not all_chunks_metadata:
            print("⚠️  No chunks generated. Check your data.")
            return {}
        
        # Extract chunk texts for embedding
        chunk_texts = [meta.chunk_text for meta in all_chunks_metadata]
        
        # Generate embeddings
        print(f"\nGenerating embeddings using {self.embedding_generator.model_name}...")
        embeddings = self.embedding_generator.embed_chunks(chunk_texts)
        
        print(f"✓ Generated embeddings: shape {embeddings.shape}")
        
        # Save embeddings
        print(f"\nSaving embeddings to {output_embeddings_file}...")
        np.save(output_embeddings_file, embeddings)
        
        # Save metadata
        print(f"Saving metadata to {output_metadata_file}...")
        metadata_dicts = [meta.to_dict() for meta in all_chunks_metadata]
        with open(output_metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata_dicts, f, indent=2, ensure_ascii=False)
        
        # Calculate statistics
        stats = {
            "total_articles": len(articles),
            "total_chunks": len(all_chunks_metadata),
            "avg_chunks_per_article": len(all_chunks_metadata) / len(articles),
            "embedding_dimension": embeddings.shape[1],
            "model_name": self.embedding_generator.model_name,
            "chunk_size": self.chunker.chunk_size,
            "overlap_size": self.chunker.overlap_size,
            "processed_at": datetime.now().isoformat()
        }
        
        # Print summary
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"Articles processed: {stats['total_articles']}")
        print(f"Chunks generated: {stats['total_chunks']}")
        print(f"Avg chunks/article: {stats['avg_chunks_per_article']:.1f}")
        print(f"Embedding dimension: {stats['embedding_dimension']}")
        print(f"\nFiles saved:")
        print(f"  - {output_embeddings_file} ({embeddings.nbytes / 1024 / 1024:.1f} MB)")
        print(f"  - {output_metadata_file}")
        print("=" * 60)
        
        return stats


# ------------------ USAGE EXAMPLE ------------------

if __name__ == "__main__":
    """
    IMPORTANT CONCEPT:
    
    Your original thinking was mostly correct! Here's the clarification:
    
    ✓ YES: Chunk each article with overlap WITHIN that article
    ✓ YES: Store metadata at same index as embeddings
    ✓ NO: Don't duplicate ALL metadata 5x - store chunk-specific metadata
    
    BETTER APPROACH:
    Instead of storing full article metadata 5 times, we store:
    1. Chunk-specific info (chunk_id, chunk_index, chunk_text)
    2. Parent article reference (article_id, article_link, article_title)
    3. Lightweight article metadata (authors, date, etc.)
    
    This way:
    - Each chunk has unique identifier
    - Each chunk links back to parent article
    - Metadata is synchronized with embeddings by index
    - You can reconstruct full article or show just the relevant chunk
    """
    
    # Initialize pipeline
    pipeline = RAGPipeline(
        chunk_size=512,      # ~100-150 words per chunk
        overlap_size=50,     # ~10% overlap
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"  # Fast & good
    )
    
    # Process dataset
    stats = pipeline.process_dataset(
        input_file="Dataset/hf_blogs_data.json",
        output_embeddings_file="embeddings.npy",
        output_metadata_file="metadata.json"
    )
    
    print("\n✓ Ready for vector database!")
    print("\nNext steps:")
    print("1. Load embeddings.npy and metadata.json")
    print("2. Insert into vector DB (FAISS, Pinecone, Weaviate, etc.)")
    print("3. During retrieval: query → get top-k chunks → use metadata to show source")