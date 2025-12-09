"""
Production-ready ChromaDB Vector Database Pipeline
Builds and updates vector database from HuggingFace blog articles.

Features:
- Initial database creation
- Incremental updates for periodic runs
- Proper error handling and logging
- Progress tracking and statistics
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional
from datetime import datetime

import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)

# ------------------ Configuration ------------------

class Config:
    """Configuration for vector database pipeline"""
    
    # Paths
    VECTOR_STORE_PATH = "vector_store"
    COLLECTION_NAME = "hf_blogs_vectors"
    
    # Ollama settings
    OLLAMA_URL = "http://localhost:11434"
    OLLAMA_MODEL = "nomic-embed-text:latest"
    
    # Chunking settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100
    
    # Input/Output
    INPUT_JSON = "Dataset/hf_blogs_data.json"
    PROCESSED_LINKS_FILE = "vector_store/processed_links.json"

# ------------------ Logging Setup ------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("vector_db_pipeline.log")
    ]
)
logger = logging.getLogger("vector_db_pipeline")

# ------------------ Processed Links Tracker ------------------

class ProcessedLinksTracker:
    """Track which articles have been processed to enable incremental updates"""
    
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.processed_links: Set[str] = self._load()
    
    def _load(self) -> Set[str]:
        """Load processed links from file"""
        if self.filepath.exists():
            try:
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                    links = set(data.get("processed_links", []))
                    logger.info(f"Loaded {len(links)} processed links from {self.filepath}")
                    return links
            except Exception as e:
                logger.warning(f"Could not load processed links: {e}")
                return set()
        return set()
    
    def _save(self):
        """Save processed links to file"""
        try:
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(self.filepath, 'w') as f:
                json.dump({
                    "processed_links": list(self.processed_links),
                    "last_updated": datetime.now().isoformat(),
                    "total_count": len(self.processed_links)
                }, f, indent=2)
            logger.debug(f"Saved {len(self.processed_links)} processed links")
        except Exception as e:
            logger.error(f"Failed to save processed links: {e}")
    
    def is_processed(self, link: str) -> bool:
        """Check if link has been processed"""
        return link in self.processed_links
    
    def mark_processed(self, link: str):
        """Mark link as processed"""
        self.processed_links.add(link)
        self._save()
    
    def get_unprocessed(self, all_links: List[str]) -> List[str]:
        """Get list of unprocessed links"""
        return [link for link in all_links if link not in self.processed_links]

# ------------------ Vector Database Manager ------------------

class VectorDBManager:
    """Manages ChromaDB vector database operations"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        
        # Initialize embedding function
        logger.info(f"Initializing Ollama embedding function: {self.config.OLLAMA_MODEL}")
        self.embedding_function = OllamaEmbeddingFunction(
            url=self.config.OLLAMA_URL,
            model_name=self.config.OLLAMA_MODEL,
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP
        )
        
        # Initialize ChromaDB client
        logger.info(f"Initializing ChromaDB client at {self.config.VECTOR_STORE_PATH}")
        self.client = chromadb.PersistentClient(path=self.config.VECTOR_STORE_PATH)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.config.COLLECTION_NAME,
            embedding_function=self.embedding_function
        )
        
        # Initialize processed links tracker
        self.tracker = ProcessedLinksTracker(self.config.PROCESSED_LINKS_FILE)
        
        logger.info(f"✓ Collection '{self.config.COLLECTION_NAME}' ready")
        logger.info(f"✓ Current collection size: {self.collection.count()} chunks")
    
    def load_articles(self, filepath: str) -> List[Dict]:
        """Load articles from JSON file"""
        logger.info(f"Loading articles from {filepath}")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            logger.info(f"Loaded {len(articles)} articles")
            return articles
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {filepath}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading articles: {e}")
            raise
    
    def process_article(self, article: Dict, global_chunk_id: int) -> int:
        """
        Process a single article: chunk and add to vector database.
        
        Args:
            article: Article dictionary
            global_chunk_id: Starting chunk ID
            
        Returns: Next available chunk ID
        """
        # Validate article
        if article.get("_error"):
            logger.warning(f"Skipping article with error: {article.get('link', 'unknown')}")
            return global_chunk_id
        
        text = article.get("Text", "")
        if not text:
            logger.warning(f"Skipping article with no text: {article.get('link', 'unknown')}")
            return global_chunk_id
        
        link = article.get("link", "")
        title = article.get("Title", "Untitled")
        publish_date = article.get("Publish Date", "")
        
        # Check if already processed
        if self.tracker.is_processed(link):
            logger.debug(f"Article already processed: {title}")
            return global_chunk_id
        
        # Chunk the text
        try:
            chunks = self.text_splitter.split_text(text)
        except Exception as e:
            logger.error(f"Error chunking article '{title}': {e}")
            return global_chunk_id
        
        if not chunks:
            logger.warning(f"No chunks generated for article: {title}")
            return global_chunk_id
        
        # Add chunks to collection
        chunk_ids = []
        chunk_documents = []
        chunk_metadatas = []
        
        for chunk_text in chunks:
            chunk_ids.append(f"id{global_chunk_id}")
            chunk_documents.append(chunk_text)
            chunk_metadatas.append({
                "link": link,
                "Title": title,
                "Publish Date": publish_date
            })
            global_chunk_id += 1
        
        try:
            self.collection.add(
                ids=chunk_ids,
                documents=chunk_documents,
                metadatas=chunk_metadatas
            )
            logger.info(f"✓ Added {len(chunks)} chunks from: {title}")
        except Exception as e:
            logger.error(f"Error adding chunks to collection for '{title}': {e}")
            return global_chunk_id
        
        # Mark as processed
        self.tracker.mark_processed(link)
        
        return global_chunk_id
    
    def build_database(self, input_file: Optional[str] = None, force_rebuild: bool = False):
        """
        Build or update vector database from JSON file.
        
        Args:
            input_file: Path to input JSON file (uses config default if None)
            force_rebuild: If True, clear existing collection and rebuild from scratch
        """
        input_file = input_file or self.config.INPUT_JSON
        
        # Load articles
        articles = self.load_articles(input_file)
        
        # Check if force rebuild
        if force_rebuild:
            logger.warning("Force rebuild requested - clearing existing collection")
            self.client.delete_collection(self.config.COLLECTION_NAME)
            self.collection = self.client.create_collection(
                name=self.config.COLLECTION_NAME,
                embedding_function=self.embedding_function
            )
            # Reset tracker
            self.tracker.processed_links.clear()
            logger.info("Collection cleared and recreated")
        
        # Filter for unprocessed articles
        all_links = [a.get("link", "") for a in articles if not a.get("_error")]
        unprocessed_links = self.tracker.get_unprocessed(all_links)
        
        if not unprocessed_links:
            logger.info("✓ All articles already processed. Database is up-to-date.")
            self._print_summary(articles)
            return
        
        logger.info(f"Found {len(unprocessed_links)} new articles to process")
        
        # Get starting chunk ID (max existing ID + 1)
        existing_count = self.collection.count()
        global_chunk_id = existing_count + 1
        
        # Process articles
        stats = {
            "total_articles": len(articles),
            "new_articles": len(unprocessed_links),
            "chunks_added": 0,
            "articles_processed": 0,
            "articles_skipped": 0,
            "errors": 0
        }
        
        logger.info("=" * 60)
        logger.info("PROCESSING ARTICLES")
        logger.info("=" * 60)
        
        for i, article in enumerate(articles, 1):
            link = article.get("link", "")
            
            # Skip if already processed
            if link not in unprocessed_links:
                continue
            
            try:
                chunks_before = global_chunk_id
                global_chunk_id = self.process_article(article, global_chunk_id)
                chunks_added = global_chunk_id - chunks_before
                
                if chunks_added > 0:
                    stats["chunks_added"] += chunks_added
                    stats["articles_processed"] += 1
                else:
                    stats["articles_skipped"] += 1
                
                # Progress update
                if i % 10 == 0:
                    logger.info(f"Progress: {i}/{len(articles)} articles checked")
                
            except Exception as e:
                logger.error(f"Error processing article {i}: {e}")
                stats["errors"] += 1
        
        # Final summary
        self._print_summary(articles, stats)
    
    def _print_summary(self, articles: List[Dict], stats: Optional[Dict] = None):
        """Print processing summary"""
        logger.info("=" * 60)
        logger.info("DATABASE STATUS")
        logger.info("=" * 60)
        
        final_count = self.collection.count()
        processed_count = len(self.tracker.processed_links)
        
        logger.info(f"Total articles in dataset: {len(articles)}")
        logger.info(f"Total articles processed: {processed_count}")
        logger.info(f"Total chunks in database: {final_count}")
        
        if stats:
            logger.info(f"\nThis run:")
            logger.info(f"  New articles processed: {stats['articles_processed']}")
            logger.info(f"  New chunks added: {stats['chunks_added']}")
            logger.info(f"  Articles skipped: {stats['articles_skipped']}")
            logger.info(f"  Errors encountered: {stats['errors']}")
        
        logger.info("=" * 60)
        
        if processed_count == len(articles):
            logger.info("✓ Database is up-to-date. All articles processed.")
        else:
            remaining = len(articles) - processed_count
            logger.info(f"⚠ {remaining} articles still need processing")
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        return {
            "collection_name": self.config.COLLECTION_NAME,
            "total_chunks": self.collection.count(),
            "processed_articles": len(self.tracker.processed_links),
            "vector_store_path": self.config.VECTOR_STORE_PATH
        }

# ------------------ Main Execution ------------------

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build and update ChromaDB vector database from HuggingFace blog articles"
    )
    parser.add_argument(
        "--input",
        default=Config.INPUT_JSON,
        help=f"Input JSON file (default: {Config.INPUT_JSON})"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild: clear existing database and rebuild from scratch"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show database statistics only (don't process)"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize manager
        logger.info("Initializing Vector Database Manager...")
        manager = VectorDBManager()
        
        if args.stats:
            # Just show stats
            stats = manager.get_stats()
            print("\n" + "=" * 60)
            print("DATABASE STATISTICS")
            print("=" * 60)
            for key, value in stats.items():
                print(f"{key}: {value}")
            print("=" * 60)
        else:
            # Build/update database
            manager.build_database(
                input_file=args.input,
                force_rebuild=args.rebuild
            )
            logger.info("✓ Pipeline completed successfully")
        
    except KeyboardInterrupt:
        logger.warning("\n⚠ Process interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()