"""
FAISS Index Generator for Cohere Embeddings v3
Generates vector databases for each corpus using Cohere embed-english-v3.0:
- Uses Cohere's embed-english-v3.0 model (1024 dimensions)
- Fast API-based embeddings
- Optimized for search with input_type="search_document"
"""

import json
import os
import pickle
import logging
from pathlib import Path
from typing import List, Dict
import numpy as np
import faiss
from dotenv import load_dotenv
from tqdm import tqdm
import cohere

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class CohereFAISSGenerator:
    """Generate FAISS vector databases using Cohere embeddings."""
    
    def __init__(self, env_path: str = ".env"):
        """Initialize with API keys from environment file."""
        if os.path.exists(env_path):
            load_dotenv(env_path)
        else:
            logger.warning(f"Environment file not found at {env_path}, assuming variables are set.")
        
        api_key = os.getenv('COHERE_API_KEY')
        if not api_key:
            raise ValueError("COHERE_API_KEY not found in environment variables")
            
        self.cohere_client = cohere.Client(api_key=api_key)
        
        # Cohere embed-english-v3.0 is 1024 dimensions
        self.model_name = "embed-english-v3.0"
        self.dimension = 1024
    
    def load_corpus(self, corpus_path: str) -> List[Dict]:
        """Load corpus from JSONL file."""
        documents = []
        logger.info(f"Loading corpus from {corpus_path}...")
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Reading documents"):
                doc = json.loads(line.strip())
                documents.append(doc)
        
        logger.info(f"Loaded {len(documents)} documents")
        return documents
    
    def get_embeddings_cohere(self, texts: List[str], batch_size: int = 96, 
                             checkpoint_path: str = None) -> np.ndarray:
        """
        Get embeddings from Cohere API with checkpoint support.
        
        Cohere v3 supports up to 96 documents per batch.
        """
        embeddings = []
        start_idx = 0
        
        # Load checkpoint if exists
        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
                embeddings = checkpoint_data['embeddings']
                start_idx = checkpoint_data['last_index']
            logger.info(f"Resuming from index {start_idx}/{len(texts)}")
        
        # Process remaining batches
        for i in tqdm(range(start_idx, len(texts), batch_size), desc="Getting Cohere embeddings"):
            batch = texts[i:i + batch_size]
            try:
                response = self.cohere_client.embed(
                    texts=batch,
                    model=self.model_name,
                    input_type="search_document",  # For indexing documents
                    embedding_types=["float"]
                )
                embeddings.extend(response.embeddings.float)
            except Exception as e:
                logger.error(f"Error fetching embeddings at batch {i}: {e}")
                # If rate limited, wait and retry
                if "rate" in str(e).lower():
                    logger.warning("Rate limited, waiting 60 seconds...")
                    import time
                    time.sleep(60)
                    try:
                        response = self.cohere_client.embed(
                            texts=batch,
                            model=self.model_name,
                            input_type="search_document",
                            embedding_types=["float"]
                        )
                        embeddings.extend(response.embeddings.float)
                    except Exception as e2:
                        logger.error(f"Retry failed: {e2}")
                        raise e2
                else:
                    raise e
            
            # Save checkpoint every 50 batches (~4800 docs)
            if checkpoint_path and (i - start_idx) % (50 * batch_size) == 0 and i > start_idx:
                checkpoint_data = {
                    'embeddings': embeddings,
                    'last_index': i + batch_size
                }
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(checkpoint_data, f)
                logger.info(f"Checkpoint saved at {i + batch_size}/{len(texts)}")
        
        # Delete checkpoint after completion
        if checkpoint_path and os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            logger.info("Checkpoint deleted (completed)")
        
        return np.array(embeddings, dtype='float32')
    
    def create_faiss_index(self, embeddings: np.ndarray) -> faiss.IndexFlatIP:
        """Create FAISS IndexFlatIP (Inner Product) and add embeddings."""
        logger.info(f"Creating FAISS IndexFlatIP with dimension {self.dimension}")
        
        logger.info("Normalizing embeddings for Inner Product (Cosine Similarity)...")
        faiss.normalize_L2(embeddings)
        
        # Create Inner Product index (for normalized vectors, IP = Cosine Similarity)
        index = faiss.IndexFlatIP(self.dimension)
        
        # Add embeddings to index
        logger.info(f"Adding {len(embeddings)} vectors to index...")
        index.add(embeddings)
        
        logger.info(f"Index created with {index.ntotal} vectors")
        return index
    
    def save_index(self, index: faiss.Index, doc_ids: List[str], output_dir: str):
        """Save FAISS index and document IDs."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(output_dir, 'index.faiss')
        logger.info(f"Saving FAISS index to {index_path}")
        faiss.write_index(index, index_path)
        
        # Save doc IDs
        ids_path = os.path.join(output_dir, 'doc_ids.json')
        logger.info(f"Saving document IDs to {ids_path}")
        with open(ids_path, 'w', encoding='utf-8') as f:
            json.dump(doc_ids, f)
        
        logger.info("Index and IDs saved successfully")
    
    def process_corpus(self, corpus_name: str, corpus_path: str, output_dir: str):
        """Process entire corpus: load, embed, index, and save."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing corpus: {corpus_name}")
        logger.info(f"{'='*60}")
        
        # Load corpus
        documents = self.load_corpus(corpus_path)
        
        # Extract text and IDs
        texts = []
        doc_ids = []
        for doc in documents:
            # Concatenate title and text
            title = doc.get('title', '')
            text = doc.get('text', '')
            full_text = f"{title}\n{text}".strip() if title else text
            texts.append(full_text)
            doc_ids.append(doc['_id'])
        
        # Generate embeddings with checkpoint
        checkpoint_path = f".cache/cohere_checkpoint_{corpus_name}.pkl"
        embeddings = self.get_embeddings_cohere(texts, checkpoint_path=checkpoint_path)
        
        # Create FAISS index
        index = self.create_faiss_index(embeddings)
        
        # Save index and IDs
        self.save_index(index, doc_ids, output_dir)
        
        logger.info(f"✓ Completed processing {corpus_name}")


def main():
    """Main function to generate all indices."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate FAISS indices with Cohere embeddings')
    parser.add_argument('--domain', type=str, required=True, 
                       choices=['clapnq', 'cloud', 'fiqa', 'govt', 'all'],
                       help='Domain to process')
    parser.add_argument('--data-dir', type=str, default='data/passage_level_processed',
                       help='Directory containing corpus files')
    parser.add_argument('--output-dir', type=str, default='indices',
                       help='Base directory for output indices')
    parser.add_argument('--env-path', type=str, default='.env',
                       help='Path to .env file')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = CohereFAISSGenerator(env_path=args.env_path)
    
    # Define corpus paths
    domains = ['clapnq', 'cloud', 'fiqa', 'govt'] if args.domain == 'all' else [args.domain]
    
    for domain in domains:
        corpus_path = f"{args.data_dir}/{domain}/corpus.jsonl"
        output_dir = f"{args.output_dir}/{domain}/cohere"
        
        if not os.path.exists(corpus_path):
            logger.warning(f"Corpus not found: {corpus_path}, skipping...")
            continue
        
        generator.process_corpus(domain, corpus_path, output_dir)
    
    logger.info("\n" + "="*60)
    logger.info("All indices generated successfully!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
