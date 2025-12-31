"""
FAISS IndexFlatL2 Vector Database Generator
Generates vector databases for each corpus using VoyageAI embedding models:
- clapnq: voyage-3-large (VoyageAI)
- govt: voyage-3-large (VoyageAI)
- cloud: voyage-3-large (VoyageAI)
- fiqa: voyage-finance-2 (VoyageAI)
"""

import json
import os
import pickle
from pathlib import Path
from typing import List, Dict
import numpy as np
import faiss
from dotenv import load_dotenv
from tqdm import tqdm

# Import embedding client
import voyageai


class FAISSVectorDBGenerator:
    """Generate FAISS vector databases for different corpora with specific embeddings."""
    
    def __init__(self, env_path: str):
        """Initialize with API keys from environment file."""
        load_dotenv(env_path)
        
        self.voyage_client = voyageai.Client(api_key=os.getenv('VOYAGE_API_KEY'))
        
        self.model_configs = {
            'clapnq': {
                'model': 'voyage-3-large',
                'dimension': 1024
            },
            'govt': {
                'model': 'voyage-3-large',
                'dimension': 1024
            },
            'cloud': {
                'model': 'voyage-3-large',
                'dimension': 1024
            },
            'fiqa': {
                'model': 'voyage-finance-2',
                'dimension': 1024
            }
        }
    
    def load_corpus(self, corpus_path: str) -> List[Dict]:
        """Load corpus from JSONL file."""
        documents = []
        print(f"Loading corpus from {corpus_path}...")
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Reading documents"):
                doc = json.loads(line.strip())
                documents.append(doc)
        
        print(f"Loaded {len(documents)} documents")
        return documents
    
    def get_embeddings_voyage(self, texts: List[str], model: str, batch_size: int = 64, 
                             checkpoint_path: str = None) -> np.ndarray:
        """Get embeddings from VoyageAI API with checkpoint support."""
        embeddings = []
        start_idx = 0
        
        # Load checkpoint if exists
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
                embeddings = checkpoint_data['embeddings']
                start_idx = checkpoint_data['last_index']
            print(f"Resuming from batch {start_idx}/{len(texts)}")
        
        # Process remaining batches
        for i in tqdm(range(start_idx, len(texts), batch_size), desc="Getting VoyageAI embeddings"):
            batch = texts[i:i + batch_size]
            response = self.voyage_client.embed(
                texts=batch,
                model=model,
                input_type="document"
            )
            embeddings.extend(response.embeddings)
            
            # Save checkpoint every 50 batches (~3200 docs)
            if checkpoint_path and (i - start_idx) % (50 * batch_size) == 0 and i > start_idx:
                checkpoint_data = {
                    'embeddings': embeddings,
                    'last_index': i + batch_size
                }
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(checkpoint_data, f)
                print(f"Checkpoint saved at {i + batch_size}/{len(texts)}")
        
        # Delete checkpoint after completion
        if checkpoint_path and os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print("Checkpoint deleted (completed)")
        
        return np.array(embeddings, dtype='float32')
    
    def get_embeddings(self, texts: List[str], corpus_name: str, checkpoint_path: str = None) -> np.ndarray:
        """Get embeddings based on corpus configuration."""
        config = self.model_configs[corpus_name]
        model = config['model']
        
        print(f"\nGenerating embeddings for {corpus_name} using VoyageAI/{model}")
        
        return self.get_embeddings_voyage(texts, model, checkpoint_path=checkpoint_path)
    
    def create_faiss_index(self, embeddings: np.ndarray, dimension: int) -> faiss.IndexFlatL2:
        """Create FAISS IndexFlatL2 and add embeddings."""
        print(f"Creating FAISS IndexFlatL2 with dimension {dimension}")
        
        print("Normalizing embeddings for L2 distance...")
        faiss.normalize_L2(embeddings)
        
        # Create L2 index
        index = faiss.IndexFlatL2(dimension)
        
        # Add embeddings to index
        print(f"Adding {len(embeddings)} vectors to index...")
        index.add(embeddings)
        
        print(f"Index created successfully with {index.ntotal} vectors")
        return index
    
    def save_index_and_metadata(self, index: faiss.IndexFlatL2, documents: List[Dict], 
                               output_dir: Path, corpus_name: str):
        """Save FAISS index and document metadata."""
        # Create output directory
        corpus_dir = output_dir / corpus_name / "voyage"
        corpus_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = corpus_dir / "faiss_index.bin"
        faiss.write_index(index, str(index_path))
        print(f"Saved FAISS index to {index_path}")
        
        # Save document metadata
        metadata_path = corpus_dir / "documents.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(documents, f)
        print(f"Saved document metadata to {metadata_path}")
        
        # Save configuration
        config_path = corpus_dir / "config.json"
        config = {
            'corpus_name': corpus_name,
            'model': self.model_configs[corpus_name]['model'],
            'provider': 'voyageai',
            'dimension': self.model_configs[corpus_name]['dimension'],
            'num_documents': len(documents),
            'index_type': 'IndexFlatL2'
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Saved configuration to {config_path}")
    
    def process_corpus(self, corpus_path: Path, corpus_name: str, output_dir: Path):
        """Process a single corpus: load, embed, create index, and save."""
        print(f"\n{'='*80}")
        print(f"Processing corpus: {corpus_name}")
        print(f"{'='*80}")
        
        # Check if already completed
        corpus_dir = output_dir / corpus_name / "voyage"
        corpus_dir.mkdir(parents=True, exist_ok=True)
        
        index_path = corpus_dir / "faiss_index.bin"
        if index_path.exists():
            print(f"FAISS index already exists for {corpus_name}, skipping...")
            return
        
        # Load documents
        documents = self.load_corpus(corpus_path)
        
        # Extract text for embedding
        texts = [doc.get('text', '') for doc in documents]
        
        # Get embeddings with checkpoint support
        checkpoint_path = output_dir / f".{corpus_name}_checkpoint.pkl"
        embeddings = self.get_embeddings(texts, corpus_name, checkpoint_path=str(checkpoint_path))
        
        # Create FAISS index
        dimension = self.model_configs[corpus_name]['dimension']
        index = self.create_faiss_index(embeddings, dimension)
        
        # Save everything
        self.save_index_and_metadata(index, documents, output_dir, corpus_name)
        
        print(f"\nCompleted processing {corpus_name}")
    
    def process_all_corpora(self, corpora_dir: Path, output_dir: Path):
        """Process all corpora in the passage_level directory."""
        corpora = ['clapnq', 'cloud', 'fiqa', 'govt']
        
        print(f"\nStarting FAISS database generation for {len(corpora)} corpora")
        print(f"Input directory: {corpora_dir}")
        print(f"Output directory: {output_dir}")
        
        for corpus_name in corpora:
            corpus_path = corpora_dir / f"{corpus_name}.jsonl"
            
            if not corpus_path.exists():
                print(f"\nWarning: {corpus_path} not found, skipping...")
                continue
            
            try:
                self.process_corpus(corpus_path, corpus_name, output_dir)
            except Exception as e:
                print(f"\nError processing {corpus_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\n{'='*80}")
        print("All corpora processing completed!")
        print(f"{'='*80}")


def main():
    """Main function to generate all FAISS databases."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate FAISS databases for corpora")
    parser.add_argument(
        '--corpus',
        type=str,
        choices=['clapnq', 'cloud', 'fiqa', 'govt', 'all'],
        default='all',
        help='Specific corpus to process, or "all" for all corpora'
    )
    args = parser.parse_args()
    
    # Define paths relative to project root
    # Assuming script is run from project root or src/pipeline/indexing/voyage_gen
    # We'll resolve the project root based on the script location
    script_path = Path(__file__).resolve()
    # Go up 4 levels: src/pipeline/indexing/voyage_gen -> src/pipeline/indexing -> src/pipeline -> src -> root
    project_root = script_path.parents[4]
    
    env_path = project_root / ".env"
    corpora_dir = project_root / "data/passage_level_raw"
    output_dir = project_root / "indices"
    
    # Verify paths exist
    if not env_path.exists():
        # Try looking in current directory
        if Path(".env").exists():
            env_path = Path(".env")
        else:
            print(f"Warning: Environment file not found at {env_path}")
            
    if not corpora_dir.exists():
        raise FileNotFoundError(f"Corpora directory not found: {corpora_dir}")
    
    # Create generator
    generator = FAISSVectorDBGenerator(str(env_path))
    
    # Process specific corpus or all
    if args.corpus == 'all':
        generator.process_all_corpora(corpora_dir, output_dir)
    else:
        corpus_path = corpora_dir / f"{args.corpus}.jsonl"
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
        
        print(f"\n{'='*80}")
        print(f"Processing single corpus: {args.corpus}")
        print(f"{'='*80}\n")
        
        generator.process_corpus(corpus_path, args.corpus, output_dir)
        
        print(f"\n{'='*80}")
        print(f"Completed processing {args.corpus}!")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
