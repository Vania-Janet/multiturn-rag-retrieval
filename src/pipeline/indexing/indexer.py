import os
import logging
import argparse
import json
import pickle
import numpy as np
import torch
import random
from tqdm import tqdm
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set seeds for reproducibility
RANDOM_SEED = 42

def set_seed(seed: int = RANDOM_SEED):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Set seeds at import time
set_seed()

# Try imports
try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError:
    pass # Handled in specific indexers

try:
    from rank_bm25 import BM25Okapi
    from nltk.tokenize import word_tokenize
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')
except ImportError:
    BM25Okapi = None
    word_tokenize = None

try:
    from elasticsearch import Elasticsearch, helpers
except ImportError:
    pass

try:
    import scipy.sparse
except ImportError:
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("indexing.log")
    ]
)
logger = logging.getLogger(__name__)

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1024  # Optimized for A100
NUM_WORKERS = 16

def load_corpus(domain: str, processed_dir: str) -> List[Dict[str, Any]]:
    """Loads the processed corpus for a domain."""
    corpus_path = os.path.join(processed_dir, domain, "corpus.jsonl")
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus not found at {corpus_path}")
    
    documents = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Loading {domain} corpus"):
            documents.append(json.loads(line))
    return documents

class BGEIndexer:
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", output_dir: str = "indices", index_subdir: str = "bge", batch_size: int = None):
        self.model_name = model_name
        self.output_dir = output_dir
        self.index_subdir = index_subdir
        self.device = DEVICE
        self.batch_size = batch_size if batch_size is not None else BATCH_SIZE
        
    def build(self, documents: List[Dict[str, Any]], domain: str):
        logger.info(f"Initializing BGE Indexer with {self.model_name} on {self.device}")
        
        # Create output directory
        index_dir = os.path.join(self.output_dir, domain, self.index_subdir)
        os.makedirs(index_dir, exist_ok=True)
        
        # Checkpoint check
        embeddings_path = os.path.join(index_dir, "embeddings.npy")
        ids_path = os.path.join(index_dir, "doc_ids.json")
        
        if os.path.exists(embeddings_path) and os.path.exists(ids_path):
            logger.info(f"Found existing embeddings for {domain} in {index_dir}. Skipping encoding.")
            return

        # Load Model
        model = SentenceTransformer(self.model_name, device=self.device)
        if self.device == "cuda":
            model.half() # FP16
            
        texts = [doc['text'] for doc in documents]
        doc_ids = [doc['id'] for doc in documents]
        
        logger.info(f"Encoding {len(texts)} documents...")
        
        # Encode
        if self.device == "cuda" and torch.cuda.device_count() > 1:
            logger.info(f"🚀 Using {torch.cuda.device_count()} GPUs for parallel encoding!")
            
            # Start the multi-process pool on all available CUDA devices
            pool = model.start_multi_process_pool()
            
            # Compute the embeddings using the multi-process pool
            embeddings = model.encode_multi_process(texts, pool, batch_size=self.batch_size)
            
            # Stop the multi-process pool
            model.stop_multi_process_pool(pool)
            
            # Ensure embeddings are numpy array and normalized
            # encode_multi_process returns numpy array, but normalization might need check
            # SentenceTransformer.encode_multi_process usually normalizes if normalize_embeddings=True is passed to encode?
            # Actually encode_multi_process doesn't take normalize_embeddings directly in older versions, 
            # but let's check if we can normalize manually if needed.
            # However, BGE models usually need normalization.
            
            # Manual normalization if needed (L2 norm)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            # Evitar división por cero
            embeddings = embeddings / np.maximum(norms, 1e-12)
            
        else:
            embeddings = model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
                device=self.device
            )
        
        # Save Embeddings and IDs (Checkpoint)
        logger.info("Saving embeddings and IDs...")
        np.save(embeddings_path, embeddings)
        with open(ids_path, 'w') as f:
            json.dump(doc_ids, f)
            
        # Build FAISS Index
        logger.info("Building FAISS index...")
        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d) # Inner Product for normalized embeddings = Cosine Similarity
        index.add(embeddings)
        
        faiss_path = os.path.join(index_dir, "index.faiss")
        faiss.write_index(index, faiss_path)
        logger.info(f"BGE Index saved to {faiss_path}")

class BM25Indexer:
    def __init__(self, output_dir: str = "indices"):
        self.output_dir = output_dir
        
    def build(self, documents: List[Dict[str, Any]], domain: str):
        if BM25Okapi is None or word_tokenize is None:
            raise ImportError("rank_bm25 and nltk are required for BM25Indexer. Please install them.")

        logger.info(f"Initializing BM25 Indexer for {domain}")
        
        index_dir = os.path.join(self.output_dir, domain, "bm25")
        os.makedirs(index_dir, exist_ok=True)
        
        index_path = os.path.join(index_dir, "index.pkl")
        if os.path.exists(index_path):
            logger.info(f"BM25 index already exists for {domain}. Skipping.")
            return

        logger.info("Tokenizing corpus...")
        tokenized_corpus = []
        for doc in tqdm(documents, desc="Tokenizing"):
            tokenized_corpus.append(word_tokenize(doc['text'].lower()))
            
        logger.info("Building BM25 index...")
        bm25 = BM25Okapi(tokenized_corpus)
        
        # Save
        logger.info(f"Saving BM25 index to {index_path}...")
        with open(index_path, 'wb') as f:
            pickle.dump(bm25, f)
            
        # Save IDs mapping
        ids_path = os.path.join(index_dir, "doc_ids.json")
        doc_ids = [doc['id'] for doc in documents]
        with open(ids_path, 'w') as f:
            json.dump(doc_ids, f)
            
        logger.info("BM25 Indexing complete.")

class ELSERIndexer:
    def __init__(self, output_dir: str = "indices"):
        self.output_dir = output_dir
        self.es_url = os.getenv("ELASTICSEARCH_URL")
        self.es_user = os.getenv("ELASTICSEARCH_USER")
        self.es_password = os.getenv("ELASTICSEARCH_PASSWORD")
        self.model_id = ".elser_model_2" # Default ELSER model

    def build(self, documents: List[Dict[str, Any]], domain: str):
        if not self.es_url:
            logger.warning("ELASTICSEARCH_URL not set. Skipping ELSER indexing.")
            return

        logger.info(f"Initializing ELSER Indexer for {domain} at {self.es_url}")
        
        try:
            es = Elasticsearch(
                self.es_url,
                basic_auth=(self.es_user, self.es_password) if self.es_user else None,
                verify_certs=False,
                request_timeout=300,  # 5 minutes timeout
                max_retries=3,
                retry_on_timeout=True
            )
            
            if not es.ping():
                logger.error("Could not connect to Elasticsearch.")
                return

            index_name = f"mt-rag-{domain}-elser"
            pipeline_name = "elser-ingest-pipeline"

            # 1. Create Ingest Pipeline if not exists
            try:
                es.ingest.put_pipeline(
                    id=pipeline_name,
                    processors=[
                        {
                            "inference": {
                                "model_id": self.model_id,
                                "target_field": "ml.tokens",
                                "field_map": {
                                    "text": "text_field"
                                },
                                "inference_config": {
                                    "text_expansion": {
                                        "results_field": "tokens"
                                    }
                                }
                            }
                        }
                    ]
                )
                logger.info(f"Ingest pipeline {pipeline_name} created/updated.")
            except Exception as e:
                logger.error(f"Failed to create ingest pipeline: {e}")
                return

            # 2. Create Index with Mapping
            if not es.indices.exists(index=index_name):
                es.indices.create(
                    index=index_name,
                    mappings={
                        "properties": {
                            "ml.tokens": {
                                "type": "sparse_vector" # Or rank_features depending on version
                            },
                            "text": {
                                "type": "text"
                            },
                            "id": {
                                "type": "keyword"
                            }
                        }
                    }
                )
                logger.info(f"Index {index_name} created.")
            else:
                logger.info(f"Index {index_name} already exists. Appending/Overwriting.")

            # 3. Bulk Index
            def generate_actions():
                for doc in documents:
                    yield {
                        "_index": index_name,
                        "_id": doc['id'],
                        "pipeline": pipeline_name,
                        "_source": {
                            "text": doc['text'],
                            "id": doc['id'],
                            "title": doc.get('title', '')
                        }
                    }

            logger.info(f"Bulk indexing {len(documents)} documents to {index_name}...")
            
            # Try parallel bulk for better performance
            from elasticsearch.helpers import parallel_bulk
            successes = 0
            for success, info in parallel_bulk(
                es,
                generate_actions(),
                chunk_size=50,  # Increased from 10
                thread_count=4,  # Parallel threads
                request_timeout=300,
                max_chunk_bytes=10485760  # 10MB
            ):
                if success:
                    successes += 1
                else:
                    logger.warning(f"Failed to index document: {info}")
                    
            failed = len(documents) - successes
            logger.info(f"ELSER Indexing complete. Success: {successes}, Failed: {failed}")

        except Exception as e:
            logger.error(f"ELSER Indexing failed: {e}")

class BGEM3Indexer:
    """BGE-M3 Multi-representation Indexer supporting Dense, Sparse, and ColBERT modes."""
    
    def __init__(self, model_name: str = "BAAI/bge-m3", output_dir: str = "indices", 
                 index_subdir: str = "bgem3", mode: str = "dense", batch_size: int = None):
        """
        Args:
            model_name: HuggingFace model name
            output_dir: Base output directory
            index_subdir: Subdirectory for this index (e.g., "bgem3_dense", "bgem3_sparse", "bgem3_colbert", "bgem3")
            mode: One of "dense", "sparse", "colbert", or "all" (generates all three)
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.index_subdir = index_subdir
        self.mode = mode.lower()
        self.device = DEVICE
        self.batch_size = batch_size if batch_size is not None else (128 if self.mode == "colbert" else 256)
        
        if self.mode not in ["dense", "sparse", "colbert", "all"]:
            raise ValueError(f"Invalid mode: {self.mode}. Must be one of: dense, sparse, colbert, all")
    
    def build(self, documents: List[Dict[str, Any]], domain: str):
        """Build BGE-M3 index for the specified domain."""
        from FlagEmbedding import BGEM3FlagModel
        
        logger.info(f"Initializing BGE-M3 Indexer (mode={self.mode}) on {self.device}")
        
        # Create output directory
        index_dir = os.path.join(self.output_dir, domain, self.index_subdir)
        os.makedirs(index_dir, exist_ok=True)
        
        texts = [doc['text'] for doc in documents]
        doc_ids = [doc['id'] for doc in documents]
        
        # Load model once
        logger.info(f"Loading BGE-M3 model: {self.model_name}")
        model = BGEM3FlagModel(self.model_name, use_fp16=True if self.device == "cuda" else False)
        
        if self.mode == "all":
            # Generate all three representations in one pass
            self._build_all(model, texts, doc_ids, index_dir, domain)
        elif self.mode == "dense":
            self._build_dense(model, texts, doc_ids, index_dir, domain)
        elif self.mode == "sparse":
            self._build_sparse(model, texts, doc_ids, index_dir, domain)
        elif self.mode == "colbert":
            self._build_colbert(model, texts, doc_ids, index_dir, domain)
    
    def _build_dense(self, model, texts, doc_ids, index_dir, domain):
        """Build dense embeddings index."""
        embeddings_path = os.path.join(index_dir, "embeddings.npy")
        ids_path = os.path.join(index_dir, "doc_ids.json")
        
        if os.path.exists(embeddings_path) and os.path.exists(ids_path):
            logger.info(f"Dense embeddings already exist for {domain}. Skipping.")
            return
        
        logger.info(f"Encoding {len(texts)} documents (dense)...")
        embeddings = model.encode(
            texts, 
            batch_size=self.batch_size,
            max_length=8192,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )['dense_vecs']
        
        # Normalize embeddings
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        logger.info("Saving dense embeddings...")
        np.save(embeddings_path, embeddings)
        with open(ids_path, 'w') as f:
            json.dump(doc_ids, f)
        
        # Build FAISS index
        logger.info("Building FAISS index...")
        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)
        
        faiss_path = os.path.join(index_dir, "index.faiss")
        faiss.write_index(index, faiss_path)
        logger.info(f"Dense index saved to {faiss_path}")
    
    def _build_sparse(self, model, texts, doc_ids, index_dir, domain):
        """Build sparse embeddings index."""
        index_path = os.path.join(index_dir, "index.npz")
        ids_path = os.path.join(index_dir, "doc_ids.json")
        
        if os.path.exists(index_path) and os.path.exists(ids_path):
            logger.info(f"Sparse embeddings already exist for {domain}. Skipping.")
            return
        
        logger.info(f"Encoding {len(texts)} documents (sparse)...")
        
        # Encode in batches to manage memory
        all_sparse_dicts = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Sparse encoding"):
            batch_texts = texts[i:i + self.batch_size]
            result = model.encode(
                batch_texts,
                batch_size=self.batch_size,
                max_length=8192,
                return_dense=False,
                return_sparse=True,
                return_colbert_vecs=False
            )
            all_sparse_dicts.extend(result['lexical_weights'])
        
        logger.info(f"Saving sparse index to {index_path}...")
        
        # Save as pickle (dict format is easier for sparse retrieval)
        with open(index_path, 'wb') as f:
            pickle.dump(all_sparse_dicts, f)
        
        with open(ids_path, 'w') as f:
            json.dump(doc_ids, f)
        
        logger.info("Sparse index saved.")
    
    def _build_colbert(self, model, texts, doc_ids, index_dir, domain):
        """Build ColBERT multi-vector index."""
        embeddings_path = os.path.join(index_dir, "colbert_vecs.npy")
        ids_path = os.path.join(index_dir, "doc_ids.json")
        
        if os.path.exists(embeddings_path) and os.path.exists(ids_path):
            logger.info(f"ColBERT embeddings already exist for {domain}. Skipping.")
            return
        
        logger.info(f"Encoding {len(texts)} documents (ColBERT)...")
        
        # Encode in batches
        all_colbert_vecs = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="ColBERT encoding"):
            batch_texts = texts[i:i + self.batch_size]
            result = model.encode(
                batch_texts,
                batch_size=self.batch_size,
                max_length=8192,
                return_dense=False,
                return_sparse=False,
                return_colbert_vecs=True
            )
            all_colbert_vecs.extend(result['colbert_vecs'])
        
        logger.info(f"Saving ColBERT vectors to {embeddings_path}...")
        # Save as numpy array of objects (each object is a 2D array)
        np.save(embeddings_path, np.array(all_colbert_vecs, dtype=object), allow_pickle=True)
        
        with open(ids_path, 'w') as f:
            json.dump(doc_ids, f)
        
        logger.info("ColBERT index saved.")
    
    def _build_all(self, model, texts, doc_ids, base_index_dir, domain):
        """Build all three representations (dense, sparse, colbert) in a single encoding pass."""
        logger.info(f"Encoding {len(texts)} documents (ALL: dense + sparse + colbert)...")
        
        # Create subdirectories for each representation type
        dense_dir = os.path.join(base_index_dir, "dense")
        sparse_dir = os.path.join(base_index_dir, "sparse")
        colbert_dir = os.path.join(base_index_dir, "colbert")
        
        for d in [dense_dir, sparse_dir, colbert_dir]:
            os.makedirs(d, exist_ok=True)
        
        # Check if all already exist
        dense_exists = os.path.exists(os.path.join(dense_dir, "embeddings.npy"))
        sparse_exists = os.path.exists(os.path.join(sparse_dir, "index.npz"))
        colbert_exists = os.path.exists(os.path.join(colbert_dir, "colbert_vecs.npy"))
        
        if dense_exists and sparse_exists and colbert_exists:
            logger.info(f"All representations already exist for {domain}. Skipping.")
            return
        
        # Encode all in batches
        all_dense = []
        all_sparse = []
        all_colbert = []
        
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Encoding all representations"):
            batch_texts = texts[i:i + self.batch_size]
            result = model.encode(
                batch_texts,
                batch_size=self.batch_size,
                max_length=8192,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=True
            )
            
            all_dense.append(result['dense_vecs'])
            all_sparse.extend(result['lexical_weights'])
            all_colbert.extend(result['colbert_vecs'])
        
        # Save dense
        logger.info("Saving dense embeddings...")
        dense_embeddings = np.vstack(all_dense)
        dense_embeddings = dense_embeddings / np.linalg.norm(dense_embeddings, axis=1, keepdims=True)
        np.save(os.path.join(dense_dir, "embeddings.npy"), dense_embeddings)
        with open(os.path.join(dense_dir, "doc_ids.json"), 'w') as f:
            json.dump(doc_ids, f)
        
        # Build FAISS for dense
        d = dense_embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(dense_embeddings)
        faiss.write_index(index, os.path.join(dense_dir, "index.faiss"))
        
        # Save sparse
        logger.info("Saving sparse embeddings...")
        with open(os.path.join(sparse_dir, "index.npz"), 'wb') as f:
            pickle.dump(all_sparse, f)
        with open(os.path.join(sparse_dir, "doc_ids.json"), 'w') as f:
            json.dump(doc_ids, f)
        
        # Save colbert
        logger.info("Saving ColBERT vectors...")
        np.save(os.path.join(colbert_dir, "colbert_vecs.npy"), np.array(all_colbert, dtype=object), allow_pickle=True)
        with open(os.path.join(colbert_dir, "doc_ids.json"), 'w') as f:
            json.dump(doc_ids, f)
        
        logger.info("✓ All three representations saved successfully!")


class SpladeIndexer:
    def __init__(self, model_name: str = "naver/splade-cocondenser-ensembledistil", output_dir: str = "indices", batch_size: int = 128):
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = DEVICE
        self.batch_size = batch_size
        
    def build(self, documents: List[Dict[str, Any]], domain: str):
        logger.info(f"Initializing SPLADE Indexer with {self.model_name}")
        
        index_dir = os.path.join(self.output_dir, domain, "splade")
        os.makedirs(index_dir, exist_ok=True)
        
        # Paths
        index_path = os.path.join(index_dir, "index.npz") # Sparse matrix
        ids_path = os.path.join(index_dir, "doc_ids.json")
        
        if os.path.exists(index_path) and os.path.exists(ids_path):
            logger.info(f"SPLADE index already exists for {domain}. Skipping.")
            return

        from transformers import AutoModelForMaskedLM, AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        model.to(self.device)
        
        # Enable DataParallel if multiple GPUs are available
        if torch.cuda.device_count() > 1:
            logger.info(f"🚀 Using {torch.cuda.device_count()} GPUs for SPLADE encoding!")
            model = torch.nn.DataParallel(model)
            
        model.eval()
        
        # Function to encode
        def encode_batch(texts):
            with torch.no_grad():
                # Tokenize needs to happen on CPU usually before moving to device, 
                # but for simplicity we tokenize centrally then move tensors.
                # For optimal multi-gpu, tokens should be moved to correct device by DataParallel, 
                # but DataParallel expects tensor inputs.
                inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
                
                # Move inputs to device (if using DataParallel, it splits them)
                # But inputs is a dict of tensors.
                if torch.cuda.device_count() > 1:
                   # For DataParallel we typically pass inputs directly but they need to be on cuda:0?
                   # Actually simple DataParallel usually requires inputs on the main device.
                   input_ids = inputs["input_ids"].to(self.device)
                   attention_mask = inputs["attention_mask"].to(self.device)
                   token_type_ids = inputs.get("token_type_ids")
                   if token_type_ids is not None:
                       token_type_ids = token_type_ids.to(self.device)
                       outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                   else:
                       outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    inputs = inputs.to(self.device)
                    outputs = model(**inputs)

                logits = outputs.logits
                
                # SPLADE logic: log(1 + relu(logits)) * mask
                # If DataParallel used, logits are gathered back to self.device (cuda:0)
                
                # We need attention mask on the same device as logits for multiplication
                if torch.cuda.device_count() > 1:
                     mask = inputs["attention_mask"].to(logits.device).unsqueeze(-1)
                else:
                     mask = inputs["attention_mask"].unsqueeze(-1)
                     
                values = torch.log(1 + torch.relu(logits)) * mask
                # Max pooling
                sparse_vecs = torch.max(values, dim=1).values
                return sparse_vecs.cpu()

        logger.info(f"Encoding {len(documents)} documents with SPLADE...")
        
        # Batch processing
        all_sparse_vecs = []
        texts = [doc['text'] for doc in documents]
        
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Encoding"):
            batch_texts = texts[i : i + self.batch_size]
            sparse_batch = encode_batch(batch_texts)
            all_sparse_vecs.append(sparse_batch)
            
        if not all_sparse_vecs:
             logger.warning("No documents encoded.")
             return

        full_tensor = torch.cat(all_sparse_vecs, dim=0)
        
        # Convert to scipy sparse matrix to save space
        # Thresholding could be applied here (e.g. keep only > 0)
        # SPLADE outputs are already Relu'd so they are >= 0.
        # But we only want non-zero elements
        
        numpy_matrix = full_tensor.numpy()
        sparse_matrix = scipy.sparse.csr_matrix(numpy_matrix)
        
        logger.info(f"Saving SPLADE index to {index_path}...")
        scipy.sparse.save_npz(index_path, sparse_matrix)
        
        # Save IDs
        doc_ids = [doc['id'] for doc in documents]
        with open(ids_path, 'w') as f:
            json.dump(doc_ids, f)
            
        logger.info("SPLADE Indexing complete.")

def main():
    parser = argparse.ArgumentParser(description="Build Indices for Retrieval Task A")
    parser.add_argument("--domains", nargs="+", default=["clapnq", "cloud", "fiqa", "govt"], help="Domains to index")
    parser.add_argument("--models", nargs="+", default=["bge", "bm25"], choices=["bge", "bm25", "elser", "bge-m3", "splade"], help="Models to use")
    parser.add_argument("--data_dir", default="data/passage_level_processed", help="Path to processed data")
    parser.add_argument("--output_dir", default="indices", help="Path to save indices")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set seed again with user-provided value
    set_seed(args.seed)
    logger.info(f"Random seed set to {args.seed} for reproducibility")
    
    for domain in args.domains:
        logger.info(f"--- Processing Domain: {domain} ---")
        try:
            documents = load_corpus(domain, args.data_dir)
            
            if "bge" in args.models:
                indexer = BGEIndexer(model_name="BAAI/bge-base-en-v1.5", output_dir=args.output_dir, index_subdir="bge")
                indexer.build(documents, domain)
                
            if "bge-m3" in args.models:
                # Usar BGEM3Indexer específico para aprovechar capacidades multi-vector
                indexer = BGEM3Indexer(model_name="BAAI/bge-m3", output_dir=args.output_dir, index_subdir="bge-m3", mode="all")
                indexer.build(documents, domain)
                
            if "bm25" in args.models:
                indexer = BM25Indexer(output_dir=args.output_dir)
                indexer.build(documents, domain)
                
            if "elser" in args.models:
                indexer = ELSERIndexer(output_dir=args.output_dir)
                indexer.build(documents, domain)

            if "splade" in args.models:
                indexer = SpladeIndexer(output_dir=args.output_dir)
                indexer.build(documents, domain)
                
        except Exception as e:
            logger.error(f"Failed to index {domain}: {e}")

if __name__ == "__main__":
    main()
