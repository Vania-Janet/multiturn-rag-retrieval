#!/usr/bin/env python3
"""
Hierarchical Chunking Strategy.

This script implements a structure-aware chunking approach.
It attempts to split documents while preserving semantic boundaries 
like paragraphs and sentences.
"""

import argparse
import json
import logging
import os
from typing import Dict, List, Any
from tqdm import tqdm

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    raise ImportError("Please install langchain-text-splitters: pip install langchain-text-splitters")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_jsonl(data: List[Dict[str, Any]], file_path: str):
    """Save data to a JSONL file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def process_documents(
    documents: List[Dict[str, Any]], 
    strategy: str,
    chunk_size: int = 1000,
    overlap: int = 200
) -> List[Dict[str, Any]]:
    """
    Split documents into chunks respecting hierarchy.
    
    Args:
        documents: List of document dictionaries.
        strategy: Chunking strategy (e.g., 'semantic').
        chunk_size: Target size of each chunk.
        overlap: Overlap between chunks.
        
    Returns:
        List of chunk dictionaries.
    """
    # The 'semantic' strategy uses recursive splitting to keep related text together.
    # It prioritizes splitting on paragraphs (\n\n), then lines (\n), then sentences.
    separators = ["\n\n", "\n", ". ", " ", ""]
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=separators,
        is_separator_regex=False,
    )
    
    chunked_data = []
    
    for doc in tqdm(documents, desc="Chunking documents"):
        doc_id = doc.get("id") or doc.get("_id")
        text = doc.get("text", "")
        title = doc.get("title", "")
        
        if not text:
            continue
            
        # Combine title and text
        full_text = f"{title}\n\n{text}" if title else text
        
        chunks = splitter.split_text(full_text)
        
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            
            chunk_entry = {
                "id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "original_id": doc_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "title": title,
                    "source": doc.get("url", ""),
                    "strategy": strategy,
                    "type": "hierarchical"
                }
            }
            chunked_data.append(chunk_entry)
            
    return chunked_data

def main():
    parser = argparse.ArgumentParser(description="Hierarchical Chunking")
    parser.add_argument("--input", required=True, help="Path to input JSONL file")
    parser.add_argument("--output", required=True, help="Path to output JSONL file")
    parser.add_argument("--strategy", type=str, default="semantic", choices=["semantic"], help="Chunking strategy")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Target chunk size in characters")
    parser.add_argument("--overlap", type=int, default=200, help="Overlap size in characters")
    
    args = parser.parse_args()
    
    logger.info(f"Starting hierarchical chunking: {args.input} -> {args.output}")
    logger.info(f"Config: strategy={args.strategy}, size={args.chunk_size}")
    
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return

    try:
        documents = load_jsonl(args.input)
        logger.info(f"Loaded {len(documents)} documents")
        
        chunks = process_documents(documents, args.strategy, args.chunk_size, args.overlap)
        logger.info(f"Generated {len(chunks)} chunks")
        
        save_jsonl(chunks, args.output)
        logger.info("Processing complete")
        
    except Exception as e:
        logger.exception(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
