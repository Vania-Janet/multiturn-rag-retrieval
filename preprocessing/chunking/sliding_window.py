#!/usr/bin/env python3
"""
Sliding Window Chunking Strategy.

This script implements a fixed-size sliding window chunking approach.
It splits documents into chunks of a specified size with a defined overlap.
"""

import argparse
import json
import logging
import os
from typing import Dict, List, Any
from tqdm import tqdm

# Try to import LangChain, handle if missing (though it should be installed)
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
    chunk_size: int, 
    overlap: int
) -> List[Dict[str, Any]]:
    """
    Split documents into chunks using a sliding window approach.
    
    Args:
        documents: List of document dictionaries.
        chunk_size: Maximum size of each chunk (in characters).
        overlap: Overlap between chunks (in characters).
        
    Returns:
        List of chunk dictionaries.
    """
    # We use RecursiveCharacterTextSplitter as it's robust and handles
    # word boundaries better than naive string slicing.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunked_data = []
    
    for doc in tqdm(documents, desc="Chunking documents"):
        doc_id = doc.get("id") or doc.get("_id")
        text = doc.get("text", "")
        title = doc.get("title", "")
        
        if not text:
            continue
            
        # Combine title and text for better context if title exists
        full_text = f"{title}\n{text}" if title else text
        
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
                    "strategy": "sliding_window"
                }
            }
            chunked_data.append(chunk_entry)
            
    return chunked_data

def main():
    parser = argparse.ArgumentParser(description="Sliding Window Chunking")
    parser.add_argument("--input", required=True, help="Path to input JSONL file")
    parser.add_argument("--output", required=True, help="Path to output JSONL file")
    parser.add_argument("--chunk_size", type=int, default=512, help="Chunk size in characters")
    parser.add_argument("--overlap", type=int, default=128, help="Overlap size in characters")
    
    args = parser.parse_args()
    
    logger.info(f"Starting sliding window chunking: {args.input} -> {args.output}")
    logger.info(f"Config: size={args.chunk_size}, overlap={args.overlap}")
    
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return

    try:
        documents = load_jsonl(args.input)
        logger.info(f"Loaded {len(documents)} documents")
        
        chunks = process_documents(documents, args.chunk_size, args.overlap)
        logger.info(f"Generated {len(chunks)} chunks")
        
        save_jsonl(chunks, args.output)
        logger.info("Processing complete")
        
    except Exception as e:
        logger.exception(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
