#!/usr/bin/env python3
"""
Build Processed Corpus.

This script orchestrates the preprocessing of raw corpus files.
It filters empty passages and prepares the data for indexing.
Chunking has been removed as per user request; raw passages are used directly.

Usage:
    python preprocessing/build_processed_corpus.py --domains all
    python preprocessing/build_processed_corpus.py --domains fiqa govt
"""

import argparse
import logging
import os
import sys
import json
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
# Paths are relative to the project root. Run this script from the project root.
DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "passage_level_raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "passage_level_processed")

DOMAINS = ["clapnq", "cloud", "fiqa", "govt"]

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

def process_domain(domain: str):
    """Process a single domain: Load, Filter, Save."""
    input_file = os.path.join(RAW_DIR, f"{domain}.jsonl")
    output_file = os.path.join(PROCESSED_DIR, domain, "corpus.jsonl")

    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return

    logger.info(f"Processing {domain}...")
    
    try:
        documents = load_jsonl(input_file)
        logger.info(f"Loaded {len(documents)} documents for {domain}")

        # Filter out empty passages (specifically found in Cloud and FiQA)
        original_count = len(documents)
        # Filter logic: keep if text is not None and stripped length > 0
        documents = [doc for doc in documents if doc.get("text") and doc.get("text", "").strip()]
        
        if len(documents) < original_count:
            logger.info(f"Filtered {original_count - len(documents)} empty documents for {domain}")
        
        logger.info(f"Saving {len(documents)} documents to {output_file}")
        save_jsonl(documents, output_file)
        
    except Exception as e:
        logger.error(f"Error processing {domain}: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Build Processed Corpus")
    parser.add_argument(
        "--domains", 
        nargs="+", 
        default=["all"], 
        help="Domains to process (or 'all')"
    )
    
    args = parser.parse_args()
    
    domains_to_process = args.domains
    if "all" in domains_to_process:
        domains_to_process = DOMAINS
        
    for domain in domains_to_process:
        if domain in DOMAINS:
            process_domain(domain)
        else:
            logger.warning(f"Unknown domain: {domain}")

if __name__ == "__main__":
    main()
