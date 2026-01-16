#!/usr/bin/env python3
"""
Build search indices for all domains and retrieval models.

Usage:
    # Build all indices for all domains
    python scripts/build_indices.py --all
    
    # Build specific index for specific domain
    python scripts/build_indices.py --domain clapnq --model bm25
    
    # Build all models for one domain
    python scripts/build_indices.py --domain fiqa --model all
    
    # Rebuild existing indices
    python scripts/build_indices.py --all --force
"""

import argparse
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.logger import setup_logger
from utils.hf_manager import HFManager


DOMAINS = ["clapnq", "fiqa", "govt", "cloud"]
MODELS = ["bm25", "elser", "bge-m3", "bge-base-1.5", "splade", "bgem3_dense", "bgem3_sparse", "bgem3_colbert", "bgem3_all"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build search indices for RAG benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--domain", "-d",
        help=f"Domain name or 'all'. Options: {', '.join(DOMAINS)}"
    )
    
    parser.add_argument(
        "--model", "-m",
        help=f"Model name or 'all'. Options: {', '.join(MODELS)}"
    )
    
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Build all indices for all domains"
    )
    
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing processed corpora (default: data/processed/)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("indices"),
        help="Output directory for indices (default: indices/)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild existing indices"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for indexing (default: 100)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging"
    )
    
    return parser.parse_args()


def resolve_domains(domain_name, build_all):
    """Resolve domain name to list of domains."""
    if build_all or domain_name == "all":
        return DOMAINS
    elif domain_name in DOMAINS:
        return [domain_name]
    elif domain_name is None:
        raise ValueError("Must specify --domain or --all")
    else:
        raise ValueError(f"Unknown domain: {domain_name}")


def resolve_models(model_name, build_all):
    """Resolve model name to list of models."""
    if build_all or model_name == "all":
        return MODELS
    elif model_name in MODELS:
        return [model_name]
    elif model_name is None:
        raise ValueError("Must specify --model or --all")
    else:
        raise ValueError(f"Unknown model: {model_name}")


def build_index(domain, model, corpus_dir, output_dir, force, batch_size, logger):
    """Build a single index."""
    # Check if corpus exists
    corpus_path = corpus_dir / domain / "corpus.jsonl"
    if not corpus_path.exists():
        logger.warning(f"Corpus not found: {corpus_path}")
        return False
    
    # Check if index already exists and is complete
    index_dir = output_dir / domain / model
    done_flag = index_dir / "_SUCCESS"
    
    if done_flag.exists() and not force:
        logger.info(f"Index already exists and is complete (use --force to rebuild): {index_dir}")
        return True
    
    if index_dir.exists() and not force:
        logger.warning(f"Index directory exists but seems incomplete (missing _SUCCESS). Rebuilding: {index_dir}")
    
    # Create output directory
    index_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Building {model} index for {domain}")
    logger.info(f"  Corpus: {corpus_path}")
    logger.info(f"  Output: {index_dir}")
    
    try:
        # Import indexing classes
        from pipeline.indexing.indexer import BGEIndexer, BM25Indexer, ELSERIndexer, SpladeIndexer, BGEM3Indexer, load_corpus
        
        # Load documents
        documents = load_corpus(domain, str(corpus_dir))
        
        if model == "bm25":
            logger.info("  Building BM25 index...")
            indexer = BM25Indexer(output_dir=str(output_dir))
            indexer.build(documents, domain)
            
        elif model == "elser":
            logger.info("  Building ELSER index...")
            indexer = ELSERIndexer(output_dir=str(output_dir))
            indexer.build(documents, domain)
            
        elif model == "splade":
            logger.info("  Building SPLADE index...")
            indexer = SpladeIndexer(output_dir=str(output_dir))
            indexer.build(documents, domain)
            
        elif model == "bge-m3":
            logger.info("  Building BGE-M3 index...")
            indexer = BGEIndexer(model_name="BAAI/bge-m3", output_dir=str(output_dir), index_subdir="bge-m3", batch_size=batch_size)
            indexer.build(documents, domain)

        elif model == "bgem3_dense":
            logger.info("  Building BGEM3 Dense index...")
            indexer = BGEM3Indexer(output_dir=str(output_dir), index_subdir="bgem3_dense", mode="dense", batch_size=batch_size)
            indexer.build(documents, domain)

        elif model == "bgem3_sparse":
            logger.info("  Building BGEM3 Sparse index...")
            indexer = BGEM3Indexer(output_dir=str(output_dir), index_subdir="bgem3_sparse", mode="sparse", batch_size=batch_size)
            indexer.build(documents, domain)

        elif model == "bgem3_colbert":
            logger.info("  Building BGEM3 ColBERT index...")
            # Reduce batch size for ColBERT
            colbert_batch = max(1, batch_size // 4)
            indexer = BGEM3Indexer(output_dir=str(output_dir), index_subdir="bgem3_colbert", mode="colbert", batch_size=colbert_batch)
            indexer.build(documents, domain)

        elif model == "bgem3_all":
            logger.info("  Building ALL BGEM3 indices (Dense, Sparse, ColBERT) in one pass...")
            # Reduce batch size as we are storing 3x tensors in memory/VRAM
            combined_batch = max(1, batch_size // 4) # Conservative for 24GB VRAM
            indexer = BGEM3Indexer(output_dir=str(output_dir), index_subdir="bgem3", mode="all", batch_size=combined_batch)
            indexer.build(documents, domain)
            
        elif model == "bge-base-1.5":
            logger.info("  Building BGE-base-en-v1.5 index (baseline)...")
            indexer = BGEIndexer(model_name="BAAI/bge-base-en-v1.5", output_dir=str(output_dir), index_subdir="bge", batch_size=batch_size)  # FIXED: Was bge-large
            indexer.build(documents, domain)
            
        # Mark as successful
        done_flag.touch()
        logger.info(f"  ✓ Index built successfully")
        return True
        
    except Exception as e:
        logger.error(f"  ✗ Failed to build index: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    
    # Ensure logs directory exists
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logger = setup_logger("build_indices", log_file=log_dir / "build_indices.log", level=log_level)
    
    # Resolve domains and models
    domains = resolve_domains(args.domain, args.all)
    models = resolve_models(args.model, args.all)
    
    logger.info(f"Building {len(models)} model(s) for {len(domains)} domain(s)")
    logger.info(f"Domains: {', '.join(domains)}")
    logger.info(f"Models: {', '.join(models)}")
    
    # Initialize HF Manager
    hf_manager = HFManager()
    
    # Build indices
    total_indices = len(domains) * len(models)
    current_index = 0
    failed_indices = []
    
    for domain in domains:
        for model in models:
            current_index += 1
            logger.info(f"\n{'='*80}")
            logger.info(f"Index {current_index}/{total_indices}: {domain}/{model}")
            logger.info(f"{'='*80}\n")
            
            success = build_index(
                domain=domain,
                model=model,
                corpus_dir=args.corpus_dir,
                output_dir=args.output_dir,
                force=args.force,
                batch_size=args.batch_size,
                logger=logger
            )
            
            if success:
                # Upload to Hugging Face if enabled
                if hf_manager.enabled:
                    # Construct path: indices/domain/model
                    # Note: Some models might have different internal structure, 
                    # but usually they are under output_dir/domain/model
                    index_path = args.output_dir / domain / model
                    if index_path.exists():
                        hf_manager.upload_directory(index_path)
            else:
                failed_indices.append(f"{domain}/{model}")
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Total indices: {total_indices}")
    logger.info(f"Successful: {total_indices - len(failed_indices)}")
    logger.info(f"Failed: {len(failed_indices)}")
    
    if failed_indices:
        logger.error("\nFailed indices:")
        for idx in failed_indices:
            logger.error(f"  - {idx}")
        sys.exit(1)
    else:
        logger.info("\n✓ All indices built successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
