#!/usr/bin/env python3
"""
Validate submission file format for MT-RAG Task A.

Checks:
- Valid JSONL format
- Required fields present
- Correct data types
- Score ranges
- Document ID format
- No duplicate task_ids
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Set
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate MT-RAG Task A submission format"
    )
    
    parser.add_argument(
        "submission_file",
        type=Path,
        help="Submission JSONL file to validate"
    )
    
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict validation (fail on warnings)"
    )
    
    parser.add_argument(
        "--expected-queries",
        type=int,
        help="Expected number of queries (optional check)"
    )
    
    return parser.parse_args()


def validate_record_format(
    record: Dict[str, Any],
    line_num: int,
    logger: logging.Logger
) -> tuple[bool, List[str]]:
    """
    Validate a single submission record.
    
    Returns:
        (is_valid, errors)
    """
    errors = []
    
    # Check required fields
    required_fields = ["task_id", "run_name", "retrieved_docs"]
    for field in required_fields:
        if field not in record:
            errors.append(f"Line {line_num}: Missing required field '{field}'")
    
    if errors:
        return False, errors
    
    # Validate task_id format
    task_id = record["task_id"]
    if not isinstance(task_id, str):
        errors.append(f"Line {line_num}: task_id must be string")
    elif "::" not in task_id:
        errors.append(
            f"Line {line_num}: task_id should be format 'conv_id::turn', "
            f"got '{task_id}'"
        )
    
    # Validate run_name
    if not isinstance(record["run_name"], str):
        errors.append(f"Line {line_num}: run_name must be string")
    elif len(record["run_name"]) == 0:
        errors.append(f"Line {line_num}: run_name cannot be empty")
    
    # Validate retrieved_docs
    retrieved_docs = record["retrieved_docs"]
    
    if not isinstance(retrieved_docs, list):
        errors.append(f"Line {line_num}: retrieved_docs must be a list")
        return False, errors
    
    if len(retrieved_docs) == 0:
        errors.append(f"Line {line_num}: retrieved_docs cannot be empty")
    
    if len(retrieved_docs) > 1000:
        errors.append(
            f"Line {line_num}: Too many docs ({len(retrieved_docs)}), "
            "max 1000 allowed"
        )
    
    # Validate each document
    doc_ids_seen = set()
    
    for i, doc in enumerate(retrieved_docs):
        if not isinstance(doc, dict):
            errors.append(
                f"Line {line_num}, doc {i}: Document must be a dict"
            )
            continue
        
        # Check required doc fields
        if "document_id" not in doc:
            errors.append(
                f"Line {line_num}, doc {i}: Missing 'document_id'"
            )
        
        if "score" not in doc:
            errors.append(
                f"Line {line_num}, doc {i}: Missing 'score'"
            )
        
        # Validate document_id
        doc_id = doc.get("document_id")
        if not isinstance(doc_id, str):
            errors.append(
                f"Line {line_num}, doc {i}: document_id must be string"
            )
        elif len(doc_id) == 0:
            errors.append(
                f"Line {line_num}, doc {i}: document_id cannot be empty"
            )
        
        # Check for duplicate doc_ids in same query
        if doc_id in doc_ids_seen:
            errors.append(
                f"Line {line_num}, doc {i}: Duplicate document_id '{doc_id}'"
            )
        doc_ids_seen.add(doc_id)
        
        # Validate score
        score = doc.get("score")
        if not isinstance(score, (int, float)):
            errors.append(
                f"Line {line_num}, doc {i}: score must be numeric"
            )
        # Note: Scores can be negative (e.g., log-probs), don't enforce range
    
    is_valid = len(errors) == 0
    return is_valid, errors


def validate_submission(
    submission_path: Path,
    strict: bool = False,
    expected_queries: int = None,
    logger: logging.Logger = None
) -> bool:
    """
    Validate entire submission file.
    
    Returns:
        True if valid, False otherwise
    """
    if logger is None:
        logger = setup_logger("validate_submission")
    
    logger.info(f"Validating submission: {submission_path}")
    
    # Check file exists
    if not submission_path.exists():
        logger.error(f"File not found: {submission_path}")
        return False
    
    # Check file extension
    if submission_path.suffix != ".jsonl":
        logger.warning(f"File should have .jsonl extension")
    
    # Read and validate
    all_errors = []
    task_ids_seen: Set[str] = set()
    valid_records = 0
    total_records = 0
    
    try:
        with open(submission_path, "r") as f:
            for line_num, line in enumerate(f, start=1):
                total_records += 1
                
                # Parse JSON
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    all_errors.append(f"Line {line_num}: Invalid JSON - {e}")
                    continue
                
                # Validate record
                is_valid, errors = validate_record_format(record, line_num, logger)
                
                if errors:
                    all_errors.extend(errors)
                else:
                    valid_records += 1
                
                # Check for duplicate task_ids
                task_id = record.get("task_id")
                if task_id in task_ids_seen:
                    all_errors.append(
                        f"Line {line_num}: Duplicate task_id '{task_id}'"
                    )
                task_ids_seen.add(task_id)
    
    except Exception as e:
        logger.error(f"Failed to read file: {e}")
        return False
    
    # Print results
    logger.info("\n" + "="*80)
    logger.info("VALIDATION RESULTS")
    logger.info("="*80)
    logger.info(f"Total records: {total_records}")
    logger.info(f"Valid records: {valid_records}")
    logger.info(f"Invalid records: {total_records - valid_records}")
    logger.info(f"Total errors: {len(all_errors)}")
    
    if expected_queries and total_records != expected_queries:
        logger.warning(
            f"Expected {expected_queries} queries, got {total_records}"
        )
        if strict:
            all_errors.append(
                f"Query count mismatch: expected {expected_queries}, "
                f"got {total_records}"
            )
    
    # Print errors
    if all_errors:
        logger.error("\nERRORS FOUND:")
        for error in all_errors[:50]:  # Show first 50 errors
            logger.error(f"  {error}")
        
        if len(all_errors) > 50:
            logger.error(f"  ... and {len(all_errors) - 50} more errors")
    
    # Final verdict
    is_valid = len(all_errors) == 0
    
    logger.info("="*80)
    if is_valid:
        logger.info("✓ VALIDATION PASSED")
        logger.info(f"\nSubmission is ready for evaluation!")
    else:
        logger.error("✗ VALIDATION FAILED")
        logger.error(f"\nPlease fix the {len(all_errors)} error(s) above")
    logger.info("="*80)
    
    return is_valid


def main():
    args = parse_args()
    logger = setup_logger("validate_submission")
    
    is_valid = validate_submission(
        submission_path=args.submission_file,
        strict=args.strict,
        expected_queries=args.expected_queries,
        logger=logger
    )
    
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
