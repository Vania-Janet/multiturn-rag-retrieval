
import logging
from typing import Dict, List, Tuple
import re

logger = logging.getLogger(__name__)

def get_parent_id(chunk_id: str) -> str:
    """
    Derive parent ID from chunk ID.
    Assumes chunk ID format: parent_id-start-end or just parent_id
    """
    # Heuristic: split by '-' and check if last parts are numbers
    parts = chunk_id.split('-')
    if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
        return "-".join(parts[:-2])
    return chunk_id

def clean_merge(left: str, right: str) -> str:
    """
    Merge two text segments, detecting and removing textual overlap.
    Safely falls back to newline separation if no overlap found.
    """
    # Check for largest overlap suffix of left matching prefix of right
    # Optimization: Only check limited window to improve speed
    min_len = min(len(left), len(right))
    check_len = min(min_len, 2000) # Check up to 2000 chars is usually enough for sliding windows
    
    # Iterate from largest possible overlap downwards
    # Minimum overlap to consider is 20 chars to avoid accidental hits on common words
    for i in range(check_len, 19, -1):
        if left.endswith(right[:i]):
            return left + right[i:]
            
    return left + "\n\n" + right

def build_parent_store(corpus: Dict[str, str]) -> Dict[str, str]:
    """
    Reconstruct parent documents from a corpus of chunks.
    
    Args:
        corpus: Dict mapping chunk_id -> chunk_text
        
    Returns:
        Dict mapping parent_id -> parent_text
    """
    logger.info("Building parent document store from chunks...")
    
    # 1. Group chunks by parent ID
    parent_groups: Dict[str, List[Tuple[int, str]]] = {}
    
    for chunk_id, text in corpus.items():
        parent_id = get_parent_id(chunk_id)
        
        # Extract start offset
        try:
            parts = chunk_id.split('-')
            start_offset = int(parts[-2])
        except (ValueError, IndexError):
            start_offset = 0 # Fallback
            
        if parent_id not in parent_groups:
            parent_groups[parent_id] = []
        parent_groups[parent_id].append((start_offset, text))
        
    # 2. Reconstruct text
    parent_store = {}
    for parent_id, chunks in parent_groups.items():
        # Sort by offset
        chunks.sort(key=lambda x: x[0])
        
        if len(chunks) == 1:
            full_text = chunks[0][1]
        else:
            full_text = chunks[0][1]
            
            # Smart reconstruction with overlap detection
            for i in range(1, len(chunks)):
                _, text = chunks[i]
                full_text = clean_merge(full_text, text)
        
        # Truncate to first 2000 chars for faster tokenization (BGE only uses first 512 tokens)
        # This preserves most relevant context while making reranking 5x faster
        if len(full_text) > 2000:
            full_text = full_text[:2000]
                
        parent_store[parent_id] = full_text
        
    logger.info(f"Reconstructed {len(parent_store)} parent documents from {len(corpus)} chunks")
    return parent_store
