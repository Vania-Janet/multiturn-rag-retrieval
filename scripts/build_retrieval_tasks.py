import json
import os
import logging
from pathlib import Path
from typing import List, Dict, Any
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DOMAINS = {
    "clapnq": "clapnq",
    "cloud": "cloud",
    "fiqa": "fiqa",
    "govt": "govt"
}

def get_domain_from_collection(collection_name: str) -> str:
    for key, domain in DOMAINS.items():
        if key in collection_name:
            return domain
    return "unknown"

def build_tasks(conversations_file: str, output_base_dir: str):
    logger.info(f"Loading conversations from {conversations_file}")
    with open(conversations_file, 'r') as f:
        conversations = json.load(f)
    
    logger.info(f"Loaded {len(conversations)} conversations")
    
    tasks_by_domain = {d: [] for d in DOMAINS.values()}
    
    for conv in conversations:
        conv_id = conv.get("conversation_id")
        # If conversation_id is missing, generate one or skip? 
        # The sample had it. The head output didn't show it explicitly in the first few bytes, 
        # but let's assume it's there or use index.
        if not conv_id:
            conv_id = conv.get("id", str(hash(json.dumps(conv))))

        collection_name = conv.get("retriever", {}).get("collection", {}).get("name", "")
        domain = get_domain_from_collection(collection_name)
        
        if domain == "unknown":
            continue
            
        messages = conv.get("messages", [])
        history = []
        
        turn_count = 0
        for i, msg in enumerate(messages):
            speaker = msg.get("speaker")
            text = msg.get("text")
            
            history.append({"speaker": speaker, "text": text})
            
            if speaker == "user":
                turn_count += 1
                task_id = f"{conv_id}::{turn_count}"
                
                task = {
                    "task_id": task_id,
                    "conversation_id": conv_id,
                    "turn": turn_count,
                    "Collection": collection_name,
                    "input": list(history) # Copy history up to this point
                }
                
                tasks_by_domain[domain].append(task)
    
    # Save tasks
    for domain, tasks in tasks_by_domain.items():
        if not tasks:
            continue
            
        output_dir = Path(output_base_dir) / domain
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{domain}_tasks.jsonl"
        logger.info(f"Saving {len(tasks)} tasks to {output_file}")
        
        with open(output_file, 'w') as f:
            for task in tasks:
                f.write(json.dumps(task) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/conversations/conversations.json")
    parser.add_argument("--output", default="data/retrieval_tasks")
    args = parser.parse_args()
    
    build_tasks(args.input, args.output)
