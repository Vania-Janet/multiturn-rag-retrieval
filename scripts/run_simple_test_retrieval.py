#!/usr/bin/env python3
"""
Simple script to run test set retrieval directly without the complex run.py
"""
import os
os.environ['HF_HOME'] = '/workspace/cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/workspace/cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/cache/transformers'

import sys
sys.path.insert(0, '/workspace/mt-rag-benchmark/task_a_retrieval')

import json
from pathlib import Path
from tqdm import tqdm

# Import retrieval components directly
from src.pipeline.retrieval.sparse import SPLADERetriever
from src.pipeline.retrieval.dense import DenseRetriever, BGERetriever
from src.pipeline.retrieval.fusion import reciprocal_rank_fusion
from src.pipeline.reranking import BGEReranker

def load_queries(query_file):
    """Load test queries."""
    queries = []
    with open(query_file, 'r') as f:
        for line in f:
            queries.append(json.loads(line.strip()))
    return queries

def run_hybrid_retrieval(domain, queries, use_voyage=False, top_k=10):
    """Run hybrid SPLADE + Dense retrieval."""
    print(f"\n🔧 Initializing retrievers for {domain}...")
    
    # Initialize SPLADE
    index_path = Path(f"indices/{domain}/splade")
    sparse_retriever = SPLADERetriever(
        model_name="naver/splade-cocondenser-ensembledistil",
        index_path=index_path,
        config={}
    )
    print("  ✅ SPLADE loaded")
    
    # Initialize Dense (Voyage or BGE)
    if use_voyage:
        index_path = Path(f"indices/{domain}/voyage")
        dense_retriever = VoyageRetriever(
            model_name="voyage-2",
            index_path=index_path,
            config={}
        )
        print("  ✅ Voyage loaded")
    else:
        index_path = Path(f"indices/{domain}/bge")
        dense_retriever = BGERetriever(
            model_name="BAAI/bge-base-en-v1.5",
            index_path=index_path,
            config={}
        )
        print("  ✅ BGE loaded")
    
    # Run retrieval
    results = []
    print(f"\n🔍 Retrieving for {len(queries)} queries...")
    
    for query in tqdm(queries):
        task_id = query["task_id"]
        question = query["question"]
        
        # Get results from both retrievers
        sparse_results = sparse_retriever.search(question, top_k=300)
        dense_results = dense_retriever.search(question, top_k=300)
        
        # Fuse with RRF
        fused = reciprocal_rank_fusion(
            {
                "sparse": sparse_results,
                "dense": dense_results
            },
            k=60
        )
        
        # Take top k
        top_results = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Format result
        contexts = []
        for doc_id, score in top_results:
            # Get document text
            doc_text = sparse_retriever.get_document_text(doc_id)
            contexts.append({
                "document_id": doc_id,
                "text": doc_text,
                "score": float(score)
            })
        
        results.append({
            "task_id": task_id,
            "question": question,
            "Collection": domain,
            "contexts": contexts
        })
    
    return results

def run_hybrid_with_reranking(domain, queries, top_k=10):
    """Run hybrid + reranking for Cloud."""
    print(f"\n🔧 Initializing retrievers and reranker for {domain}...")
    
    # Initialize SPLADE
    index_path = Path(f"indices/{domain}/splade")
    sparse_retriever = SPLADERetriever(
        model_name="naver/splade-cocondenser-ensembledistil",
        index_path=index_path,
        config={}
    )
    print("  ✅ SPLADE loaded")
    
    # Initialize BGE
    index_path = Path(f"indices/{domain}/bge")
    dense_retriever = BGERetriever(
        model_name="BAAI/bge-base-en-v1.5",
        index_path=index_path,
        config={}
    )
    print("  ✅ BGE loaded")
    
    # Initialize reranker
    reranker = BGEReranker(
        model_name="BAAI/bge-reranker-v2-m3",
        use_fp16=True,
        batch_size=32
    )
    print("  ✅ BGE Reranker loaded")
    
    # Run retrieval
    results = []
    print(f"\n🔍 Retrieving for {len(queries)} queries...")
    
    for query in tqdm(queries):
        task_id = query["task_id"]
        question = query["question"]
        
        # Get results from both retrievers
        sparse_results = sparse_retriever.search(question, top_k=300)
        dense_results = dense_retriever.search(question, top_k=300)
        
        # Fuse with RRF
        fused = reciprocal_rank_fusion(
            {
                "sparse": sparse_results,
                "dense": dense_results
            },
            k=60
        )
        
        # Take top 100 for reranking
        top_100 = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:100]
        
        # Prepare for reranking
        candidates = []
        for doc_id, _ in top_100:
            doc_text = sparse_retriever.get_document_text(doc_id)
            candidates.append({
                "document_id": doc_id,
                "text": doc_text
            })
        
        # Rerank
        reranked = reranker.rerank(question, candidates, top_k=top_k)
        
        results.append({
            "task_id": task_id,
            "question": question,
            "Collection": domain,
            "contexts": reranked
        })
    
    return results

def main():
    # Configuration
    CONFIGS = {
        "clapnq": {"use_voyage": True, "rerank": False},
        "fiqa": {"use_voyage": False, "rerank": False},
        "govt": {"use_voyage": True, "rerank": False},
        "cloud": {"use_voyage": False, "rerank": True}
    }
    
    output_base = Path("experiments/test_submission")
    output_base.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for domain, config in CONFIGS.items():
        print("\n" + "="*60)
        print(f"📂 Processing {domain.upper()}")
        print("="*60)
        
        # Load queries
        query_file = f"data/retrieval_tasks/{domain}/{domain}_test_questions.jsonl"
        queries = load_queries(query_file)
        print(f"📖 Loaded {len(queries)} queries")
        
        # Run retrieval
        if config["rerank"]:
            results = run_hybrid_with_reranking(domain, queries, top_k=10)
        else:
            results = run_hybrid_retrieval(domain, queries, use_voyage=config["use_voyage"], top_k=10)
        
        # Save results
        output_file = output_base / domain / "retrieval_results.jsonl"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        print(f"✅ Saved {len(results)} results to {output_file}")
        all_results.extend(results)
    
    # Create submission file
    print("\n" + "="*60)
    print("📦 Creating submission file...")
    print("="*60)
    
    submission_file = "submission.jsonl"
    with open(submission_file, 'w') as f:
        for result in all_results:
            submission_record = {
                "task_id": result["task_id"],
                "Collection": result["Collection"],
                "contexts": result["contexts"]
            }
            f.write(json.dumps(submission_record, ensure_ascii=False) + '\n')
    
    print(f"✅ Created {submission_file} with {len(all_results)} tasks")
    
    # Run format checker
    print("\n" + "="*60)
    print("✅ Running format checker...")
    print("="*60)
    
    import subprocess
    result = subprocess.run([
        "python3", "src/pipeline/evaluation/format_checker.py",
        "--mode", "retrieval_taska",
        "--prediction_file", submission_file,
        "--input_file", "src/pipeline/evaluation/rag_taskAC.jsonl"
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr)
    
    print("\n🎉 DONE!")

if __name__ == "__main__":
    main()
