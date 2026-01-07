#!/usr/bin/env python3
"""
Analyze keyword preservation in query rewrites for BM25 compatibility.

This script compares original queries vs rewritten queries to identify:
1. Lost technical terms (acronyms, entities, product names)
2. Queries with significant keyword changes
3. Correlation between keyword loss and performance degradation
"""

import json
import re
from pathlib import Path
from collections import defaultdict
import argparse

def extract_keywords(text):
    """Extract potential technical keywords from text."""
    # Extract all-caps acronyms (AWS, VPC, API, etc.)
    acronyms = set(re.findall(r'\b[A-Z]{2,}\b', text))
    
    # Extract CamelCase terms (AutoScaling, CloudFormation, etc.)
    camelcase = set(re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', text))
    
    # Extract hyphenated technical terms (auto-scaling, machine-learning, etc.)
    hyphenated = set(re.findall(r'\b[a-z]+-[a-z]+(?:-[a-z]+)*\b', text.lower()))
    
    # Extract version numbers (v2, 3.0, etc.)
    versions = set(re.findall(r'\bv?\d+(?:\.\d+)*\b', text.lower()))
    
    return {
        'acronyms': acronyms,
        'camelcase': camelcase,
        'hyphenated': hyphenated,
        'versions': versions
    }

def analyze_rewrite_quality(original, rewritten):
    """Analyze what was lost or changed in the rewrite."""
    orig_keywords = extract_keywords(original)
    rewr_keywords = extract_keywords(rewritten)
    
    lost_keywords = {
        'acronyms': orig_keywords['acronyms'] - rewr_keywords['acronyms'],
        'camelcase': orig_keywords['camelcase'] - rewr_keywords['camelcase'],
        'hyphenated': orig_keywords['hyphenated'] - rewr_keywords['hyphenated'],
        'versions': orig_keywords['versions'] - rewr_keywords['versions']
    }
    
    total_lost = sum(len(v) for v in lost_keywords.values())
    total_original = sum(len(v) for v in orig_keywords.values())
    
    loss_rate = total_lost / total_original if total_original > 0 else 0
    
    return {
        'lost_keywords': lost_keywords,
        'total_lost': total_lost,
        'total_original': total_original,
        'loss_rate': loss_rate
    }

def main():
    parser = argparse.ArgumentParser(description="Analyze keyword loss in query rewrites")
    parser.add_argument("--experiment", required=True, help="Experiment directory (e.g., experiments/01-query/bm25_r1_condensation)")
    parser.add_argument("--domain", required=True, help="Domain name (e.g., cloud)")
    parser.add_argument("--baseline", help="Baseline experiment for performance comparison")
    
    args = parser.parse_args()
    
    # Load rewrite log
    rewrite_log_path = Path(args.experiment) / args.domain / "query_rewrites.jsonl"
    
    if not rewrite_log_path.exists():
        print(f"❌ Query rewrite log not found: {rewrite_log_path}")
        print(f"Run the experiment first to generate the log.")
        return
    
    # Load rewrites
    rewrites = []
    with open(rewrite_log_path, 'r') as f:
        for line in f:
            rewrites.append(json.loads(line))
    
    print(f"\n📊 Analyzing {len(rewrites)} query rewrites from {args.domain}")
    print("="*80)
    
    # Analyze each rewrite
    high_loss_queries = []
    total_loss_rate = 0
    
    for entry in rewrites:
        original = entry['original']
        rewritten = entry['rewritten'][0] if isinstance(entry['rewritten'], list) else entry['rewritten']
        
        analysis = analyze_rewrite_quality(original, rewritten)
        total_loss_rate += analysis['loss_rate']
        
        if analysis['total_lost'] > 0:
            high_loss_queries.append({
                'task_id': entry['task_id'],
                'original': original,
                'rewritten': rewritten,
                'analysis': analysis
            })
    
    avg_loss_rate = total_loss_rate / len(rewrites) if rewrites else 0
    
    print(f"\n📈 Overall Statistics:")
    print(f"  - Total queries analyzed: {len(rewrites)}")
    print(f"  - Queries with keyword loss: {len(high_loss_queries)} ({len(high_loss_queries)/len(rewrites)*100:.1f}%)")
    print(f"  - Average keyword loss rate: {avg_loss_rate*100:.1f}%")
    
    # Show worst cases
    if high_loss_queries:
        print(f"\n🔍 Top 10 Queries with Most Keyword Loss:")
        print("="*80)
        
        sorted_losses = sorted(high_loss_queries, key=lambda x: x['analysis']['total_lost'], reverse=True)
        
        for i, entry in enumerate(sorted_losses[:10], 1):
            print(f"\n{i}. Task ID: {entry['task_id']}")
            print(f"   Lost {entry['analysis']['total_lost']} keywords ({entry['analysis']['loss_rate']*100:.1f}% loss)")
            
            if entry['analysis']['lost_keywords']['acronyms']:
                print(f"   ❌ Lost Acronyms: {', '.join(entry['analysis']['lost_keywords']['acronyms'])}")
            if entry['analysis']['lost_keywords']['hyphenated']:
                print(f"   ❌ Lost Hyphenated: {', '.join(entry['analysis']['lost_keywords']['hyphenated'])}")
            if entry['analysis']['lost_keywords']['versions']:
                print(f"   ❌ Lost Versions: {', '.join(entry['analysis']['lost_keywords']['versions'])}")
            
            print(f"   📝 Original: {entry['original'][:100]}...")
            print(f"   ✏️  Rewritten: {entry['rewritten'][:100]}...")
    
    # If baseline provided, correlate with performance
    if args.baseline:
        baseline_path = Path(args.baseline) / args.domain / "metrics.json"
        current_path = Path(args.experiment) / args.domain / "metrics.json"
        
        if baseline_path.exists() and current_path.exists():
            with open(baseline_path, 'r') as f:
                baseline_metrics = json.load(f)
            with open(current_path, 'r') as f:
                current_metrics = json.load(f)
            
            baseline_ndcg5 = baseline_metrics.get('NDCG@5', baseline_metrics.get('ndcg_cut_5', 0))
            current_ndcg5 = current_metrics.get('NDCG@5', current_metrics.get('ndcg_cut_5', 0))
            
            print(f"\n📉 Performance Impact:")
            print(f"  - Baseline NDCG@5: {baseline_ndcg5:.4f}")
            print(f"  - Current NDCG@5: {current_ndcg5:.4f}")
            print(f"  - Delta: {current_ndcg5 - baseline_ndcg5:.4f} ({(current_ndcg5 - baseline_ndcg5)/baseline_ndcg5*100:+.1f}%)")
            print(f"\n💡 Hypothesis: Average keyword loss of {avg_loss_rate*100:.1f}% correlates with {(current_ndcg5 - baseline_ndcg5)/baseline_ndcg5*100:+.1f}% performance change")

if __name__ == "__main__":
    main()
