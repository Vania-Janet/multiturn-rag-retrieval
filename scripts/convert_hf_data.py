import json
import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Input HF data file (e.g. test.jsonl)")
    parser.add_argument("--output_file", type=str, required=True, help="Output file formatted for experiment (questions.jsonl)")
    args = parser.parse_args()

    # The HF data has "query", "positive", "negative".
    # The pipeline expects a format that usually includes an "_id" or uses the text as key.
    # To be safe, we'll try to reconstruct the format expected by `run.py`.
    # Based on previous `run.py` analysis:
    # It looks for "text" or "input" (list of turns).
    # It also looks for "task_id" or "_id".
    
    # Since we don't have IDs in the HF file (it's just query/pos/neg), we'll generate hash IDs 
    # OR we can assume these are single-turn queries.
    
    # Important: The USER wants to evaluate THIS data.
    # The benchmark evaluation (pytrec_eval) REQUIRES qrels (query_id -> doc_id -> score).
    # The HF test.jsonl contains positives, so we can GENERATE the qrels from it!
    
    print(f"Converting {args.input_file} -> {args.output_file} + qrels")
    
    tasks = []
    
    # We will generate a qrels file alongside the output file
    qrels_path = args.output_file.replace(".jsonl", "_qrels.tsv")
    
    import hashlib
    
    with open(args.input_file, 'r') as fin, open(args.output_file, 'w') as ftasks, open(qrels_path, 'w') as fqrels:
        fqrels.write("query-id\tcorpus-id\tscore\n")
        
        for idx, line in enumerate(fin):
            item = json.loads(line)
            query_text = item['query']
            positive_text = item['positive']
            
            # Generate a stable ID for the query
            query_hash = hashlib.md5(query_text.encode('utf-8')).hexdigest()
            # To be compatible with previous ID formats, we can use hash prefix
            task_id = f"{query_hash}::1" # Assume turn 1
            
            # Task Object for pipeline
            task_obj = {
                "task_id": task_id,
                "input": [{"speaker": "user", "text": query_text}],
                "turn": 1,
                "text": query_text # BEIR format support
            }
            ftasks.write(json.dumps(task_obj) + "\n")
            
            # For QRELS, we have a problem: We have the positive text, but not the DOC ID.
            # The benchmark requires retrieving by DOC ID.
            # IF the positive text exists in the corpus, we need to find its ID.
            # If we don't have the corpus ID mapping, we can't produce valid QRELS for retrieval metrics.
            
            # Wait, the user said "ESOS SON LOS SPLITS QUE DEBES USAR".
            # The file `test.jsonl` contains `positive` which is the text.
            # If the retrieval system returns DOC IDs, we can't score unless we know which DOC ID corresponds to that text.
            
            # However, maybe we are just supposed to RUN the retrieval on these queries?
            # But the user also complained about "results too low", implying evaluation.
            
            # Let's check if the corpus allows mapping text -> id.
            # But scanning the whole corpus for every query is slow.
            
            # ALTERNATIVE: Maybe the HF dataset is just for FINE-TUNING and checking loss?
            # User said: "con la data de splits que ahi mismo en ese repositorio esta?"
            # "ademas usa toda la potencia... para que sea rapido... ademas los resultados son demasiado bajos no?"
            
            # If I run inference on these queries, I will get a list of retrieved DocIDs.
            # To evaluate, I need ground truth. content of `test.jsonl` IS ground truth (query, positive).
            
            pass 

    print("Partial conversion script created - Need to resolve Doc ID mapping issue.")

if __name__ == "__main__":
    main()
