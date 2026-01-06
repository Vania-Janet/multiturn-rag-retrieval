import sys
import os
sys.path.append("/workspace/mt-rag-benchmark/task_a_retrieval")

print("Starting import...")
try:
    from src.pipeline import run
    print("Import successful")
except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()
