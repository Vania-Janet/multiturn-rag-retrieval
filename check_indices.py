import os
from pathlib import Path

print("=" * 80)
print("VERIFICACIÓN DE ÍNDICES PARA TODOS LOS MODELOS")
print("=" * 80)

indices_to_check = {
    "BM25": {
        "clapnq": "indices/clapnq/bm25",
        "cloud": "indices/cloud/bm25",
        "fiqa": "indices/fiqa/bm25",
        "govt": "indices/govt/bm25",
    },
    "SPLADE": {
        "clapnq": "indices/clapnq/splade",
        "cloud": "indices/cloud/splade",
        "fiqa": "indices/fiqa/splade",
        "govt": "indices/govt/splade",
    },
    "BGE-M3": {
        "clapnq": "indices/clapnq/bge-m3",
        "cloud": "indices/cloud/bge-m3",
        "fiqa": "indices/fiqa/bge-m3",
        "govt": "indices/govt/bge-m3",
    },
    "Voyage": {
        "clapnq": "indices/clapnq/voyage",
        "cloud": "indices/cloud/voyage",
        "fiqa": "indices/fiqa/voyage",
        "govt": "indices/govt/voyage",
    },
}

for model_name, domains in indices_to_check.items():
    print(f"\n{model_name}:")
    all_exist = True
    for domain, path in domains.items():
        exists = os.path.exists(path) and os.path.isdir(path)
        if exists:
            # Count files
            files = list(Path(path).glob("*"))
            print(f"  ✓ {domain:8} - {path} ({len(files)} archivos)")
        else:
            print(f"  ✗ {domain:8} - {path} NO EXISTE")
            all_exist = False
    
    if all_exist:
        print(f"  → {model_name}: TODOS LOS ÍNDICES EXISTEN")
    else:
        print(f"  → {model_name}: FALTAN ÍNDICES")

print("\n" + "=" * 80)
print("VERIFICACIÓN COMPLETADA")
print("=" * 80)
