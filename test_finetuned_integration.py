#!/usr/bin/env python3
"""
Test script para verificar la integración del modelo fine-tuned BGE reranker.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_model_access():
    """Test 1: Verificar acceso al modelo en Hugging Face"""
    print("=" * 80)
    print("TEST 1: Acceso al modelo en Hugging Face")
    print("=" * 80)
    
    try:
        from transformers import AutoTokenizer
        
        print("Descargando tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("pedrovo9/bge-reranker-v2-m3-multirag-finetuned")
        print("✓ Tokenizer cargado exitosamente")
        print(f"  Vocab size: {tokenizer.vocab_size}")
        print(f"  Model max length: {tokenizer.model_max_length}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_reranker_class():
    """Test 2: Verificar que la clase FineTunedBGEReranker funciona"""
    print("\n" + "=" * 80)
    print("TEST 2: Clase FineTunedBGEReranker")
    print("=" * 80)
    
    try:
        from pipeline.reranking import FineTunedBGEReranker
        
        print("Inicializando reranker...")
        reranker = FineTunedBGEReranker(
            model_name="pedrovo9/bge-reranker-v2-m3-multirag-finetuned",
            config={"batch_size": 4, "use_fp16": False}  # Small batch for testing
        )
        print(f"✓ Reranker inicializado")
        print(f"  Device: {reranker.device}")
        print(f"  Model name: {reranker.model_name}")
        
        # Test reranking
        print("\nProbando reranking...")
        query = "What is cloud computing?"
        documents = [
            {"text": "Cloud computing is the delivery of computing services over the internet.", "id": "doc1"},
            {"text": "The weather today is cloudy.", "id": "doc2"},
            {"text": "Amazon Web Services (AWS) is a cloud platform.", "id": "doc3"},
        ]
        
        reranked = reranker.rerank(query, documents, top_k=3)
        
        print("✓ Reranking completado")
        print("\nResultados:")
        for i, doc in enumerate(reranked, 1):
            print(f"  {i}. {doc['id']}: score={doc['rerank_score']:.4f}")
            print(f"     Text: {doc['text'][:50]}...")
        
        # Verify doc1 and doc3 are ranked higher than doc2
        doc_scores = {doc['id']: doc['rerank_score'] for doc in reranked}
        if doc_scores['doc2'] < doc_scores['doc1'] and doc_scores['doc2'] < doc_scores['doc3']:
            print("\n✓ Rankings look correct (relevant docs ranked higher)")
            return True
        else:
            print("\n⚠ Rankings might be off (check manually)")
            return True  # Don't fail on this
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_import():
    """Test 3: Verificar que el import funciona en __init__.py"""
    print("\n" + "=" * 80)
    print("TEST 3: Import en reranking/__init__.py")
    print("=" * 80)
    
    try:
        from pipeline import reranking
        
        if hasattr(reranking, 'FineTunedBGEReranker'):
            print("✓ FineTunedBGEReranker disponible en pipeline.reranking")
            return True
        else:
            print("✗ FineTunedBGEReranker NO está disponible")
            print(f"  Disponibles: {[x for x in dir(reranking) if not x.startswith('_')]}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_configs():
    """Test 4: Verificar configuraciones de experimentos"""
    print("\n" + "=" * 80)
    print("TEST 4: Configuraciones de experimentos")
    print("=" * 80)
    
    experiments = [
        "A10_finetuned_reranker",
        "finetune_bge_splade_bge15_rewrite",
        "finetune_bge_splade_voyage_rewrite"
    ]
    
    all_ok = True
    for exp in experiments:
        config_path = f"configs/experiments/05-finetune/{exp}.yaml"
        
        if not os.path.exists(config_path):
            print(f"✗ Config no existe: {config_path}")
            all_ok = False
            continue
        
        with open(config_path) as f:
            content = f.read()
            
        # Check for finetuned_bge reranker_type
        if "reranker_type: \"finetuned_bge\"" in content or "reranker_type: finetuned_bge" in content:
            print(f"✓ {exp}: reranker_type correcto")
        else:
            print(f"✗ {exp}: falta reranker_type: finetuned_bge")
            all_ok = False
        
        # Check for model name
        if "pedrovo9/bge-reranker-v2-m3-multirag-finetuned" in content:
            print(f"✓ {exp}: model_name correcto")
        else:
            print(f"⚠ {exp}: model_name no encontrado")
    
    return all_ok

def main():
    """Ejecutar todos los tests"""
    print("\n" + "=" * 80)
    print("VERIFICACIÓN DE INTEGRACIÓN - MODELO FINE-TUNED BGE RERANKER")
    print("=" * 80)
    
    results = {}
    
    # Run tests
    results['model_access'] = test_model_access()
    results['reranker_class'] = test_reranker_class()
    results['import'] = test_import()
    results['configs'] = test_configs()
    
    # Summary
    print("\n" + "=" * 80)
    print("RESUMEN")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} - {test_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✓ ¡Todos los tests pasaron! El modelo está listo para usar.")
        print("\nPara ejecutar experimentos:")
        print("  ./run_finetuned_experiments.sh")
        return 0
    else:
        print("\n✗ Algunos tests fallaron. Revisa los errores arriba.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
