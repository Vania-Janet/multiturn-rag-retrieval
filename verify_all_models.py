from pathlib import Path
import sys
sys.path.insert(0, 'src')

from utils.config_loader import merge_configs

def substitute_domain(obj, domain_name):
    if isinstance(obj, dict):
        return {k: substitute_domain(v, domain_name) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [substitute_domain(item, domain_name) for item in obj]
    elif isinstance(obj, str):
        return obj.replace("{domain}", domain_name)
    else:
        return obj

experiments = {
    "bm25_r1_condensation": "sparse",
    "bm25_r2_multi": "sparse",
    "splade_r1_condensation": "sparse",
    "bgem3_r1_condensation": "dense",
    "bgem3_r2_multi": "dense",
    "voyage_r1_condensation": "dense",
    "voyage_r2_multi": "dense",
}

print("=" * 80)
print("VERIFICACIÓN DE TODOS LOS MODELOS - Experimentos 01-Query")
print("=" * 80)

for exp_name, expected_type in experiments.items():
    print(f"\n{'='*80}")
    print(f"Experimento: {exp_name}")
    print(f"{'='*80}")
    
    try:
        config = merge_configs(
            base_config="configs/base.yaml",
            domain_config="configs/domains/clapnq.yaml",
            experiment_config=f"configs/experiments/01-query/{exp_name}.yaml"
        )
        config = substitute_domain(config, "clapnq")
        
        # Verificar configuración
        retrieval_type = config['retrieval']['type']
        rewriter_type = config['query_transform']['rewriter_type']
        max_rewrites = config['query_transform']['rewriter_config']['max_rewrites']
        query_file = config['data']['query_file']
        
        # Para vLLM, verificar retrieval_type
        if rewriter_type == 'vllm':
            param_retrieval_type = config['query_transform']['rewriter_config'].get('retrieval_type', 'NOT SET')
            
            print(f"✓ Retrieval type: {retrieval_type}")
            print(f"✓ Rewriter: {rewriter_type}")
            print(f"✓ Max rewrites: {max_rewrites}")
            print(f"✓ Query file: {query_file}")
            print(f"✓ Retrieval_type param: {param_retrieval_type}")
            
            # Verificar que el parámetro coincide con el tipo esperado
            if expected_type in param_retrieval_type or (expected_type == "dense" and param_retrieval_type == "dense"):
                print(f"✓ Retrieval_type correcto para {retrieval_type}")
            else:
                print(f"⚠️  ADVERTENCIA: Se esperaba '{expected_type}' pero tiene '{param_retrieval_type}'")
                
        else:
            print(f"✓ Retrieval type: {retrieval_type}")
            print(f"✓ Rewriter: {rewriter_type} (no requiere retrieval_type)")
            print(f"✓ Max rewrites: {max_rewrites}")
            print(f"✓ Query file: {query_file}")
        
        # Verificar que el archivo existe
        import os
        if os.path.exists(query_file):
            with open(query_file) as f:
                first_line = f.readline()
                import json
                data = json.loads(first_line)
                if 'input' in data:
                    print(f"✓ Formato correcto: tiene campo 'input' estructurado")
                else:
                    print(f"⚠️  ADVERTENCIA: No tiene campo 'input', tiene: {list(data.keys())}")
        else:
            print(f"✗ Query file NO EXISTE: {query_file}")
            
        # Verificar modelo específico
        if 'model_name' in config['retrieval']:
            model = config['retrieval']['model_name']
            print(f"✓ Modelo: {model}")
        elif 'method' in config['retrieval']:
            method = config['retrieval']['method']
            print(f"✓ Método: {method}")
            
    except Exception as e:
        print(f"✗ ERROR al cargar config: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
print("VERIFICACIÓN COMPLETADA")
print("=" * 80)
