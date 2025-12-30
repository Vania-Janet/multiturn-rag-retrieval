# 🚀 Guía Completa: Servidor A100 con 2 GPUs + Docker + TMUX

## PASO 1: Transferir Código (Desde tu Mac)

```bash
# En tu terminal local (NO en el servidor)
cd /Users/vania/Downloads/rag-ss3/mt-rag-benchmark/task_a_retrieval

# Reemplaza con tus datos: usuario@IP_SERVIDOR
rsync -avz --progress \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude 'indices/' \
  --exclude '.git/objects' \
  ./ usuario@IP_SERVIDOR:~/mt-rag-benchmark/

# Pushea tu código actualizado a git primero
git add -A
git commit -m "Final config for A100 run"
git push
```

---

## PASO 2: Conectar y Setup Inicial

```bash
# Conéctate al servidor
ssh usuario@IP_SERVIDOR

# Instala dependencias del sistema
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip git tmux htop docker.io
sudo usermod -aG docker $USER  # Permite usar Docker sin sudo
newgrp docker  # Aplica el cambio sin desconectarte

# Navega al proyecto
cd ~/mt-rag-benchmark
```

---

## PASO 3: Setup Docker + Elasticsearch + ELSER

```bash
# Crea una sesión tmux para Elasticsearch (para que no se caiga)
tmux new -s elasticsearch

# Dentro de tmux, lanza Elasticsearch con ELSER habilitado
docker run -d --name elasticsearch \
  --restart unless-stopped \
  -p 9200:9200 \
  -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "xpack.ml.enabled=true" \
  -e "ES_JAVA_OPTS=-Xms4g -Xmx4g" \
  docker.elastic.co/elasticsearch/elasticsearch:8.11.0

# Espera 30 segundos a que arranque
echo "Esperando a Elasticsearch..."
sleep 30

# Verifica que esté corriendo
curl -X GET "localhost:9200/_cluster/health?pretty"

# Sal de tmux (Ctrl+B, luego D)
# Presiona: Ctrl + B
# Luego presiona: D
```

---

## PASO 4: Setup Python Environment

```bash
# Crea y activa entorno virtual
python3.11 -m venv .venv
source .venv/bin/activate

# Instala PyTorch con CUDA
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Instala dependencias del proyecto
pip install -r requirements.txt

# Instala FAISS GPU (crítico para A100)
pip uninstall -y faiss-cpu
pip install faiss-gpu

# Descarga datos de NLTK (necesario para BM25)
python -c "import nltk; nltk.download('punkt')"

# Verifica GPU
python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}')"
```

---

## PASO 5: Construcción de Índices (Paralelo en 2 GPUs)

```bash
# Crea sesión tmux para indexing
tmux new -s indexing

# Configura variables de entorno para ELSER
export ELASTICSEARCH_URL="http://localhost:9200"
export ELASTICSEARCH_USER=""
export ELASTICSEARCH_PASSWORD=""
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Lanza indexing en GPU 0 (clapnq + cloud)
CUDA_VISIBLE_DEVICES=0 python src/pipeline/indexing/build_indices.py \
    --models bge bge-m3 bm25 elser \
    --domains clapnq cloud \
    > logs_indexing_gpu0.log 2>&1 &

# Lanza indexing en GPU 1 (fiqa + govt)
CUDA_VISIBLE_DEVICES=1 python src/pipeline/indexing/build_indices.py \
    --models bge bge-m3 bm25 elser \
    --domains fiqa govt \
    > logs_indexing_gpu1.log 2>&1 &

echo "✅ Indexing corriendo en paralelo. Monitorea con:"
echo "   tail -f logs_indexing_gpu0.log"
echo "   tail -f logs_indexing_gpu1.log"
echo "   watch -n 1 nvidia-smi"

# Sal de tmux: Ctrl+B, luego D
```

**⏳ Espera a que termine (puede tardar 30-60 minutos). Monitorea con:**
```bash
# Reattach a la sesión
tmux attach -t indexing

# O revisa logs desde fuera
tail -f logs_indexing_gpu0.log
```

---

## PASO 6: Ejecutar Experimentos (Paralelo)

Una vez que terminen los índices, corre los experimentos.

```bash
# Crea sesión para experimentos
tmux new -s experiments

# Activa el entorno
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
export ELASTICSEARCH_URL="http://localhost:9200"

# --- GPU 0: Replicaciones ---
# BM25 (CPU, no usa GPU pero lo lanzamos aquí para organizar)
CUDA_VISIBLE_DEVICES=0 python scripts/run_experiment.py \
    --experiment replication_bm25 --domain all \
    > logs_replication_bm25.log 2>&1 &

# BGE 1.5 Replication
CUDA_VISIBLE_DEVICES=0 python scripts/run_experiment.py \
    --experiment replication_bge15 --domain all \
    > logs_replication_bge15.log 2>&1 &

# --- GPU 1: Baselines Avanzados ---
# BGE M3 Full History
CUDA_VISIBLE_DEVICES=1 python scripts/run_experiment.py \
    --experiment A1_baseline_bgem3_fullhist --domain all \
    > logs_baseline_bgem3.log 2>&1 &

# BM25 Full History (CPU)
CUDA_VISIBLE_DEVICES=1 python scripts/run_experiment.py \
    --experiment A0_baseline_bm25_fullhist --domain all \
    > logs_baseline_bm25.log 2>&1 &

# ELSER (ambos, ya que usa Elasticsearch)
python scripts/run_experiment.py \
    --experiment replication_elser --domain all \
    > logs_replication_elser.log 2>&1 &

python scripts/run_experiment.py \
    --experiment A0_baseline_elser_fullhist --domain all \
    > logs_baseline_elser.log 2>&1 &

echo "✅ Experimentos corriendo. Logs disponibles en logs_*.log"

# Sal de tmux: Ctrl+B, luego D
```

---

## PASO 7: Monitorear Progreso

```bash
# Ver sesiones activas
tmux ls

# Reconectar a sesiones
tmux attach -t experiments  # Para ver experimentos
tmux attach -t indexing     # Para ver indexing

# Ver logs en tiempo real
tail -f logs_baseline_bgem3.log

# Ver uso de GPU
watch -n 1 nvidia-smi

# Ver procesos Python corriendo
ps aux | grep python
```

---

## 🎯 Resumen de Sesiones TMUX

| Sesión | Propósito | Comando para Reconectar |
|--------|-----------|-------------------------|
| `elasticsearch` | Docker Elasticsearch | `tmux attach -t elasticsearch` |
| `indexing` | Construcción de índices | `tmux attach -t indexing` |
| `experiments` | Ejecución de experimentos | `tmux attach -t experiments` |

---

## 📊 Resultados

Al terminar, encontrarás los resultados en:
```
experiments/
├── replication_bm25/
│   ├── clapnq/
│   │   ├── retrieval_results.jsonl
│   │   ├── metrics.json
│   │   └── analysis_report.json
│   ├── cloud/
│   ├── fiqa/
│   └── govt/
├── replication_bge15/
├── A1_baseline_bgem3_fullhist/
└── ...
```

**Tiempo estimado total: 2-4 horas** (dependiendo del tamaño del corpus y velocidad de las A100).

---

## 📋 Checklist de Configuración

- [ ] Código transferido al servidor
- [ ] Docker instalado y Elasticsearch corriendo
- [ ] Python 3.11 + venv configurado
- [ ] PyTorch con CUDA instalado
- [ ] FAISS GPU instalado
- [ ] Índices construidos (bge, bge-m3, bm25, elser)
- [ ] Experimentos lanzados
- [ ] Logs monitoreados

---

## 🐛 Troubleshooting

### Elasticsearch no arranca
```bash
docker logs elasticsearch
docker restart elasticsearch
```

### GPU no detectada
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### Proceso bloqueado
```bash
# Ver procesos
ps aux | grep python

# Matar proceso específico
kill -9 <PID>
```

### Sesión tmux perdida
```bash
# Listar sesiones
tmux ls

# Matar sesión
tmux kill-session -t <nombre>
```
