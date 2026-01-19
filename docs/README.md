# Documentation

## Quick Reference

**Main guides:**
- [../README.md](../README.md) - Main project documentation
- [../DOCKER_USAGE.md](../DOCKER_USAGE.md) - Docker setup and usage
- [../CHANGELOG.md](../CHANGELOG.md) - Recent changes

**Results and validation:**
- [VALIDACION_ESTADISTICA_COMPLETA.md](VALIDACION_ESTADISTICA_COMPLETA.md) - Statistical validation (777 queries, 96.14% accuracy)
- [RESUMEN_PARA_PROFESORA.md](RESUMEN_PARA_PROFESORA.md) - Executive summary
- [experiment_table.md](experiment_table.md) - All experiments overview

**Technical details:**
- [methodology.md](methodology.md) - Evaluation methodology
- [leakage_policy.md](leakage_policy.md) - Data leakage prevention

**Presentation materials:**
- [PRESENTATION_GUIDE.md](PRESENTATION_GUIDE.md) - Presentation guide
- [resumen_ejecutivo_presentacion.md](resumen_ejecutivo_presentacion.md) - Executive presentation
- [GRAFICOS_PRESENTACION.md](GRAFICOS_PRESENTACION.md) - Charts and visualizations

## Data Availability

**Training data:** Included in repository
- `data/passage_level_processed/` - Corpus documents
- `data/retrieval_tasks/` - Queries and ground truth

**Baseline results:** [HuggingFace Dataset](https://huggingface.co/datasets/vania-janet/MTRAG_taskA_results)
- 679 files, 10.8 GB
- Complete experiment results
- Statistical analysis reports

## Key Results

**NDCG@10 Performance:**

| Method | ClapNQ | Cloud | FiQA | Govt | Avg |
|--------|--------|-------|------|------|-----|
| BM25 | 0.378 | 0.459 | 0.328 | 0.482 | 0.412 |
| BGE-1.5 | 0.461 | 0.521 | 0.398 | 0.556 | 0.484 |
| BGE-M3 | 0.489 | 0.548 | 0.421 | 0.579 | 0.509 |

**Statistical validation:** 96.14% accuracy on 777 queries across 4 domains.
