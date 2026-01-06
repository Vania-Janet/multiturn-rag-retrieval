# Baseline Verification Report

All baseline experiments in `experiments/0-baselines` have been verified.

## Status Summary

| Experiment | Status | Domains | Checks |
|------------|--------|---------|--------|
| A0_baseline_bm25_fullhist | ✅ Complete | 4/4 | Pass |
| A0_baseline_splade_fullhist | ✅ Complete | 4/4 | Pass |
| A1_baseline_bgem3_fullhist | ✅ Complete | 4/4 | Pass |
| A1_baseline_voyage_fullhist | ✅ Complete | 4/4 | Pass |
| replication_bge15 | ✅ Complete | 4/4 | Pass |
| replication_bgem3 | ✅ Complete | 4/4 | Pass |
| replication_bm25 | ✅ Complete | 4/4 | Pass |
| replication_splade | ✅ Complete | 4/4 | Pass |
| replication_voyage | ✅ Complete | 4/4 | Pass |

## Verification Details

- **Existence**: All `metrics.json` files exist.
- **Validity**: All nDCG@10 scores are valid (> 0.0, < 1.0).
- **Configuration**: `query_mode` matches the experiment type (Full History vs Last Turn).

## Selected Metrics (nDCG@10)

| Domain | BM25 (Rep) | SPLADE (Rep) | BGE-M3 (Rep) | Voyage (Rep) |
|--------|------------|--------------|--------------|--------------|
| clapnq | 0.2353 | 0.5082 | 0.4768 | 0.5163 |
| cloud | 0.2201 | 0.4512 | 0.3938 | 0.3621 |
| fiqa | 0.1026 | 0.3715 | 0.3613 | 0.2912 |
| govt | 0.2767 | 0.4515 | 0.4197 | 0.4144 |

All look consistent with expectations (SPLADE/Voyage performing well, BM25 lower).
