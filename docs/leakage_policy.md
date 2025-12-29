# Data Leakage Prevention Policy

## Critical Principle

**Zero tolerance for data leakage between train/validation/test splits.**

Any form of information flow from test to train invalidates all results.

## Leakage Sources & Prevention

### 1. Conversation-Level Splits (ClapNQ)

#### Risk
Splitting individual turns from the same conversation creates leakage: later turns contain information from earlier turns.

#### Prevention
✅ **Conversation-level splitting**: All turns from a conversation stay in the same split.

```yaml
# splits/clapnq.yaml
train_conversations: [0-799]  # All turns from these conversations
val_conversations: [800-899]
test_conversations: [900-999]
```

❌ **Never**: Split turns independently

### 2. Hyperparameter Tuning

#### Risk
Tuning on test set leads to overfitting on the evaluation set.

#### Prevention
✅ **Two-stage process**:
1. Tune on validation set
2. Final evaluation on test set (once)

✅ **Document all tuning**:
- Log which hyperparameters were explored
- Report validation performance during tuning
- Report test performance only for final model

❌ **Never**: 
- Look at test metrics during development
- Tune based on test performance
- Run multiple "final" evaluations

### 3. Domain-Specific Parameters

#### Risk
Setting domain parameters by observing test performance.

#### Prevention
✅ **Fixed a priori rules**:
- Define domain parameters before seeing test data
- Base on domain knowledge, not empirical test tuning
- Document rationale for each parameter choice

```python
# configs/domains/clapnq.yaml
# RATIONALE: Code queries need exact matching (favor sparse)
retrieval:
  sparse_weight: 0.6  # Favor BM25/SPLADE for code
  dense_weight: 0.4
```

✅ **Permitted**: Use validation set to select weights

❌ **Never**: Iterate on test set to find "best" parameters

### 4. Query-Document Overlap

#### Risk
Training documents appearing in test queries' ground truth.

#### Prevention
✅ **Disjoint qrels**: Ensure no query-document pairs overlap between splits

✅ **Validation**: Run overlap detection script
```bash
python scripts/validate_splits.py --check-overlap
```

❌ **Never**: Use same qrels for train and test

### 5. Fine-tuning Data

#### Risk
Fine-tuning on examples from test set.

#### Prevention
✅ **Training data generation**: Use only train split queries

```python
# scripts/training/prepare_training_data.py
train_queries = load_split('clapnq', 'train')  # Only train
val_queries = load_split('clapnq', 'val')      # For validation
# NEVER load test queries for training data
```

✅ **Document fine-tuning data**: Save train data used

❌ **Never**: Include test queries in training data

### 6. Model Selection

#### Risk
Selecting models based on test performance.

#### Prevention
✅ **Validation-based selection**:
```python
# Select best checkpoint on validation
best_epoch = select_best(val_metrics)
# Evaluate best checkpoint on test (once)
test_metrics = evaluate(best_epoch, test_data)
```

✅ **Pre-register model choices**: Document model selection before test evaluation

❌ **Never**: Try multiple models and pick best on test

### 7. Error Analysis

#### Risk
Analyzing test errors and adjusting system.

#### Prevention
✅ **Error analysis workflow**:
1. Analyze validation errors → adjust system
2. Re-evaluate on validation
3. Final test evaluation (no adjustments after)

✅ **Post-hoc analysis**: Analyze test errors for paper discussion only (no system changes)

❌ **Never**: Debug on test set and re-run

### 8. Preprocessing & Feature Engineering

#### Risk
Designing features or preprocessing based on test data patterns.

#### Prevention
✅ **Fixed preprocessing**: Define all preprocessing before test evaluation

✅ **Validation-driven**: Use validation set to design features

```python
# Good: Decide on validation
if val_performance_improves:
    use_preprocessing = True
# Bad: Decide on test
if test_performance_improves:  # ❌ LEAKAGE
    use_preprocessing = True
```

### 9. Index Building

#### Risk
Building indices differently based on test queries.

#### Prevention
✅ **Uniform indexing**: Same indexing for all documents (train/val/test)

✅ **Test-agnostic**: Build indices without looking at test queries

### 10. Ensemble/Fusion Weights

#### Risk
Tuning fusion weights (sparse/dense, RRF k) on test.

#### Prevention
✅ **Validation tuning**:
```yaml
# Tune on validation set
experiments:
  - sparse_weight: [0.3, 0.5, 0.7]
  - dense_weight: [0.7, 0.5, 0.3]
  
# Select best on validation
best_config = grid_search(val_data)

# Apply to test (once)
test_results = evaluate(best_config, test_data)
```

## Verification Checklist

Before submitting results, verify:

- [ ] Test queries never used during development
- [ ] Hyperparameters tuned only on validation
- [ ] Domain parameters defined before test evaluation
- [ ] No query-document overlap between splits
- [ ] Training data contains only train split
- [ ] Model selection based on validation only
- [ ] No system changes after seeing test results
- [ ] Preprocessing fixed before test evaluation
- [ ] Test evaluation run exactly once per experiment
- [ ] Documented rationale for all design choices

## Audit Trail

All experiments log:
- Split definitions used
- Hyperparameters and their source (validation tuning)
- Model checkpoint selected (based on validation)
- Test evaluation timestamp (should be once)

Location: `experiments/{experiment}/logs/audit.json`

## Reviewer Transparency

For paper submission, we provide:
- This policy document
- Split definitions (`splits/`)
- Experiment logs with timestamps
- Pre-registered experiment plan (if applicable)

## Contact

For questions about our leakage prevention policy, see `docs/methodology.md` or contact the authors.

## Last Updated

2025-01-15
