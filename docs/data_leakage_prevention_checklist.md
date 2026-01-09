# Data Leakage Prevention Checklist - Dual-Dataset Study

## Overview
Data leakage occurs when information from outside the training dataset is used to create the model, leading to overly optimistic performance estimates that don't generalize to real-world scenarios. This checklist ensures proper experimental methodology for the **dual-dataset study** with card_transdata.csv and creditcard.csv.

---

## ✅ Dataset Isolation (Critical for Dual-Dataset Studies)

### Cross-Dataset Contamination Prevention
- [ ] **Maintain separate directories** - Each dataset has its own results/ subdirectory
- [ ] **Independent splits** - card_transdata and creditcard have separate train/val/test splits
- [ ] **Separate scalers** - Each dataset has its own fitted StandardScaler
- [ ] **No dataset merging** - Never combine or pool data from both datasets
- [ ] **No cross-dataset evaluation** - Models trained on one dataset are never evaluated on the other
- [ ] **Independent experiment logs** - Each dataset maintains its own experiment tracking
- [ ] **Comparison happens post-hoc** - Aggregate results only in final analysis (Notebook 07)

---

## ✅ Split Creation

### Before Creating Splits
- [ ] **Understand the target distribution** - Check class balance before splitting
- [ ] **Decide on split ratios** - 70/15/15 for train/val/test is used in this project
- [ ] **Set random seed** - Use `RANDOM_SEED = 42` for reproducibility
- [ ] **Verify dataset identity** - Confirm which dataset (card_transdata or creditcard) is being processed

### During Split Creation
- [ ] **Stratified split** - Preserve class distribution across train/val/test sets
- [ ] **Test set isolation** - Physically separate test set immediately after creation
- [ ] **Save split indices** - Store train/val/test indices as `.npy` files in dataset-specific directory
- [ ] **Verify distributions** - Confirm class balance is preserved in all splits
- [ ] **Document split sizes** - Record number of samples in each split
- [ ] **Check index uniqueness** - Ensure no overlap between train/val/test indices

### After Split Creation
- [ ] **Never modify splits** - Use the same indices across all experiments for that dataset
- [ ] **Test set quarantine** - Do not look at test set until final evaluation (ONE-TIME)
- [ ] **Validation set purpose** - Use only for hyperparameter tuning and model selection
- [ ] **Load indices from saved files** - Never re-split in subsequent notebooks

---

## ✅ Feature Preprocessing

### StandardScaler (Feature Normalization)
- [ ] **Fit on training data only** - `scaler.fit(X_train)` for each dataset independently
- [ ] **Transform all splits separately** - `scaler.transform(X_train/val/test)`
- [ ] **Save fitted scaler** - Store scaler as `.pkl` file in dataset-specific directory
- [ ] **Reuse same scaler** - Apply same scaler to all experiments for that dataset
- [ ] **Never fit on test data** - Test data must be "unseen" to preprocessing
- [ ] **No scaler sharing** - card_transdata and creditcard use different fitted scalers
- [ ] **Load scaler from saved file** - Never re-fit in subsequent notebooks

### Feature Engineering (if applicable)
- [ ] **Calculate statistics on training only** - Mean, std, min, max from training set
- [ ] **Apply transformations consistently** - Same transformations to val/test
- [ ] **No target leakage** - Don't use target variable for feature creation
- [ ] **Time-based features** - For time-series: don't use future information

---

## ✅ Neural Network Training

### Hyperparameter Selection
- [ ] **Use validation set only** - All hyperparameter tuning on validation data
- [ ] **Never use test metrics** - Test set metrics do not influence decisions
- [ ] **Document all trials** - Log every experiment configuration
- [ ] **Avoid peeking** - Don't informally "check" test performance during development

### Early Stopping
- [ ] **Monitor validation metrics** - Early stopping uses `val_loss` only
- [ ] **Set patience appropriately** - `patience=15` in this project
- [ ] **Save best validation model** - Checkpoint based on validation performance
- [ ] **Test set remains untouched** - Early stopping never sees test data

### Class Weight Calculation
- [ ] **Compute from training set** - Class weights based on training distribution
- [ ] **Don't include val/test** - Validation/test distributions are irrelevant
- [ ] **Document class ratios** - Record fraud vs legitimate ratios from training

### Model Checkpointing
- [ ] **Save based on validation** - Best model = best validation metric
- [ ] **Version control models** - Name models with timestamp/experiment ID
- [ ] **Track model provenance** - Record which data/config produced each model

---

## ✅ Threshold Optimization

### Validation-Based Optimization
- [ ] **Use validation set only** - Find optimal threshold on validation data
- [ ] **Consider business costs** - Weight false positives vs false negatives
- [ ] **Generate PR curves** - Precision-Recall curve from validation predictions
- [ ] **Document threshold choice** - Record optimal threshold and rationale
- [ ] **Never use test set** - Test threshold must not influence optimization

### Application to Test Set
- [ ] **Apply fixed threshold** - Use threshold determined from validation
- [ ] **No iterative adjustments** - Don't modify threshold based on test results
- [ ] **Report final performance** - Test metrics reported without changes

---

## ✅ Final Evaluation

### One-Time Test Set Usage
- [ ] **Select best model first** - Choose model based on validation metrics
- [ ] **Run test evaluation once** - Single evaluation on test set
- [ ] **No cherry-picking** - Report results regardless of outcome
- [ ] **No model adjustments** - Don't retrain after seeing test results
- [ ] **Complete transparency** - Report all test metrics honestly

### Reporting Requirements
- [ ] **Validation vs test comparison** - Show both sets of metrics
- [ ] **Confidence intervals** (optional) - Statistical uncertainty estimates
- [ ] **Failure analysis** - Analyze errors without retraining
- [ ] **Deployment recommendations** - Based on test performance only

---

## ✅ Cross-Experiment Consistency

### Data Reuse
- [ ] **Same split indices** - All experiments use identical train/val/test splits
- [ ] **Same preprocessing** - Reuse fitted scaler across all experiments
- [ ] **Same random seed** - Consistent `RANDOM_SEED = 42` throughout
- [ ] **Document any deviations** - Explicitly note if splits/preprocessing change

### Experiment Logging
- [ ] **Log all hyperparameters** - Record complete configuration for each run
- [ ] **Track data versions** - Note if preprocessing changes between experiments
- [ ] **Version control** - Git commit before major experiment runs
- [ ] **Reproducibility manifest** - Document exact steps to reproduce results

---

## 🚨 Common Leakage Pitfalls to Avoid

### ❌ Don't Do This:
1. **Normalizing before splitting** - Fit scaler on full dataset
2. **Peeking at test set** - Informally checking test performance during development
3. **Iterative test evaluation** - Running test evaluation multiple times
4. **Test-based model selection** - Choosing model because it performed well on test
5. **Threshold tuning on test** - Optimizing decision threshold using test data
6. **Class weights from full data** - Computing class weights from train+val+test
7. **Feature selection on full data** - Selecting features using test set information
8. **Imputation leakage** - Filling missing values using statistics from test set

### ✅ Do This Instead:
1. **Split first, then normalize** - Fit scaler only on training data
2. **Strict test isolation** - Never look at test until final evaluation
3. **Single test evaluation** - One-time test set usage only
4. **Validation-based selection** - All decisions from validation metrics
5. **Validation threshold tuning** - Optimize threshold on validation set
6. **Training-only statistics** - All statistics computed from training data only
7. **Training-based feature selection** - Select features using training set
8. **Training-based imputation** - Imputation strategy from training data

---

## 📋 Verification Protocol (Dual-Dataset)

Before running final test evaluation on either dataset, verify:

```python
# Verification checklist code for dual-dataset study
import numpy as np
import os

def verify_dataset_isolation(dataset_name):
    """Verify no leakage for a specific dataset"""
    print(f"\n{'='*60}")
    print(f"Verifying {dataset_name} dataset isolation...")
    print(f"{'='*60}")
    
    base_path = f'results/{dataset_name}'
    
    # 1. Verify split indices are saved and consistent
    train_idx = np.load(f'{base_path}/train_indices.npy')
    val_idx = np.load(f'{base_path}/val_indices.npy')
    test_idx = np.load(f'{base_path}/test_indices.npy')
    
    print(f"✓ Split sizes: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test")
    
    # 2. Verify no overlap between splits
    train_set = set(train_idx)
    val_set = set(val_idx)
    test_set = set(test_idx)
    
    assert len(train_set.intersection(val_set)) == 0, "Train/val overlap!"
    assert len(train_set.intersection(test_set)) == 0, "Train/test overlap!"
    assert len(val_set.intersection(test_set)) == 0, "Val/test overlap!"
    print(f"✓ No index overlap between splits")
    
    # 3. Verify scaler exists
    assert os.path.exists(f'{base_path}/fitted_scaler.pkl'), "Scaler not saved!"
    print(f"✓ Fitted scaler saved")
    
    # 4. Verify experiment logs exist
    assert os.path.exists(f'{base_path}/logs/'), "Experiment logs directory missing!"
    print(f"✓ Experiment logs directory exists")
    
    print(f"✅ All leakage prevention checks passed for {dataset_name}!")

# Verify both datasets
verify_dataset_isolation('card_transdata')
verify_dataset_isolation('creditcard')

# 5. Verify no cross-dataset contamination
print(f"\n{'='*60}")
print("Verifying cross-dataset isolation...")
print(f"{'='*60}")

# Ensure split indices are completely different (different source data)
ct_train = set(np.load('results/card_transdata/train_indices.npy'))
cc_train = set(np.load('results/creditcard/train_indices.npy'))

# These should be different sizes (different datasets)
assert len(ct_train) != len(cc_train), "Datasets appear to be same size (suspicious!)"
print(f"✓ Datasets are independently split (different sizes)")
print(f"✅ Cross-dataset isolation verified!")
```

---

## 🔬 Dual-Dataset Specific Checklist

### Architecture Transfer (card_transdata → creditcard)
- [ ] **Select architecture on card_transdata only** - Use validation PR-AUC
- [ ] **Transfer architecture config (not weights)** - Apply layer structure, not trained parameters
- [ ] **Train from scratch on creditcard** - No weight transfer or fine-tuning
- [ ] **Independent regularization tuning** - Optimize regularization separately on creditcard
- [ ] **Document transfer rationale** - Explain why selected architecture should generalize

### Cross-Dataset Comparison Rules
- [ ] **Post-hoc comparison only** - Aggregate results after all experiments complete
- [ ] **Same metrics across datasets** - Use identical evaluation metrics for fair comparison
- [ ] **Report both validation AND test** - Show validation-based decisions AND test outcomes
- [ ] **Discuss generalization** - Analyze which insights transfer vs dataset-specific findings
- [ ] **Acknowledge baseline differences** - Explain if Random Forest dominates one dataset but not the other

### Experiment IDs and Logging
- [ ] **Consistent experiment IDs** - Use same IDs for equivalent configs across datasets
  - Example: ARCH-03 = [128, 64, 32] on BOTH card_transdata and creditcard
- [ ] **Dataset column in logs** - Clearly mark which dataset each experiment used
- [ ] **Separate log files** - Each dataset maintains its own nn_experiments.csv
- [ ] **Cross-reference experiments** - Enable matching by experiment_id across datasets

---

## 📊 Reporting Standards for Dual-Dataset Study

### Required Reporting Elements

1. **Dataset Characteristics Table**
   ```
   | Dataset | Samples | Features | Fraud % | Feature Type |
   |---------|---------|----------|---------|--------------|
   | card_transdata | 1M | 7 | 0.8% | Interpretable |
   | creditcard | 284K | 30 | 0.17% | PCA-transformed |
   ```

2. **Baseline Performance Table** (per dataset)
   ```
   | Dataset | Model | PR-AUC | Recall | F1 |
   |---------|-------|--------|--------|-----|
   | card_transdata | LR | X.XX | X.XX | X.XX |
   | card_transdata | RF | X.XX | X.XX | X.XX |
   | creditcard | LR | X.XX | X.XX | X.XX |
   | creditcard | RF | X.XX | X.XX | X.XX |
   ```

3. **Architecture Comparison Table** (validation performance)
   ```
   | Experiment ID | Architecture | card_transdata PR-AUC | creditcard PR-AUC |
   |---------------|--------------|----------------------|-------------------|
   | ARCH-01 | [32] | X.XX | X.XX |
   | ARCH-03 | [128, 64, 32] | X.XX | X.XX |
   | ...
   ```

4. **Test Set Results** (ONE-TIME, final evaluation)
   ```
   | Dataset | Model | Test PR-AUC | Test Recall | Test F1 | Notes |
   |---------|-------|-------------|-------------|---------|-------|
   | creditcard | Best NN | X.XX | X.XX | X.XX | Optimized threshold applied |
   ```

5. **Generalization Analysis**
   - Training-validation gap comparison across datasets
   - Which architectural choices transferred successfully
   - Dataset-specific findings vs generalizable insights

---

## 🎯 Final Checklist Summary (27 Mandatory Rules)

### Split Creation
1. ✅ Use stratified splitting to preserve class balance
2. ✅ Create splits BEFORE any preprocessing
3. ✅ Save split indices as `.npy` files immediately
4. ✅ Never re-split data in subsequent notebooks
5. ✅ Maintain SEPARATE splits for each dataset

### Scaler Fitting
6. ✅ Fit StandardScaler ONLY on training set
7. ✅ Transform validation and test sets using fitted scaler (never re-fit)
8. ✅ Save fitted scaler as `.pkl` immediately after fitting
9. ✅ All subsequent notebooks load saved scaler (never re-fit)
10. ✅ Maintain SEPARATE scalers for each dataset

### Hyperparameter Selection
11. ✅ All hyperparameter decisions based ONLY on validation set
12. ✅ Architecture selection: based on validation PR-AUC
13. ✅ Regularization selection: based on validation PR-AUC
14. ✅ Never use test set for model selection

### Threshold Optimization
15. ✅ Optimize classification threshold ONLY on validation set
16. ✅ Test ALL candidate thresholds on validation set
17. ✅ Select optimal threshold before touching test set
18. ✅ Apply fixed threshold to test set (no iteration)

### Test Set Protocol
19. ✅ Test set evaluation happens EXACTLY ONCE per dataset
20. ✅ Test evaluation is the FINAL step in Notebook 06
21. ✅ No model re-training after seeing test results
22. ✅ No threshold adjustment after seeing test results
23. ✅ Test results are reported as-is, with limitations discussed

### Cross-Dataset Isolation
24. ✅ NEVER merge datasets
25. ✅ NEVER train on one dataset and evaluate on another
26. ✅ NEVER use synthetic data to supplement real data
27. ✅ Compare results only through final analysis (Notebook 07)

---

**Remember:** Data leakage can make a mediocre model appear excellent in experiments but fail catastrophically in production. Strict adherence to this checklist ensures your neural network truly generalizes to unseen data.
