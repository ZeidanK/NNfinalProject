# Data Leakage Prevention Checklist

## Overview
Data leakage occurs when information from outside the training dataset is used to create the model, leading to overly optimistic performance estimates that don't generalize to real-world scenarios. This checklist ensures proper experimental methodology.

---

## ✅ Split Creation

### Before Creating Splits
- [ ] **Understand the target distribution** - Check class balance before splitting
- [ ] **Decide on split ratios** - 70/15/15 for train/val/test is used in this project
- [ ] **Set random seed** - Use `RANDOM_SEED = 42` for reproducibility

### During Split Creation
- [ ] **Stratified split** - Preserve class distribution across train/val/test sets
- [ ] **Test set isolation** - Physically separate test set immediately after creation
- [ ] **Save split indices** - Store train/val/test indices as `.npy` files for reuse
- [ ] **Verify distributions** - Confirm class balance is preserved in all splits
- [ ] **Document split sizes** - Record number of samples in each split

### After Split Creation
- [ ] **Never modify splits** - Use the same indices across all experiments
- [ ] **Test set quarantine** - Do not look at test set until final evaluation
- [ ] **Validation set purpose** - Use only for hyperparameter tuning and model selection

---

## ✅ Feature Preprocessing

### StandardScaler (Feature Normalization)
- [ ] **Fit on training data only** - `scaler.fit(X_train)`
- [ ] **Transform all splits separately** - `scaler.transform(X_train/val/test)`
- [ ] **Save fitted scaler** - Store scaler as `.pkl` file
- [ ] **Reuse same scaler** - Apply same scaler to all experiments
- [ ] **Never fit on test data** - Test data must be "unseen" to preprocessing

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

## 📋 Verification Protocol

Before running final test evaluation, verify:

```python
# Verification checklist code
import numpy as np

# 1. Verify split indices are saved and consistent
assert np.load('results/train_indices.npy').shape[0] > 0
assert np.load('results/val_indices.npy').shape[0] > 0
assert np.load('results/test_indices.npy').shape[0] > 0

# 2. Verify no overlap between splits
train_idx = set(np.load('results/train_indices.npy'))
val_idx = set(np.load('results/val_indices.npy'))
test_idx = set(np.load('results/test_indices.npy'))

assert len(train_idx.intersection(val_idx)) == 0, "Train/val overlap!"
assert len(train_idx.intersection(test_idx)) == 0, "Train/test overlap!"
assert len(val_idx.intersection(test_idx)) == 0, "Val/test overlap!"

# 3. Verify scaler was fit on training data only
# (Check scaler fitting code in notebook 02)

# 4. Verify test set not used in experiments CSV
import pandas as pd
experiments = pd.read_csv('results/experiment_logs/nn_experiments.csv')
# Should not see 'test' in dataset column until final evaluation

print("✅ All leakage prevention checks passed!")
```

---

## 📚 References

- **Cross-Validation Best Practices:** Use stratified K-fold on training+validation only
- **Temporal Data:** For time-series, ensure no future information leaks to past
- **External Data:** If using external datasets, ensure they don't contain test info
- **Production Deployment:** Same leakage prevention applies in production pipelines

---

## 🎓 Learning Objectives

By following this checklist, you demonstrate:

1. **Understanding of experimental methodology** - Proper scientific approach
2. **Knowledge of model evaluation** - Valid performance estimation
3. **Production ML awareness** - Real-world deployment considerations
4. **Statistical rigor** - Avoiding biased performance estimates
5. **Reproducibility standards** - Enabling others to verify results

---

**Remember:** Data leakage can make a mediocre model appear excellent in experiments but fail catastrophically in production. Strict adherence to this checklist ensures your neural network truly generalizes to unseen data.
