# Dual-Dataset Methodology for Neural Network Fraud Detection

## Overview

This project employs a **dual-dataset experimental design** to systematically investigate how neural network architectural and regularization decisions generalize across fraud detection data regimes. This methodology addresses a critical limitation in neural network research: findings derived from single datasets may reflect dataset-specific artifacts rather than fundamental design principles.

---

## Dataset Roles

### Dataset 1: card_transdata.csv (Synthetic - Exploration)

**Role**: Controlled environment for architecture exploration and ablation studies

**Characteristics**:
- ~1,000,000 transactions
- 7 interpretable features (distances, ratios, binary indicators)
- ~0.8% fraud rate (moderate class imbalance)
- Near-perfect feature separability
- Clean, noise-free synthetic generation

**Purpose**:
- Test 8 neural network architectures systematically
- Conduct controlled ablation studies (dropout, L2, batch normalization)
- Isolate effects of design decisions without confounding factors
- Identify best architecture for transfer to real-world data

**Expected Baseline Behavior**:
- Random Forest likely to achieve PR-AUC ≈ 1.0 (near-perfect)
- Linear models (Logistic Regression) may also perform very well
- **This dominance is acceptable and expected** - synthetic data's primary value is enabling clean comparisons, not challenging baselines

---

### Dataset 2: creditcard.csv (Real-World - Validation)

**Role**: Authoritative benchmark for validating design principles

**Characteristics**:
- ~284,000 transactions
- 30 features (28 PCA components + Time + Amount)
- ~0.17% fraud rate (extreme class imbalance)
- Feature anonymization via PCA transformation
- Real-world noise, ambiguity, and edge cases

**Purpose**:
- Validate that architectural insights from synthetic data generalize
- Test regularization strategies under realistic conditions
- Demonstrate production-grade neural network performance
- Provide authoritative final evaluation on held-out test set

**Expected Baseline Behavior**:
- Baselines provide realistic reference points (not guaranteed dominance)
- Extreme imbalance makes this dataset significantly harder
- Neural networks demonstrate competitive performance with controlled overfitting

---

## Why This Approach Matters

### Problem with Single-Dataset Studies

Most neural network fraud detection papers use a single dataset, leading to:
1. **Unclear Generalization**: Do findings transfer to other data regimes?
2. **Confounded Results**: Is performance due to architecture or dataset peculiarities?
3. **Limited Insights**: Can't separate design principles from data-specific tuning

### Benefits of Dual-Dataset Design

1. **Clean Exploration**: Synthetic data enables systematic architecture comparisons without noise
2. **Robust Validation**: Real-world data confirms findings aren't synthetic-only artifacts
3. **Transferable Insights**: Design choices tested across two distinct data regimes
4. **Honest Reporting**: Synthetic baseline dominance acknowledged, not hidden
5. **Academic Rigor**: Separates exploratory analysis from confirmatory evaluation

---

## Experimental Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│ Phase 1: ARCHITECTURE EXPLORATION (card_transdata.csv)         │
├─────────────────────────────────────────────────────────────────┤
│ 1. EDA and baseline training (LR, RF)                          │
│ 2. Test 8 architectures: shallow/medium/deep/wide              │
│ 3. Conduct ablation: +dropout, +L2, +batchnorm                 │
│ 4. Select best architecture by validation PR-AUC               │
│                                                                 │
│ OUTPUT: Best architecture config (e.g., [256, 128, 64])        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 2: REGULARIZATION OPTIMIZATION (creditcard.csv)          │
├─────────────────────────────────────────────────────────────────┤
│ 1. EDA and baseline training (LR, RF) on NEW dataset           │
│ 2. Apply best architecture from Phase 1                        │
│ 3. Test regularization: dropout [0.2-0.4], L2 [0.001-0.01]     │
│ 4. Select best model by validation PR-AUC                      │
│ 5. Optimize classification threshold on validation set         │
│                                                                 │
│ OUTPUT: Production-ready model with optimized threshold        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 3: ONE-TIME TEST EVALUATION (creditcard.csv)             │
├─────────────────────────────────────────────────────────────────┤
│ 1. Load best model and optimized threshold                     │
│ 2. Evaluate on held-out test set EXACTLY ONCE                  │
│ 3. Compare to baseline performance on test set                 │
│ 4. Error analysis (FP/FN patterns, business costs)             │
│                                                                 │
│ OUTPUT: Final test performance (reported as-is)                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 4: CROSS-DATASET ANALYSIS (Both datasets)                │
├─────────────────────────────────────────────────────────────────┤
│ 1. Compare NN behavior: overfitting, stability, generalization │
│ 2. Analyze which design choices transfer across datasets       │
│ 3. Synthesize findings and recommendations                     │
│ 4. Discuss limitations and future work                         │
│                                                                 │
│ OUTPUT: Generalizable insights about NN fraud detection        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Isolation Guarantees

### Strict Non-Contamination Rules

To ensure valid cross-dataset comparisons:

1. **Never merge datasets** - Each dataset maintains completely separate:
   - Train/validation/test splits
   - Fitted scalers
   - Experiment logs
   - Saved models

2. **No cross-dataset training** - Models trained on one dataset are NEVER evaluated on the other

3. **Independent preprocessing** - Each dataset undergoes its own:
   - Stratified splitting (70/15/15)
   - StandardScaler fitting (training set only)
   - Feature engineering (if any)

4. **Separate hyperparameter selection** - Architecture chosen on card_transdata, but regularization optimized independently on creditcard

5. **Comparison happens post-hoc** - Results aggregated in Notebook 07 after all experiments complete

---

## Addressing Random Forest Dominance

### Why Synthetic Data Baselines May Be "Too Good"

The synthetic dataset (card_transdata.csv) exhibits high feature separability, allowing tree-based ensembles to achieve near-perfect performance (PR-AUC → 1.0). This is **not a methodological failure** - it reflects the dataset's intended role as a controlled experimental environment.

### Why This Doesn't Invalidate Neural Networks

1. **Value lies in transferable insights**: NN design principles discovered on synthetic data generalize to real data
2. **Interpretability**: NN architectural choices are principled (depth vs width, regularization), unlike black-box tree ensembles
3. **Stability**: NNs demonstrate controlled overfitting across data regimes
4. **Production relevance**: Real-world validation (creditcard.csv) is the authoritative benchmark

### Academic Framing

The project answers: **"Do neural network design principles generalize across fraud detection data regimes?"**

NOT: **"Do neural networks beat Random Forest on synthetic data?"**

By separating exploration (synthetic) from validation (real-world), we demonstrate that NN design is robust beyond simplified benchmarks.

---

## Reproducibility Protocol

### Mandatory Procedures

1. **Global random seed**: 42 (set in config.py)
2. **Split reproducibility**: Save indices as .npy immediately after splitting
3. **Scaler reproducibility**: Save fitted scaler as .pkl immediately after fitting
4. **Experiment logging**: CSV logs with 25+ columns per experiment
5. **Model checkpointing**: Save best model by validation metric
6. **One-time test evaluation**: Test set touched EXACTLY once per dataset

### Cross-Dataset Comparison

Results from both datasets are aggregated by:
- **Experiment ID**: Same IDs used for equivalent configurations (e.g., ARCH-03 = [128, 64, 32] on both datasets)
- **Metric tracking**: Same metrics computed (PR-AUC, Recall, F1)
- **Overfitting analysis**: Training-validation gap compared across datasets

---

## Success Criteria

### Phase 1 (card_transdata.csv)
- ✅ Test all 8 architectures with proper logging
- ✅ Complete ablation study with statistical rigor
- ✅ Identify best architecture by validation PR-AUC
- ⚠️ **Not required**: Neural networks beating Random Forest

### Phase 2 (creditcard.csv)
- ✅ Apply best architecture from Phase 1
- ✅ Test regularization strategies systematically
- ✅ Optimize threshold on validation set
- ✅ Achieve competitive performance vs baselines

### Phase 3 (Test Evaluation)
- ✅ ONE-TIME test evaluation with no iteration
- ✅ Compare NN to baselines on test set
- ✅ Report results as-is with limitations

### Phase 4 (Cross-Dataset Analysis)
- ✅ Demonstrate design principles transfer
- ✅ Show controlled overfitting across datasets
- ✅ Provide actionable recommendations

---

## Limitations and Future Work

### Known Limitations

1. **Synthetic data simplicity**: May not capture all real-world complexities
2. **Single real-world dataset**: Additional datasets would strengthen generalization claims
3. **Architecture search scope**: Limited to fully-connected MLPs (no CNNs, RNNs, Transformers)
4. **Computational budget**: Cannot exhaustively test all hyperparameter combinations

### Future Directions

1. **Additional datasets**: Test on other fraud detection benchmarks (PaySim, IEEE-CIS)
2. **Advanced architectures**: Explore attention mechanisms, graph neural networks
3. **Transfer learning**: Pre-train on synthetic, fine-tune on real data
4. **Ensemble methods**: Combine NNs with tree-based models
5. **Temporal modeling**: Incorporate transaction sequences

---

## Conclusion

This dual-dataset methodology provides a rigorous framework for neural network fraud detection research. By separating controlled exploration from real-world validation, we ensure that findings are robust, generalizable, and production-relevant rather than artifacts of a single dataset's peculiarities.
