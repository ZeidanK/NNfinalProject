# Neural Network Design Rationale - Dual-Dataset Study

## Executive Summary

This document explains the reasoning behind neural network architecture choices for credit card fraud detection across **two datasets** with distinct characteristics:

1. **card_transdata.csv** (Synthetic): Controlled environment for systematic architecture exploration
2. **creditcard.csv** (Real-world): Production-grade validation benchmark

The dual-dataset approach ensures that design principles discovered in clean experimental conditions generalize to real-world fraud detection scenarios.

---

## Why Neural Networks for Credit Card Fraud Detection?

### Beyond Beating Baselines

**Critical Clarification:** This project does NOT aim to prove neural networks always beat Random Forest or Logistic Regression. Instead, we demonstrate:

1. **Transferable Design Principles**: Architectural choices (depth vs width, regularization) that work across data regimes
2. **Controlled Overfitting**: NNs can be regularized to generalize despite high capacity
3. **Interpretable Architecture**: Principled design process vs black-box hyperparameter search
4. **Production Readiness**: Stable training dynamics and threshold optimization

### When NNs Provide Value

Neural networks excel when:
- Non-linear feature interactions are critical
- Continuous probability outputs needed (threshold tuning)
- Representation learning beneficial (intermediate abstractions)
- Scalability to large datasets required
- Transfer learning from related tasks possible

**Expected Outcome on Synthetic Data**: Random Forest may achieve near-perfect performance (PR-AUC → 1.0) due to high feature separability. This is **acceptable** - the synthetic dataset's role is enabling clean ablation studies, not challenging baselines.

**Expected Outcome on Real-World Data**: Neural networks demonstrate competitive performance with controlled overfitting, validating design principles from synthetic experiments.

---

## 1. Problem Characteristics (Dual-Dataset)

### Dataset 1: card_transdata.csv (Synthetic)
- **Data Type:** Tabular (7 interpretable features)
- **Sample Size:** ~1,000,000 transactions
- **Class Distribution:** ~0.8% fraud (moderate imbalance)
- **Feature Types:** Continuous distances, binary indicators
- **Difficulty:** High separability (near-perfect baseline performance expected)
- **Role:** Architecture exploration, ablation studies

### Dataset 2: creditcard.csv (Real-World)
- **Data Type:** Tabular (30 PCA-transformed features)
- **Sample Size:** ~284,000 transactions  
- **Class Distribution:** ~0.17% fraud (extreme imbalance)
- **Feature Types:** PCA components V1-V28, plus Time and Amount
- **Difficulty:** Real-world noise, ambiguity, edge cases
- **Role:** Production-grade validation, final evaluation

### Challenge: Extreme Class Imbalance (Both Datasets)
The most critical challenge is severe class imbalance. Neural networks must:
- Focus on minority class (fraud) detection
- Balance precision (minimize false alarms) with recall (catch actual fraud)
- Learn from very few positive examples
- Handle class weights effectively

---

## 2. Dual-Dataset Experimental Design

### Phase 1: Architecture Exploration (card_transdata.csv)

**Objective**: Systematically compare 8 MLP architectures in controlled environment

**Architectures to Test**:
1. **ARCH-01**: [32] - Minimal capacity baseline
2. **ARCH-02**: [64, 32] - Small network
3. **ARCH-03**: [128, 64, 32] - Balanced (likely optimal)
4. **ARCH-04**: [256, 128, 64, 32] - Moderate depth
5. **ARCH-05**: [512, 256, 128, 64, 32, 16] - Maximum depth
6. **ARCH-06**: [256] - Wide-shallow
7. **ARCH-07**: [512] - Very wide-shallow
8. **ARCH-08**: [256, 128, 64] - Depth-width balance

**Selection Criterion**: Best validation PR-AUC → transfer to Phase 2

**Why Synthetic Data for This Phase**:
- Clean ablation studies without confounding factors
- Fast experimentation (simpler feature space)
- Isolate architectural effects from data complexity
- **Baseline dominance expected and acceptable**

### Phase 2: Regularization Optimization (creditcard.csv)

**Objective**: Validate best architecture on real-world data with proper regularization

**Regularization Strategies**:
- Dropout: [0.2, 0.3, 0.4]
- L2 regularization: [0.001, 0.01]
- Batch Normalization: [True, False]
- Combined strategies

**Selection Criterion**: Best validation PR-AUC → use for test evaluation

**Why Real-World Data for This Phase**:
- Production-grade benchmark
- Extreme imbalance challenges regularization
- PCA features test generalization beyond interpretable features
- **Authoritative final evaluation**

---

## 3. Why Neural Networks (vs Classical ML)?

### Advantages of Neural Networks for This Problem

#### 2.1 Non-Linear Pattern Learning
Fraud patterns are likely non-linear combinations of features:
- Distance from home + online order interaction
- Time patterns + transaction amount relationships
- Complex decision boundaries that linear models miss

**MLPs excel at:** Learning hierarchical non-linear transformations

#### 2.2 Representation Learning
Neural networks can learn intermediate representations:
- Lower layers: Simple feature combinations
- Middle layers: Abstract fraud patterns
- Upper layers: High-level fraud indicators

**Example:** Distance features might combine in middle layers to detect "unusual location" patterns

#### 2.3 Flexible Class Imbalance Handling
Neural networks offer multiple imbalance strategies:
- Class weights in loss function
- Custom loss functions (focal loss)
- Threshold optimization on probability outputs
- Batch-level sampling strategies

Classical ML has fewer options and less flexibility.

#### 2.4 Scalability
Neural networks scale well to:
- Large datasets (hundreds of thousands of transactions)
- Real-time prediction (fast forward pass)
- Online learning (incremental updates possible)

---

## 3. MLP Architecture Design for Tabular Data

### 3.1 Input Layer Design

**Feature Count:** 7 input features
```
Input Layer: 7 neurons (one per feature)
```

**Preprocessing Requirements:**
- StandardScaler for continuous features (distance_from_home, etc.)
- Binary features (0/1) already normalized
- No embedding layers needed (not categorical)

### 3.2 Hidden Layer Architecture Rationale

#### Shallow MLPs ([32] or [64, 32])
**When to use:**
- Baseline comparison
- When data is linearly separable
- Risk of overfitting with limited data

**Expected behavior:**
- Fast training
- May underfit complex patterns
- Good starting point

#### Medium MLPs ([128, 64, 32] or [256, 128, 64, 32])
**When to use:**
- Moderate complexity fraud patterns
- Balance between capacity and overfitting
- Most likely optimal for this problem

**Expected behavior:**
- Capture non-linear relationships
- Reasonable training time
- Good generalization with proper regularization

#### Deep MLPs (5+ layers [512, 256, 128, 64, 32, 16])
**When to use:**
- Very complex pattern hierarchies
- Large datasets (we have this)
- When medium MLPs underfit

**Expected behavior:**
- Highest capacity
- Longer training time
- Risk of overfitting without regularization
- May learn hierarchical fraud indicators

### 3.3 Width vs Depth Trade-offs

**Wide Shallow ([1024]):**
- Learns many features at once
- Less hierarchical reasoning
- May memorize training patterns

**Narrow Deep ([64, 64, 64, 64]):**
- Sequential feature transformation
- More hierarchical representations
- Better for compositional patterns

**Hypothesis:** For fraud detection, moderate depth + width is optimal since fraud patterns are non-linear but not extremely hierarchical.

---

## 4. Activation Functions

### Hidden Layers: ReLU
**Choice:** ReLU (Rectified Linear Unit)

**Rationale:**
- Prevents vanishing gradient (critical for deep networks)
- Computationally efficient
- Introduces non-linearity
- Standard choice for hidden layers

**Alternatives considered:**
- Leaky ReLU: For dead neuron prevention (experiment if needed)
- ELU: Smoother gradients (not necessary for tabular data)

### Output Layer: Sigmoid
**Choice:** Sigmoid activation

**Rationale:**
- Binary classification (fraud vs legitimate)
- Output range [0, 1] interpreted as probability
- Compatible with binary crossentropy loss
- Enables threshold optimization

---

## 5. Regularization Strategy

### 5.1 Dropout
**Purpose:** Prevent co-adaptation of neurons

**Placement:** After each hidden layer (except last)

**Rate range:** 0.2 - 0.5
- Lower (0.2): Light regularization for medium MLPs
- Higher (0.5): Aggressive regularization for deep MLPs

**Why essential:** Large dataset + complex networks = overfitting risk

### 5.2 L2 Regularization (Weight Decay)
**Purpose:** Penalize large weights

**Range:** 0.001 - 0.1
- 0.001: Light penalty, mostly for deep layers
- 0.01: Standard choice
- 0.1: Aggressive (may underfit)

**Why essential:** Prevents weights from growing too large, improves generalization

### 5.3 Batch Normalization
**Purpose:** Normalize layer inputs

**Placement:** Before or after activation (experiment both)

**Benefits:**
- Stabilizes training
- Allows higher learning rates
- Acts as mild regularization
- Reduces internal covariate shift

**Trade-offs:**
- Adds computational cost
- May interact with Dropout (order matters)
- Not always beneficial for small networks

---

## 6. Loss Function Selection

### 6.1 Binary Crossentropy (Baseline)
```python
loss = 'binary_crossentropy'
```
**When to use:** Standard starting point

**Limitation:** Treats all misclassifications equally (problematic for imbalance)

### 6.2 Weighted Binary Crossentropy
```python
class_weight = {0: 1.0, 1: weight_fraud}
```
**When to use:** Always for imbalanced data

**Calculation:**
```
weight_fraud = n_legitimate / n_fraud
```
This makes fraud misclassifications more costly.

### 6.3 Focal Loss (Advanced)
**Formula:**
```
FL(p_t) = -α(1-p_t)^γ * log(p_t)
```
**When to use:** When weighted BCE insufficient

**Benefits:**
- Focuses on hard examples
- Down-weights easy negatives (abundant legitimate transactions)
- γ parameter controls focusing strength

---

## 7. Optimization Strategy

### 7.1 Optimizer: Adam
**Choice:** Adam (Adaptive Moment Estimation)

**Rationale:**
- Adaptive learning rates per parameter
- Works well out-of-the-box
- Handles sparse gradients (relevant for imbalanced data)
- Combines momentum + RMSProp benefits

**Hyperparameters:**
- learning_rate: 0.001 (default, adjust if needed)
- beta_1: 0.9 (momentum)
- beta_2: 0.999 (variance)

**Alternatives:** RMSProp (if Adam oscillates), SGD with momentum (slower but sometimes better generalization)

### 7.2 Learning Rate Schedule
**Strategy:** Reduce on plateau

```python
ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6
)
```

**Rationale:** Allow fine-tuning when validation loss plateaus

### 7.3 Early Stopping
**Configuration:**
```python
EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)
```

**Rationale:**
- Prevents overfitting
- Saves time (no need to run full epochs)
- Returns best model automatically

---

## 8. Class Imbalance Handling in NNs

### 8.1 Class Weights (Primary Strategy)
**Implementation:**
```python
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
```

**Effect:** Fraud class errors contribute more to loss

### 8.2 Threshold Optimization (Post-Training)
**Process:**
1. Train NN with class weights
2. Generate predictions on validation set
3. Compute precision-recall for different thresholds
4. Select threshold optimizing F1-score or business metric

**Why essential:** Default 0.5 threshold is suboptimal for imbalanced data

### 8.3 Sampling Strategies (Alternative)
**Options:**
- Oversampling fraud class (SMOTE)
- Undersampling legitimate class
- Batch-level balancing

**Not primary choice because:**
- Oversampling can cause overfitting
- Undersampling loses information
- Class weights achieve similar effect more efficiently

---

## 9. Evaluation Metrics Rationale

### Why NOT Accuracy?
With 99.2% legitimate transactions:
```python
# Dummy model predicting "all legitimate"
accuracy = 99.2%  # Looks great!
fraud_recall = 0%  # Catches zero fraud!
```

### Primary Metrics for NNs

#### 9.1 Precision (Fraud Class)
**Formula:** TP / (TP + FP)

**Interpretation:** Of predicted fraud, what % is actually fraud?

**Business impact:** High precision = fewer false alarms = less customer frustration

#### 9.2 Recall (Fraud Class)
**Formula:** TP / (TP + FN)

**Interpretation:** Of actual fraud, what % did we catch?

**Business impact:** High recall = catch more fraud = reduce financial loss

#### 9.3 F1-Score (Fraud Class)
**Formula:** 2 * (Precision * Recall) / (Precision + Recall)

**Interpretation:** Harmonic mean of precision and recall

**Use:** Model selection when precision and recall equally important

#### 9.4 PR-AUC (Precision-Recall Area Under Curve)
**Why best for this problem:**
- Summarizes precision-recall trade-off
- More informative than ROC-AUC for imbalanced data
- Threshold-independent
- Direct business relevance

#### 9.5 ROC-AUC (Secondary)
**Use:** Compare against baselines

**Limitation:** Can be misleadingly high for imbalanced data

---

## 10. Ablation Study Design

### Purpose
Isolate individual component contributions to understand:
- Which regularization helps most?
- Is complexity justified?
- What's the minimal viable architecture?

### Base Architecture
```python
base_model = [128, 64, 32]  # Medium MLP
```

### Ablation Experiments
1. Base (no regularization)
2. Base + Dropout (0.3)
3. Base + L2 (0.01)
4. Base + BatchNorm
5. Base + Dropout + L2
6. Base + Dropout + L2 + BatchNorm

### Analysis
Compare validation loss and fraud F1-score across variants to determine:
- Most effective single regularization technique
- Whether regularization combinations synergize or interfere
- Optimal regularization configuration

---

## 11. Expected Outcomes & Hypotheses

### Hypothesis 1: Medium Depth Optimal
**Prediction:** 3-4 layer MLPs ([128, 64, 32] or [256, 128, 64, 32]) will outperform both shallow and very deep networks.

**Reasoning:** Fraud patterns are non-linear but not extremely hierarchical. Too shallow = underfit, too deep = overfit.

### Hypothesis 2: Dropout Most Effective
**Prediction:** Dropout will provide greatest regularization benefit.

**Reasoning:** Prevents co-adaptation, forces redundant learning, reduces overfitting.

### Hypothesis 3: Class Weights Essential
**Prediction:** Without class weights, NNs will predict mostly legitimate transactions.

**Reasoning:** Gradient contributions overwhelmingly from majority class without weighting.

### Hypothesis 4: Threshold Optimization Improves F1
**Prediction:** Optimal threshold will be < 0.5 (favor recall).

**Reasoning:** Cost of missing fraud typically exceeds cost of false alarm investigation.

---

## 12. Comparison with Classical ML

### Neural Networks vs Logistic Regression

| Aspect | Logistic Regression | Neural Network |
|--------|-------------------|----------------|
| Decision Boundary | Linear | Non-linear |
| Feature Engineering | Manual required | Learned automatically |
| Training Time | Fast | Slower |
| Interpretability | High (coefficients) | Low (black box) |
| Capacity | Low | High |
| Overfitting Risk | Low | High (needs regularization) |
| Class Imbalance Handling | Limited options | Multiple strategies |

**Expected result:** NNs outperform LR if non-linearity exists in fraud patterns.

### Neural Networks vs Random Forest

| Aspect | Random Forest | Neural Network |
|--------|--------------|----------------|
| Handling Mixed Features | Excellent | Requires normalization |
| Interpretability | Medium (feature importance) | Low |
| Hyperparameter Sensitivity | Low | High |
| Training Time | Medium | Varies with architecture |
| Ensemble Nature | Built-in | Single model |
| Probabilistic Output | Yes | Yes |

**Expected result:** Comparable performance, but NNs offer more tuning options.

---

## 13. Production Deployment Considerations

### Model Serving
- **Format:** Save as `.h5` or SavedModel format
- **Inference Speed:** <10ms per prediction (fast forward pass)
- **Batch Prediction:** Support for real-time and batch scoring

### Threshold Deployment
- **Dynamic Threshold:** Adjust based on business priorities
- **A/B Testing:** Compare thresholds in production
- **Monitoring:** Track precision/recall over time

### Model Updates
- **Retraining Schedule:** Monthly or when drift detected
- **Online Learning:** Possible with neural networks
- **Version Control:** Track model performance over time

---

## 14. Key Takeaways for Neural Network Design

1. **Start Medium, Then Explore:** [128, 64, 32] as baseline, expand/contract based on results
2. **Always Use Class Weights:** Essential for imbalanced data
3. **Regularize Aggressively:** Dropout + L2 prevent overfitting
4. **Optimize Threshold:** Don't use default 0.5
5. **Focus on PR-AUC:** Best metric for imbalanced classification
6. **Validate Learning Curves:** Ensure proper generalization
7. **Ablate Components:** Understand what actually helps
8. **Compare with Baselines:** Justify neural network complexity

---

This rationale document ensures that every neural network design choice is justified, making the project a demonstration of deep understanding rather than random experimentation.
