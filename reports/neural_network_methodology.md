# Neural Network Methodology Report: Dual-Dataset Study

## Executive Summary

This report documents a comprehensive dual-dataset methodology for training and evaluating neural networks on imbalanced credit card fraud detection data. The approach uses a **synthetic dataset for architecture exploration** and a **real-world dataset for validation**, emphasizing **reproducibility, data leakage prevention, cross-dataset generalization, and neural network-specific techniques** for handling extreme class imbalance.

**Key Innovation:** By systematically exploring neural network designs on clean synthetic data and validating on production-grade data, we identify transferable design principles that generalize across fraud detection regimes.

---

## 1. Dual-Dataset Overview

### 1.1 Research Question

**Do neural network architectural choices (depth, width, regularization) generalize across fraud detection data regimes?**

### 1.2 Dataset 1: card_transdata.csv (Synthetic - Exploration Phase)

**Source & Description:**
- **Dataset:** Credit Card Fraud Detection (Synthetic)
- **Source:** https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud
- **Size:** ~1,000,000 transactions
- **Role:** Clean environment for architecture exploration and ablation studies

**Features (7 interpretable):**
1. `distance_from_home` - Distance from home location (continuous)
2. `distance_from_last_transaction` - Distance from previous transaction (continuous)
3. `ratio_to_median_purchase_price` - Purchase price relative to median (continuous)
4. `repeat_retailer` - Whether retailer was used before (binary)
5. `used_chip` - Whether chip was used for transaction (binary)
6. `used_pin_number` - Whether PIN was entered (binary)
7. `online_order` - Whether transaction was online (binary)

**Target Variable:** `fraud` - Binary indicator (0 = legitimate, 1 = fraud)

**Class Distribution:**
- **Legitimate Transactions:** ~99.2%
- **Fraudulent Transactions:** ~0.8%
- **Imbalance Ratio:** 1:124

**Value:** Clean patterns enable systematic architecture exploration without confounding factors. Random Forest achieves near-perfect performance (PR-AUC ≈ 1.0), making it ideal for controlled experiments.

### 1.3 Dataset 2: creditcard.csv (Real-World - Validation Phase)

**Source & Description:**
- **Dataset:** Credit Card Fraud Detection (ULB - Université Libre de Bruxelles)
- **Source:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **Size:** ~284,000 transactions
- **Role:** Production-grade validation benchmark

**Features (30 PCA-transformed + 2):**
- `V1` to `V28` - PCA-transformed features (anonymized)
- `Time` - Seconds elapsed between transactions
- `Amount` - Transaction amount

**Target Variable:** `Class` - Binary indicator (0 = legitimate, 1 = fraud)

**Class Distribution:**
- **Legitimate Transactions:** ~99.83%
- **Fraudulent Transactions:** ~0.17%
- **Imbalance Ratio:** 1:577

**Value:** Extreme imbalance and PCA transformation represent production scenarios. Authoritative test of generalization.

### 1.4 Why Dual-Dataset Approach?

**Benefits:**
1. **Clean Exploration:** Synthetic data enables systematic architecture testing
2. **Production Validation:** Real-world data confirms generalization
3. **Transferable Insights:** Identify principles that work across regimes
4. **Honest Evaluation:** Prevents overfitting to single dataset quirks
5. **Comprehensive Understanding:** Different imbalance ratios and feature types

---

## 2. Seven-Notebook Workflow

### 2.1 Notebook Architecture Overview

**Phase 1: Dual-Dataset Overview (Notebook 01)**
- Comparative exploratory data analysis
- Visualization of both datasets
- Imbalance analysis
- Feature distribution comparison

**Phase 2: card_transdata (Synthetic) - Architecture Exploration (Notebooks 02-03)**
- **Notebook 02:** Preprocessing and baseline models
- **Notebook 03:** NN architecture exploration (8 variants) + ablation study (5 experiments)

**Phase 3: creditcard (Real-World) - Validation (Notebooks 04-06)**
- **Notebook 04:** Preprocessing and baseline models
- **Notebook 05:** NN training with best architecture + regularization experiments
- **Notebook 06:** Threshold optimization + ONE-TIME test evaluation

**Phase 4: Synthesis (Notebook 07)**
- **Notebook 07:** Cross-dataset analysis and transferable design principles

### 2.2 Data Splitting Strategy

**Rationale for 70/15/15 Split:**
- **Training Set (70%):** Sufficient data for neural network learning
- **Validation Set (15%):** Hyperparameter tuning and early stopping
- **Test Set (15%):** Final unbiased performance evaluation

**Critical:** Each dataset split independently with same random seed (42).

### 2.3 Stratified Sampling (Per-Dataset)

**Implementation:**
```python
from sklearn.model_selection import train_test_split

# First split: separate test set
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.15,
    stratify=y,
    random_state=42
)

# Second split: separate train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.176,  # 0.15/0.85 ≈ 0.176 to get 15% of total
    stratify=y_temp,
    random_state=42
)
```

**Verification:** Class distribution preserved across all splits for both datasets.

### 2.4 Split Index Preservation (Dataset-Specific)

**Critical for reproducibility:**
```python
# card_transdata splits
np.save('results/card_transdata/train_indices.npy', train_indices)
np.save('results/card_transdata/val_indices.npy', val_indices)
np.save('results/card_transdata/test_indices.npy', test_indices)

# creditcard splits
np.save('results/creditcard/train_indices.npy', train_indices)
np.save('results/creditcard/val_indices.npy', val_indices)
np.save('results/creditcard/test_indices.npy', test_indices)
```

**Benefit:** All experiments use identical data splits within each dataset, enabling fair comparison.

---

## 3. Feature Preprocessing (Dataset-Aware)

### 3.1 StandardScaler for Neural Network Compatibility

**Why Normalization is Essential:**
- Neural networks sensitive to feature scales
- Gradient descent converges faster with normalized inputs
- Prevents large-scale features from dominating
- Stabilizes weight initialization

**Implementation (Per-Dataset):**
```python
from sklearn.preprocessing import StandardScaler

# Fit ONLY on training data
scaler = StandardScaler()
scaler.fit(X_train)

# Transform all splits
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Save for reuse (dataset-specific)
import joblib
joblib.dump(scaler, 'results/card_transdata/fitted_scaler.pkl')
joblib.dump(scaler, 'results/creditcard/fitted_scaler.pkl')
```

**Critical Rule:** Scaler statistics (mean, std) computed ONLY from training data of each dataset.

### 3.2 Dataset-Specific Considerations

**card_transdata:**
- 7 interpretable features (mix of continuous and binary)
- Binary features {0,1} already scaled
- Feature names directly meaningful

**creditcard:**
- 28 PCA components (already linearly transformed)
- Time and Amount features (different scales)
- StandardScaler critical for Time/Amount normalization

### 3.3 Data Leakage Prevention Checklist

**Per-Dataset Validation:**
- ✅ Scaler fit on training data only (never validation/test)
- ✅ Same scaler applied to val/test within dataset
- ✅ No target variable used in preprocessing
- ✅ Test set isolated until final evaluation (Notebook 06 only)
- ✅ No cross-contamination between datasets
- ✅ Architecture selection based only on card_transdata validation
- ✅ creditcard test set accessed exactly once

---

## 4. Neural Network Architecture Design

### 4.1 Multi-Layer Perceptron (MLP) Structure

**General Architecture:**
```
Input Layer (7 for card_transdata, 30 for creditcard)
    ↓
Dense Layer 1 + ReLU + [Dropout] + [BatchNorm]
    ↓
Dense Layer 2 + ReLU + [Dropout] + [BatchNorm]
    ↓
...
    ↓
Dense Layer N + ReLU + [Dropout] + [BatchNorm]
    ↓
Output Layer (1 neuron) + Sigmoid
```

### 4.2 Architecture Exploration on card_transdata (Notebook 03)

**8 Architecture Variants Tested:**

#### ARCH-01: Shallow Single Layer
- `[32]` - Baseline simplicity

#### ARCH-02: Shallow Two Layers
- `[64, 32]` - Minimal depth

#### ARCH-03: Medium Depth (Expected Optimal)
- `[128, 64, 32]` - Balanced width and depth

#### ARCH-04: Deep Network
- `[256, 128, 64, 32]` - Four hidden layers

#### ARCH-05: Very Deep
- `[512, 256, 128, 64, 32, 16]` - Six hidden layers

#### ARCH-06: Wide Shallow
- `[1024]` - Very wide, single layer (width vs depth test)

#### ARCH-07: Narrow Deep
- `[64, 64, 64, 64]` - Four layers, constant width

#### ARCH-08: Extra Wide Medium
- `[512, 256, 128]` - Wide layers, medium depth

**Selection Criteria:** Best validation PR-AUC on card_transdata determines architecture for creditcard (Notebook 05).

### 4.3 Architecture Transfer to creditcard

**Key Insight:** Best architecture from synthetic data (clean patterns) transferred to real-world data (messy patterns).

**Transfer Process:**
1. Select best ARCH-XX based on card_transdata validation PR-AUC
2. Apply identical layer configuration to creditcard (Notebook 05)
3. Adjust only input layer dimensions (7 → 30 features)
4. Keep all other hyperparameters constant

### 4.4 Activation Functions

**Hidden Layers:** ReLU (Rectified Linear Unit)
- Prevents vanishing gradient
- Computationally efficient
- Non-linear transformation
- Works well for both datasets

**Output Layer:** Sigmoid
- Binary classification
- Output range: [0, 1] (probability)
- Compatible with binary crossentropy loss

---

## 5. Regularization Strategies

### 5.1 Dropout

**Purpose:** Prevent co-adaptation of neurons

**Implementation:**
```python
from tensorflow.keras.layers import Dropout

model.add(Dense(128, activation='relu'))
model.add(Dropout(rate=0.3))  # Drop 30% of neurons during training
```

**Rates Tested in Ablation (Notebook 03):** 0.0, 0.2, 0.3, 0.4, 0.5

**Expected Outcome:** Higher dropout for deeper networks

### 5.2 L2 Regularization (Weight Decay)

**Purpose:** Penalize large weights

**Implementation:**
```python
from tensorflow.keras.regularizers import l2

model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
```

**Values Tested in Regularization Experiments (Notebook 05):** 0.0, 0.001, 0.01, 0.1

**Expected Outcome:** Moderate L2 (0.01) most effective

### 5.3 Batch Normalization

**Purpose:** Normalize layer inputs

**Implementation:**
```python
from tensorflow.keras.layers import BatchNormalization

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
```

**Placement:** After activation (standard approach)

**Benefits:**
- Stabilizes training
- Enables higher learning rates
- Acts as regularization
- Tested in ablation study

### 5.4 Ablation Study Design (Notebook 03 on card_transdata)

**5 Controlled Experiments (Base: [128, 64, 32]):**

| Experiment ID | Description | Dropout | L2 Reg | BatchNorm | Class Weights |
|---------------|-------------|---------|--------|-----------|---------------|
| ABL-01        | Baseline (Minimal) | ✗ | ✗ | ✗ | ✗ |
| ABL-02        | +Dropout | ✓ (0.3) | ✗ | ✗ | ✗ |
| ABL-03        | +L2 Regularization | ✗ | ✓ (0.01) | ✗ | ✗ |
| ABL-04        | +Batch Normalization | ✗ | ✗ | ✓ | ✗ |
| ABL-05        | Full Stack | ✓ (0.3) | ✓ (0.01) | ✓ | ✓ (balanced) |

**Controlled Variables:**
- Same architecture [128, 64, 32]
- Same optimizer (Adam, lr=0.001)
- Same batch size (64)
- Same random seed (42)
- Same data splits

**Analysis Questions:**
1. Which single technique provides most benefit?
2. Do regularization techniques combine synergistically?
3. Is the full stack justified, or is simpler better?

### 5.5 Regularization Experiments (Notebook 05 on creditcard)

**8 Experiments Testing L2 and Dropout Combinations:**

| Experiment ID | Description | Dropout | L2 Reg |
|---------------|-------------|---------|--------|
| REG-01        | No Regularization | 0.0 | 0.0 |
| REG-02        | Light Dropout | 0.2 | 0.0 |
| REG-03        | Medium Dropout | 0.3 | 0.0 |
| REG-04        | Light L2 | 0.0 | 0.001 |
| REG-05        | Medium L2 | 0.0 | 0.01 |
| REG-06        | Light Dropout + Light L2 | 0.2 | 0.001 |
| REG-07        | Medium Dropout + Medium L2 | 0.3 | 0.01 |
| REG-08        | Heavy Regularization | 0.5 | 0.1 |

**Purpose:** Optimize regularization for real-world data after architecture selection.

---

## 6. Class Imbalance Handling

### 6.1 Class Weight Calculation (Dataset-Specific)

**Balanced Strategy (Primary for Both Datasets):**
```python
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weight_dict = {
    0: class_weights[0],  # Legitimate
    1: class_weights[1]   # Fraud (much higher weight)
}
```

**Effect:** Fraud misclassifications contribute proportionally more to loss function.

### 6.2 Imbalance Ratios Comparison

**card_transdata:**
- Legitimate: ~99.2%, Fraud: ~0.8%
- Imbalance ratio: 1:124
- Fraud class weight ≈ 124

**creditcard:**
- Legitimate: ~99.83%, Fraud: ~0.17%
- Imbalance ratio: 1:577
- Fraud class weight ≈ 577

**Insight:** Class weights automatically adjust to dataset imbalance severity, making methodology transferable.

### 6.3 Custom Class Weights

**Formula:**
```python
weight_fraud = n_legitimate / n_fraud
weight_legitimate = 1.0

# card_transdata example: If 992,000 legitimate and 8,000 fraud
weight_fraud = 992000 / 8000 = 124

# creditcard example: If 284,315 legitimate and 492 fraud  
weight_fraud = 284315 / 492 ≈ 577
```

**Interpretation:** Each fraud error penalized proportionally to imbalance.

### 6.4 Focal Loss (Advanced - Optional)

**Purpose:** Focus on hard-to-classify examples

**Formula:**
```
FL(p_t) = -α(1-p_t)^γ * log(p_t)
```

**Implementation:**
```python
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * y_true * K.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return K.mean(K.sum(loss, axis=1))
    return focal_loss_fixed
```

**When to use:** If weighted BCE insufficient for extreme imbalance (evaluated in experiments).

---

## 7. Training Configuration (Standardized Across Datasets)

### 7.1 Optimizer

**Primary Choice:** Adam (Adaptive Moment Estimation)

**Configuration:**
```python
from tensorflow.keras.optimizers import Adam

optimizer = Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07
)
```

**Rationale:**
- Adaptive learning rates per parameter
- Handles sparse gradients (imbalanced data)
- Works well out-of-the-box
- Consistent across both datasets

### 7.2 Loss Function

**Primary:** Binary Crossentropy with Class Weights
```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'Precision', 'Recall', 'AUC']
)

history = model.fit(
    X_train, y_train,
    class_weight=class_weight_dict,  # Essential! Dataset-specific
    ...
)
```

### 7.3 Early Stopping (Prevents Overfitting)

**Configuration:**
```python
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    min_delta=0.0001,
    restore_best_weights=True,
    verbose=1
)
```

**Rationale:**
- Prevents overfitting (especially on creditcard with extreme imbalance)
- Saves training time
- Returns best validation model
- Patience=15 appropriate for both datasets

### 7.4 Learning Rate Reduction

**Configuration:**
```python
from tensorflow.keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)
```

**Benefit:** Fine-tune when training plateaus.

### 7.5 Model Checkpointing (Dataset-Aware)

**Configuration:**
```python
from tensorflow.keras.callbacks import ModelCheckpoint

# card_transdata
checkpoint = ModelCheckpoint(
    'results/card_transdata/models/neural_networks/best_model.h5',
    monitor='val_pr_auc',  # Precision-Recall AUC
    save_best_only=True,
    mode='max',
    verbose=1
)

# creditcard
checkpoint = ModelCheckpoint(
    'results/creditcard/models/neural_networks/best_model.h5',
    monitor='val_pr_auc',
    save_best_only=True,
    mode='max',
    verbose=1
)
```

**Critical:** Save model based on validation metric (PR-AUC), never test.

### 7.6 Hyperparameters (Defined in config.py)

```python
# Global training configuration
BATCH_SIZE = 64
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
LEARNING_RATE = 0.001
```

**Consistency:** Same hyperparameters used for both datasets to ensure fair comparison.

---

## 8. Evaluation Metrics (Focus on Minority Class)

### 8.1 Why Accuracy is Insufficient for Imbalanced Data

**Problem with Accuracy:**
```python
# Dummy classifier predicting all legitimate
predictions = np.zeros(len(y_test))

# card_transdata
accuracy = accuracy_score(y_test, predictions)
# Result: ~99.2% accuracy!

# creditcard (even worse!)
accuracy = accuracy_score(y_test, predictions)
# Result: ~99.83% accuracy!

fraud_recall = recall_score(y_test, predictions)
# Result: 0% recall (catches zero fraud in both datasets)
```

**Conclusion:** Accuracy is dangerously misleading for imbalanced data. A useless model appears excellent.

### 8.2 Primary Metrics (Fraud Class Focus)

#### Precision-Recall AUC (PR-AUC) - PRIMARY MODEL SELECTION METRIC

**Why PR-AUC is Best for This Problem:**
- Threshold-independent summary of precision-recall trade-off
- More informative than ROC-AUC for imbalanced data
- Directly relevant to fraud detection business decisions
- Robust across different imbalance ratios (124:1 vs 577:1)
- Used for model selection in both Notebook 03 and Notebook 05

**Implementation:**
```python
from sklearn.metrics import average_precision_score

pr_auc = average_precision_score(y_val, y_val_proba)
```

#### Precision (Fraud)
**Formula:** TP / (TP + FP)
**Interpretation:** Of predicted fraud, what % is actually fraud?
**Business Impact:** Minimize customer frustration from false alarms.

#### Recall (Fraud)
**Formula:** TP / (TP + FN)
**Interpretation:** Of actual fraud, what % did we catch?
**Business Impact:** Minimize financial loss from missed fraud.

#### F1-Score (Fraud)
**Formula:** 2 × (Precision × Recall) / (Precision + Recall)
**Interpretation:** Harmonic mean balancing precision and recall.
**Use:** Model selection when both metrics equally important.

#### ROC-AUC (Secondary)
**Role:** 
- Useful for comparison with baselines
- Can be misleadingly high for imbalanced data
- Less directly interpretable than PR-AUC
- Reported but not used for model selection

### 8.3 Confusion Matrix Analysis (Business Context)

**Structure:**
```
                  Predicted
                  Neg    Pos
Actual  Neg      TN     FP
        Pos      FN     TP
```

**Business Interpretation:**
- **True Negatives (TN):** Correctly identified legitimate transactions
- **False Positives (FP):** Legitimate transactions flagged as fraud (customer frustration)
- **False Negatives (FN):** Fraud missed by model (financial loss)
- **True Positives (TP):** Fraud successfully detected (system working!)

**Cost Analysis (Example):**
- FN cost: $100 per missed fraud (average fraud amount)
- FP cost: $25 per false alarm (customer service cost + potential churn)
- FN cost typically 4× FP cost
- Adjust threshold based on business priorities (Notebook 06)

### 8.4 Cross-Dataset Metric Comparison (Notebook 07)

**Comparative Analysis:**
- How do PR-AUC scores compare across datasets?
- Do architecture rankings generalize?
- Which regularization strategies work for both?
- Identify transferable design principles

---

## 9. Threshold Optimization (Notebook 06 on creditcard)

### 9.1 Why Default 0.5 is Suboptimal

Neural networks output probabilities [0, 1]. Default threshold:
```python
prediction = 1 if probability >= 0.5 else 0
```

**Problem:** This assumes:
- Equal class distribution (violated: 0.17% fraud vs 99.83% legitimate)
- Equal misclassification costs (violated: FN cost >> FP cost)

Both assumptions severely violated in fraud detection!

### 9.2 Validation-Based Threshold Selection

**Process (on creditcard validation set):**
```python
from sklearn.metrics import precision_recall_curve

# Get probabilities on validation set
y_val_proba = best_model.predict(X_val_scaled)

# Compute precision-recall for all thresholds
precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_proba)

# Calculate F1 for each threshold
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

# Select threshold maximizing F1
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal threshold: {optimal_threshold:.3f}")
```

**Typical result:** Optimal threshold < 0.5 (favor recall for fraud detection).

**Threshold candidates tested (config.py):**
```python
THRESHOLD_CANDIDATES = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
```

### 9.3 Business-Driven Threshold Selection

**Cost-Sensitive Approach:**
```python
cost_fn = 100  # Cost of missing fraud ($100 average fraud)
cost_fp = 25   # Cost of false alarm ($25 customer service)

# Custom scoring function
def business_cost(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = (fn * cost_fn) + (fp * cost_fp)
    return -total_cost  # Negative because we minimize

# Find threshold minimizing business cost
best_threshold = None
min_cost = float('inf')

for threshold in THRESHOLD_CANDIDATES:
    y_pred = (y_val_proba >= threshold).astype(int)
    cost = business_cost(y_val, y_pred)
    if cost < min_cost:
        min_cost = cost
        best_threshold = threshold
```

**Trade-off Analysis:**
- Lower threshold → Higher recall, lower precision (catch more fraud, more false alarms)
- Higher threshold → Lower recall, higher precision (fewer false alarms, miss more fraud)
- Optimal depends on business context

### 9.4 Precision-Recall Curve Visualization

**Implementation (Notebook 06):**
```python
from src.visualization_utils import plot_precision_recall_curve

plot_precision_recall_curve(
    y_val, 
    y_val_proba,
    title="Precision-Recall Curve: Threshold Optimization",
    save_path="results/creditcard/figures/pr_curve_threshold_optimization.png"
)
```

**Interpretation:**
- Visualize trade-off across all thresholds
- Identify knee of curve (optimal balance)
- Annotate selected threshold on plot

---

## 10. Experiment Tracking & Reproducibility

### 10.1 Random Seed Management (Universal Constant)

**All random seeds set to 42 across entire project:**
```python
import random
import numpy as np
import tensorflow as tf

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Additional TensorFlow determinism
import os
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
```

**Applied to:**
- Data splits (both datasets)
- Weight initialization (all models)
- Dropout masks
- Any stochastic operations

### 10.2 Experiment Logging Format (Dataset-Specific)

**Files:**
- `results/card_transdata/logs/nn_experiments.csv`
- `results/creditcard/logs/nn_experiments.csv`
- `results/experiment_logs/nn_experiments.csv` (consolidated)

**Columns:**
- `experiment_id`: Unique identifier (timestamp)
- `dataset`: "card_transdata" or "creditcard"
- `notebook`: Source notebook (e.g., "03_card_transdata_nn")
- `experiment_type`: "architecture" | "ablation" | "regularization"
- `architecture_id`: ARCH-01 through ARCH-08 (if applicable)
- `ablation_id`: ABL-01 through ABL-05 (if applicable)
- `regularization_id`: REG-01 through REG-08 (if applicable)
- `layers`: Layer configuration (e.g., "[128, 64, 32]")
- `dropout_rate`: Dropout value used
- `l2_reg`: L2 regularization value
- `batch_norm`: Boolean for BatchNorm usage
- `class_weight_strategy`: "balanced", "custom", or "none"
- `learning_rate`: Optimizer learning rate
- `batch_size`: Training batch size
- `epochs_trained`: Actual epochs before early stopping
- `train_loss`: Final training loss
- `val_loss`: Final validation loss
- `val_pr_auc`: Validation Precision-Recall AUC (PRIMARY METRIC)
- `val_f1_fraud`: Validation F1-score (fraud class)
- `val_precision_fraud`: Validation precision (fraud class)
- `val_recall_fraud`: Validation recall (fraud class)
- `val_roc_auc`: Validation ROC AUC (secondary)
- `val_accuracy`: Overall validation accuracy (reference only)
- `confusion_matrix`: JSON string of [[TN, FP], [FN, TP]]
- `training_time_seconds`: Time to train
- `inference_time_ms`: Average inference time per sample
- `random_seed`: Seed used (always 42)
- `date_created`: Timestamp
- `notes`: Additional comments

### 10.3 Model Versioning (Dataset-Aware)

**Naming Convention:**
```
# card_transdata architecture exploration
results/card_transdata/models/neural_networks/arch_01_mlp_32.h5
results/card_transdata/models/neural_networks/arch_03_mlp_128-64-32.h5
results/card_transdata/models/neural_networks/abl_05_full_stack.h5

# creditcard validation and final models
results/creditcard/models/neural_networks/reg_01_no_reg.h5
results/creditcard/models/neural_networks/reg_07_medium_dropout_l2.h5
results/creditcard/models/neural_networks/best_model_final.h5
```

**Metadata Storage (JSON sidecar files):**
```json
{
  "model_file": "best_model_final.h5",
  "dataset": "creditcard",
  "architecture": [128, 64, 32],
  "architecture_id": "ARCH-03",
  "dropout": 0.3,
  "l2_reg": 0.01,
  "batch_norm": true,
  "class_weights": "balanced",
  "validation_metrics": {
    "pr_auc": 0.76,
    "f1_fraud": 0.72,
    "recall_fraud": 0.78,
    "precision_fraud": 0.67
  },
  "trained_on": "2026-01-10",
  "notebook_source": "05_creditcard_nn_training_and_regularization.ipynb",
  "random_seed": 42
}
```

### 10.4 Baseline Model Storage

**Files:**
- `results/card_transdata/models/baselines/logistic_regression.pkl`
- `results/card_transdata/models/baselines/random_forest.pkl`
- `results/creditcard/models/baselines/logistic_regression.pkl`
- `results/creditcard/models/baselines/random_forest.pkl`

**Purpose:** Enable consistent comparison across all experiments.

---

## 11. Ablation Study Methodology (Notebook 03 on card_transdata)

### 11.1 Purpose
Isolate individual component contributions to neural network performance on synthetic data with clean patterns.

### 11.2 Base Architecture
```python
base_layers = [128, 64, 32]
```

**Rationale:** Medium complexity (ARCH-03), representative of typical MLP, confirmed effective in architecture exploration.

### 11.3 Ablation Experiments

| Experiment ID | Description | Dropout | L2 Reg | BatchNorm | Class Weights |
|---------------|-------------|---------|--------|-----------|---------------|
| ABL-01 | Baseline (Minimal) | ✗ | ✗ | ✗ | ✗ |
| ABL-02 | +Dropout | ✓ (0.3) | ✗ | ✗ | ✗ |
| ABL-03 | +L2 Regularization | ✗ | ✓ (0.01) | ✗ | ✗ |
| ABL-04 | +Batch Normalization | ✗ | ✗ | ✓ | ✗ |
| ABL-05 | Full Stack | ✓ (0.3) | ✓ (0.01) | ✓ | ✓ (balanced) |

**Controlled Variables:**
- Same architecture [128, 64, 32]
- Same optimizer (Adam)
- Same learning rate (0.001)
- Same batch size (64)
- Same random seed (42)
- Same data splits (card_transdata)

### 11.4 Analysis Approach

**Metrics to Compare:**
- Validation loss (overfitting indicator)
- Train-val loss gap (generalization gap)
- PR-AUC (overall performance)
- F1-fraud (balanced metric)
- Training time (efficiency)

**Questions to Answer:**
1. Which single technique provides most benefit on synthetic data?
2. Do regularization techniques combine synergistically?
3. Is the full stack justified, or is simpler better?
4. How do insights transfer to real-world data (creditcard)?

### 11.5 Expected Outcomes

**Hypotheses:**
- Class weights will have largest impact (addresses imbalance directly)
- Dropout + L2 will show synergy (complementary regularization)
- BatchNorm may stabilize training but not dramatically improve metrics
- Full stack (ABL-05) should achieve best validation PR-AUC
- Simple models (ABL-01) may overfit on training data

### 11.6 Transfer to Real-World Dataset

**Cross-Validation (Notebook 07):**
- Compare ablation insights from card_transdata with regularization experiments on creditcard
- Identify which findings generalize across data regimes
- Document dataset-specific adjustments needed

---

## 12. Final Test Set Evaluation (Notebook 06 on creditcard)

### 12.1 Test Set Usage Protocol (CRITICAL)

**ONE-TIME EVALUATION RULE:**
1. Test set accessed **exactly once** (Notebook 06 only)
2. Model selection based **only on validation metrics** (PR-AUC from Notebook 05)
3. Threshold optimization performed **only on validation set**
4. No hyperparameter adjustments after seeing test results
5. Test results reported regardless of outcome (honest evaluation)

**Rationale:** Prevents data leakage and overfitting to test set.

### 12.2 Best Model Selection (Based on creditcard Validation)

**Criteria (in order):**
1. **Highest validation PR-AUC** (from Notebook 05 regularization experiments REG-01 to REG-08)
2. If tied, highest validation F1-fraud
3. If still tied, lowest training time

**Architecture Source:**
- Base architecture transferred from card_transdata (best of ARCH-01 to ARCH-08)
- Regularization tuned on creditcard (REG-01 to REG-08)

**Not based on:**
- Test set performance (never seen before selection)
- Architecture complexity preferences
- Training speed alone

### 12.3 Test Evaluation Steps (Notebook 06)

```python
# 1. Load best model (determined from Notebook 05 validation PR-AUC)
best_model = load_model('results/creditcard/models/neural_networks/best_model_final.h5')

# 2. Apply optimal threshold (from validation optimization)
optimal_threshold = 0.30  # Example value from validation tuning

# 3. Predict on test set (ONCE AND ONLY ONCE)
y_test_proba = best_model.predict(X_test_scaled)
y_test_pred = (y_test_proba >= optimal_threshold).astype(int)

# 4. Compute all metrics
from src.evaluation_metrics import compute_classification_metrics

test_metrics = compute_classification_metrics(
    y_test, 
    y_test_pred, 
    y_test_proba
)

# 5. Save and report (NO CHANGES ALLOWED AFTER THIS POINT)
test_metrics_df = pd.DataFrame([test_metrics])
test_metrics_df.to_csv(
    'results/creditcard/tables/final_test_evaluation.csv',
    index=False
)
```

### 12.4 Error Analysis (Post-Test, No Retraining)

**Analyze without retraining:**

**False Positive Analysis:**
- Which legitimate transactions were flagged?
- Feature patterns in false alarms
- Customer segments most affected
- Business implications (customer service load)

**False Negative Analysis:**
- Which fraud patterns were missed?
- Feature patterns in missed fraud
- Financial loss implications
- Model limitations

**Feature Importance:**
- SHAP values or permutation importance
- Identify most influential features
- Compare with baseline model interpretations

**Explicitly forbidden:**
- Retraining model after seeing test errors
- Adjusting threshold based on test results
- Selecting different model based on test performance
- Using test insights for any model modifications

### 12.5 Business Impact Assessment (Notebook 06)

**Cost Analysis:**
```python
# Extract confusion matrix components
tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()

# Business costs
cost_per_fn = 100  # Average fraud amount ($100)
cost_per_fp = 25   # Customer service cost ($25)

# Calculate total costs
total_fn_cost = fn * cost_per_fn
total_fp_cost = fp * cost_per_fp
total_cost = total_fn_cost + total_fp_cost

# Baseline comparison (no model)
baseline_cost = len(y_test[y_test == 1]) * cost_per_fn  # Miss all fraud

# Model value
cost_saved = baseline_cost - total_cost
roi = (cost_saved / total_cost) * 100

print(f"Model saves ${cost_saved:,.2f} compared to no detection")
print(f"ROI: {roi:.1f}%")
```

---

## 13. Comparison with Baseline Models (Both Datasets)

### 13.1 Baseline Model Purpose

**Why include classical ML on both datasets?**
- Establish performance floor for each dataset
- Justify neural network complexity investment
- Provide interpretability comparison
- Demonstrate when NNs provide value vs when simpler models suffice
- Enable cross-dataset baseline comparison

### 13.2 Baseline Models (Notebooks 02 and 04)

#### Logistic Regression
```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(
    class_weight='balanced',
    random_state=42,
    max_iter=1000
)
lr.fit(X_train_scaled, y_train)
```

**Expected:**
- Linear baseline (fast training, interpretable)
- card_transdata: Likely good performance (linear patterns)
- creditcard: May struggle with PCA features (non-linear patterns)

#### Random Forest
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train)
```

**Expected:**
- Non-linear baseline (robust, ensemble learning)
- card_transdata: Near-perfect performance (clean synthetic data)
- creditcard: Strong competitor to neural networks
- **Key Question (Notebook 07):** When does NN complexity outweigh RF simplicity?

#### Dummy Classifier
```python
from sklearn.dummy import DummyClassifier

dummy = DummyClassifier(strategy='stratified', random_state=42)
dummy.fit(X_train_scaled, y_train)
```

**Expected:** Worst-case performance, statistical baseline (~0.5 PR-AUC).

### 13.3 Fair Comparison Requirements

**Ensure consistency across datasets:**
- Same data splits (70/15/15, random_seed=42)
- Same preprocessing (StandardScaler fit on training only)
- Same evaluation metrics (PR-AUC primary, F1, Precision, Recall)
- Same class weight strategy ('balanced')
- Documented training times for efficiency comparison

### 13.4 Baseline Performance Targets (config.py)

```python
BASELINE_PERFORMANCE_TARGETS = {
    'card_transdata': {
        'logistic_regression': {'pr_auc': 0.85, 'f1_fraud': 0.80},
        'random_forest': {'pr_auc': 0.95, 'f1_fraud': 0.92},
        'neural_network_target': {'pr_auc': 0.90, 'f1_fraud': 0.85}
    },
    'creditcard': {
        'logistic_regression': {'pr_auc': 0.65, 'f1_fraud': 0.60},
        'random_forest': {'pr_auc': 0.80, 'f1_fraud': 0.75},
        'neural_network_target': {'pr_auc': 0.75, 'f1_fraud': 0.70}
    }
}
```

**Interpretation:**
- card_transdata: Random Forest is very strong (near-ceiling performance)
- creditcard: Neural Networks have more room to demonstrate value
- Target: Neural Networks should exceed Logistic Regression, approach or exceed Random Forest

### 13.5 Baseline vs Neural Network Analysis (Notebook 07)

**Comparative Questions:**
1. Does NN architecture complexity improve over Random Forest?
2. On which dataset is the NN advantage most pronounced?
3. What is the training time vs performance trade-off?
4. When should practitioners use RF vs NN?
5. Do NNs generalize better across datasets?

---

## 14. Cross-Dataset Analysis (Notebook 07)

### 14.1 Research Questions

**Primary Questions:**
1. Do neural network architectural choices generalize across datasets?
2. Which regularization strategies work for both synthetic and real-world data?
3. How does performance scale with imbalance severity (1:124 vs 1:577)?
4. What design principles are transferable to production fraud detection?

### 14.2 Comparative Analyses

#### Architecture Generalization
**Method:**
- Compare ARCH-01 to ARCH-08 rankings on card_transdata
- Compare same architectures applied to creditcard
- Identify consistent top performers

**Expected Finding:** Medium-depth architectures (ARCH-03, ARCH-04) likely optimal for both.

#### Regularization Effectiveness
**Method:**
- Compare ABL-01 to ABL-05 results on card_transdata
- Compare REG-01 to REG-08 results on creditcard
- Identify which regularization techniques are universally beneficial

**Expected Finding:** Class weights essential for both; Dropout + L2 synergy may vary.

#### Overfitting Analysis
**Method:**
- Calculate train-val loss gaps for all experiments
- Compare overfitting severity across datasets
- Relate overfitting to imbalance ratio

**Hypothesis:** creditcard (1:577 imbalance) shows more overfitting than card_transdata (1:124).

#### Baseline Comparison
**Method:**
- Compare NN improvement over Random Forest on both datasets
- Quantify when NN complexity is justified

**Decision Rule:**
```python
nn_advantage = (nn_pr_auc - rf_pr_auc) / rf_pr_auc * 100

if nn_advantage > 5:
    recommendation = "Use Neural Network"
elif nn_advantage > 2:
    recommendation = "Consider Neural Network (marginal benefit)"
else:
    recommendation = "Use Random Forest (simpler, sufficient)"
```

### 14.3 Key Performance Indicators (KPIs)

**Metrics for Cross-Dataset Comparison:**
- Validation PR-AUC (primary)
- Train-val PR-AUC gap (generalization)
- F1-fraud score (balanced metric)
- Training time efficiency (seconds per epoch)
- Inference speed (ms per prediction)

### 14.4 Transferable Design Principles (Synthesis)

**Expected Insights (to be validated):**

1. **Architecture Depth:** Medium-depth MLPs (3-4 layers) optimal for both regimes
2. **Class Weights:** Essential for both datasets regardless of imbalance severity
3. **Dropout:** More critical for extreme imbalance (creditcard)
4. **L2 Regularization:** Moderate values (0.01) effective universally
5. **Batch Normalization:** Helpful for training stability, not primary performance driver
6. **Threshold Optimization:** Critical for both; optimal threshold << 0.5
7. **Early Stopping:** Prevents overfitting on both, more critical for creditcard

### 14.5 Dataset Characteristics Table

**Comparison Matrix (Notebook 07 Section 3):**

| Characteristic | card_transdata | creditcard |
|----------------|----------------|------------|
| **Data Source** | Synthetic | Real-world (ULB) |
| **Samples** | ~1,000,000 | ~284,000 |
| **Features** | 7 interpretable | 30 (28 PCA + Time + Amount) |
| **Imbalance Ratio** | 1:124 | 1:577 |
| **Best Baseline** | Random Forest (PR-AUC ≈ 0.95) | Random Forest (PR-AUC ≈ 0.80) |
| **NN Target** | PR-AUC ≥ 0.90 | PR-AUC ≥ 0.75 |
| **Primary Challenge** | Architecture exploration | Extreme imbalance + PCA features |
| **Production Proxy** | Clean patterns (ideal case) | Noisy patterns (realistic case) |

### 14.6 Recommendations Synthesis (Notebook 07 Final Section)

**Production Deployment Guidelines:**

**When to Use Neural Networks:**
- Extreme class imbalance (>1:500)
- Non-linear PCA-transformed features
- Real-time inference not critical (NN inference slower than RF)
- Computational resources available for training

**When to Use Random Forest:**
- Moderate imbalance (1:100 to 1:200)
- Interpretable features (business stakeholder requirement)
- Fast inference required (<1ms per prediction)
- Limited training time or resources

**When to Use Logistic Regression:**
- Need maximum interpretability (regulatory requirement)
- Very fast inference critical (<0.1ms)
- Linear patterns sufficient (validate first with RF baseline)

**Cross-Dataset Validation Protocol:**
- Always test on multiple datasets before production
- Use synthetic data for architecture exploration
- Validate on production-like data before deployment
- Monitor for distribution drift post-deployment

---

## 15. Reproducibility Checklist

### Before Starting Experiments
- [ ] Random seed set to 42 in all environments (Python, NumPy, TensorFlow)
- [ ] Data splits created and saved for both datasets separately
- [ ] StandardScaler fitted and saved per dataset
- [ ] Experiment logging CSVs initialized for both datasets
- [ ] Documentation files reviewed (notebook_implementation_guide.md, dual_dataset_methodology.md)
- [ ] config.py configured with dataset-specific paths
- [ ] Baseline performance targets defined

### Phase 1: Dual-Dataset Overview (Notebook 01)
- [ ] EDA completed for both datasets
- [ ] Class distributions visualized and documented
- [ ] Feature distributions compared
- [ ] Statistical summary tables generated

### Phase 2: card_transdata Exploration (Notebooks 02-03)
- [ ] Preprocessing pipeline implemented (Notebook 02)
- [ ] Baseline models trained and logged (Logistic Regression, Random Forest)
- [ ] 8 architecture variants tested (ARCH-01 to ARCH-08)
- [ ] 5 ablation experiments completed (ABL-01 to ABL-05)
- [ ] Best architecture identified by validation PR-AUC
- [ ] All experiments logged to card_transdata/logs/nn_experiments.csv
- [ ] Models saved to card_transdata/models/neural_networks/
- [ ] Training curves plotted and saved
- [ ] Validation metrics recorded (test set never accessed)

### Phase 3: creditcard Validation (Notebooks 04-05)
- [ ] Preprocessing pipeline implemented (Notebook 04)
- [ ] Baseline models trained and logged
- [ ] Best architecture from card_transdata transferred
- [ ] 8 regularization experiments completed (REG-01 to REG-08)
- [ ] Best model selected based on validation PR-AUC
- [ ] All experiments logged to creditcard/logs/nn_experiments.csv
- [ ] Models saved to creditcard/models/neural_networks/
- [ ] Training curves plotted and saved
- [ ] Validation metrics recorded (test set isolated)

### Phase 4: Threshold Optimization & Test Evaluation (Notebook 06)
- [ ] Best model from Notebook 05 loaded
- [ ] Threshold optimization performed on validation set only
- [ ] Optimal threshold documented (F1-maximizing or cost-minimizing)
- [ ] Precision-recall curve visualized
- [ ] ONE-TIME test evaluation performed
- [ ] Test predictions made exactly once
- [ ] All test metrics computed and saved to creditcard/tables/final_test_evaluation.csv
- [ ] Confusion matrix analyzed
- [ ] Error analysis conducted (FP and FN patterns)
- [ ] Business cost analysis completed
- [ ] No model modifications after test evaluation

### Phase 5: Cross-Dataset Synthesis (Notebook 07)
- [ ] Experiment results loaded from both datasets
- [ ] Baseline results loaded from both datasets
- [ ] Final test results loaded from creditcard
- [ ] Dataset characteristics compared in table
- [ ] Cross-dataset performance comparison visualized
- [ ] Architecture generalization analysis completed
- [ ] Regularization effectiveness compared
- [ ] Overfitting analysis (train-val gaps) across datasets
- [ ] Key transferable design principles identified (7 principles minimum)
- [ ] Limitations and future work documented
- [ ] Production deployment recommendations finalized
- [ ] Project summary and conclusion written

### Final Validation
- [ ] All 7 notebooks executable end-to-end
- [ ] All CSV logs contain complete experiment history
- [ ] All models saved with descriptive names
- [ ] All figures saved to appropriate directories
- [ ] README.md updated with final results
- [ ] PROJECT_STATUS.md reflects completion
- [ ] No data leakage (verified via checklist in docs/)
- [ ] Random seed consistency verified (all experiments use 42)
- [ ] Reproducibility tested (re-run notebooks with fresh kernel)

---

## 16. Conclusion

This dual-dataset methodology ensures that neural network experiments are:

1. **Reproducible:** Fixed seeds, saved splits, documented configs across both datasets
2. **Valid:** No data leakage, proper train/val/test separation, ONE-TIME test evaluation
3. **Rigorous:** Systematic exploration (8 architectures + 5 ablations + 8 regularization experiments), cross-dataset validation
4. **Generalizable:** Transferable design principles identified through comparative analysis
5. **Practical:** Business-relevant metrics, threshold optimization, deployment guidelines
6. **Transparent:** Complete logging, error analysis, honest reporting, limitations documented

**Key Innovation:** By systematically exploring neural network designs on synthetic data (clean environment) and validating on real-world data (production environment), we identify architectural and regularization strategies that generalize across fraud detection regimes. This approach provides practitioners with evidence-based design principles rather than dataset-specific heuristics.

**Final Deliverable:** A reproducible pipeline demonstrating:
- When neural networks provide value over classical ML (Random Forest baseline)
- Which architectural choices generalize across data regimes
- How to handle varying degrees of class imbalance (1:124 vs 1:577)
- Production-ready evaluation protocols preventing data leakage
- Business-contextualized recommendations for model deployment

The result is a neural network model whose performance can be trusted and whose training procedure can be replicated across different fraud detection contexts.
