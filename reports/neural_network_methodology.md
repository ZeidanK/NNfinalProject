# Neural Network Methodology Report

## Executive Summary

This report documents the comprehensive methodology for training and evaluating neural networks on imbalanced credit card fraud detection data. The approach emphasizes **reproducibility, data leakage prevention, and neural network-specific techniques** for handling extreme class imbalance.

---

## 1. Dataset Overview

### 1.1 Source & Description
- **Dataset:** Credit Card Fraud Detection
- **Source:** https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud
- **Size:** ~50MB tabular data
- **License:** See Kaggle dataset page

### 1.2 Features
**Input Features (7):**
1. `distance_from_home` - Distance from home location (continuous)
2. `distance_from_last_transaction` - Distance from previous transaction (continuous)
3. `ratio_to_median_purchase_price` - Purchase price relative to median (continuous)
4. `repeat_retailer` - Whether retailer was used before (binary)
5. `used_chip` - Whether chip was used for transaction (binary)
6. `used_pin_number` - Whether PIN was entered (binary)
7. `online_order` - Whether transaction was online (binary)

**Target Variable:**
- `fraud` - Binary indicator (0 = legitimate, 1 = fraud)

### 1.3 Class Distribution
- **Legitimate Transactions:** ~99.2%
- **Fraudulent Transactions:** ~0.8%
- **Imbalance Ratio:** Approximately 124:1

**Implication:** Extreme class imbalance requires specialized neural network training strategies.

---

## 2. Data Splitting Strategy

### 2.1 Rationale for 70/15/15 Split
- **Training Set (70%):** Sufficient data for neural network learning
- **Validation Set (15%):** Hyperparameter tuning and early stopping
- **Test Set (15%):** Final unbiased performance evaluation

### 2.2 Stratified Sampling
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

**Verification:** Ensure class distribution preserved across all splits.

### 2.3 Split Index Preservation
**Critical for reproducibility:**
```python
np.save('results/train_indices.npy', train_indices)
np.save('results/val_indices.npy', val_indices)
np.save('results/test_indices.npy', test_indices)
```

**Benefit:** All experiments use identical data splits, enabling fair comparison.

---

## 3. Feature Preprocessing

### 3.1 StandardScaler for Neural Network Compatibility

**Why Normalization is Essential:**
- Neural networks sensitive to feature scales
- Gradient descent converges faster with normalized inputs
- Prevents large-scale features from dominating
- Stabilizes weight initialization

**Implementation:**
```python
from sklearn.preprocessing import StandardScaler

# Fit ONLY on training data
scaler = StandardScaler()
scaler.fit(X_train)

# Transform all splits
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Save for reuse
import joblib
joblib.dump(scaler, 'results/fitted_scaler.pkl')
```

**Critical Rule:** Scaler statistics (mean, std) computed ONLY from training data.

### 3.2 Binary Features
- No additional normalization needed (already 0/1)
- Included in StandardScaler for consistency
- Scale compatible with ReLU activations

### 3.3 Data Leakage Prevention
**Checklist:**
- ✅ Scaler fit on training data only
- ✅ Same scaler applied to val/test
- ✅ No target variable used in preprocessing
- ✅ Test set isolated until final evaluation

---

## 4. Neural Network Architecture Design

### 4.1 Multi-Layer Perceptron (MLP) Structure

**General Architecture:**
```
Input (7 features) 
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

### 4.2 Architecture Variants Tested

#### Shallow MLPs
- `[32]` - Single hidden layer, 32 units
- `[64, 32]` - Two hidden layers, decreasing width

#### Medium MLPs (Expected Optimal)
- `[128, 64, 32]` - Three hidden layers
- `[256, 128, 64, 32]` - Four hidden layers

#### Deep MLPs
- `[512, 256, 128, 64, 32, 16]` - Six hidden layers

#### Width vs Depth Comparison
- `[1024]` - Very wide, single layer
- `[64, 64, 64, 64]` - Narrow, four layers

### 4.3 Activation Functions

**Hidden Layers:** ReLU (Rectified Linear Unit)
- Prevents vanishing gradient
- Computationally efficient
- Non-linear transformation

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

**Rates Tested:** 0.0, 0.2, 0.3, 0.4, 0.5

**Expected Outcome:** Higher dropout for deeper networks

### 5.2 L2 Regularization (Weight Decay)

**Purpose:** Penalize large weights

**Implementation:**
```python
from tensorflow.keras.regularizers import l2

model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
```

**Values Tested:** 0.0, 0.001, 0.01, 0.1

**Expected Outcome:** Moderate L2 (0.01) most effective

### 5.3 Batch Normalization

**Purpose:** Normalize layer inputs

**Implementation:**
```python
from tensorflow.keras.layers import BatchNormalization

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
```

**Placement:** After activation or before (experiment both)

**Benefits:**
- Stabilizes training
- Enables higher learning rates
- Acts as regularization

---

## 6. Class Imbalance Handling

### 6.1 Class Weight Calculation

**Balanced Strategy (Primary):**
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

**Effect:** Fraud misclassifications contribute more to loss function.

### 6.2 Custom Class Weights

**Formula:**
```python
weight_fraud = n_legitimate / n_fraud
weight_legitimate = 1.0

# Example: If 99,200 legitimate and 800 fraud
weight_fraud = 99200 / 800 = 124
```

**Interpretation:** Fraud errors are 124× more costly.

### 6.3 Focal Loss (Advanced)

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

**When to use:** If weighted BCE insufficient for imbalance.

---

## 7. Training Configuration

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
    class_weight=class_weight_dict,  # Essential!
    ...
)
```

### 7.3 Early Stopping

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
- Prevents overfitting
- Saves training time
- Returns best validation model

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

### 7.5 Model Checkpointing

**Configuration:**
```python
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    'results/models/best_model.h5',
    monitor='val_pr_auc',  # Precision-Recall AUC
    save_best_only=True,
    mode='max',
    verbose=1
)
```

**Critical:** Save model based on validation metric, not test.

---

## 8. Evaluation Metrics

### 8.1 Why Accuracy is Insufficient

**Problem with Accuracy:**
```python
# Dummy classifier predicting all legitimate
predictions = np.zeros(len(y_test))

accuracy = accuracy_score(y_test, predictions)
# Result: ~99.2% accuracy!

fraud_recall = recall_score(y_test, predictions)
# Result: 0% recall (catches zero fraud)
```

**Conclusion:** Accuracy is misleading for imbalanced data.

### 8.2 Primary Metrics (Fraud Class Focus)

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

#### Precision-Recall AUC (PR-AUC)
**Best metric for this problem:**
- Threshold-independent
- Summarizes precision-recall trade-off across all thresholds
- More informative than ROC-AUC for imbalanced data
- Directly relevant to business decisions

#### ROC-AUC
**Secondary metric:**
- Useful for comparison with baselines
- Can be misleadingly high for imbalanced data
- Less directly interpretable than PR-AUC

### 8.3 Confusion Matrix Analysis

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

**Cost Analysis:**
- FN cost typically >> FP cost
- Adjust threshold based on business priorities

---

## 9. Threshold Optimization

### 9.1 Why Default 0.5 is Suboptimal

Neural networks output probabilities [0, 1]. Default threshold:
```python
prediction = 1 if probability >= 0.5 else 0
```

**Problem:** This assumes:
- Equal class distribution
- Equal misclassification costs

Both assumptions violated in fraud detection!

### 9.2 Validation-Based Threshold Selection

**Process:**
```python
from sklearn.metrics import precision_recall_curve

# Get probabilities on validation set
y_val_proba = model.predict(X_val_scaled)

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

### 9.3 Business-Driven Threshold Selection

**Cost-Sensitive Approach:**
```python
cost_fn = 100  # Cost of missing fraud
cost_fp = 1    # Cost of false alarm

# Custom scoring function
def business_cost(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = (fn * cost_fn) + (fp * cost_fp)
    return -total_cost  # Negative because we minimize

# Find threshold minimizing business cost
```

---

## 10. Experiment Tracking & Reproducibility

### 10.1 Random Seed Management

**All random seeds set to 42:**
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

### 10.2 Experiment Logging Format

**File:** `results/experiment_logs/nn_experiments.csv`

**Columns:**
- `experiment_id`: Unique identifier (timestamp)
- `notebook`: Source notebook
- `model_type`: Architecture description
- `layers`: Layer configuration (e.g., "[128, 64, 32]")
- `dropout_rate`: Dropout value used
- `l2_reg`: L2 regularization value
- `batch_norm`: Boolean for BatchNorm usage
- `class_weight_strategy`: "balanced", "custom", or "none"
- `learning_rate`: Optimizer learning rate
- `batch_size`: Training batch size
- `epochs_trained`: Actual epochs before early stopping
- `precision_fraud`: Precision on fraud class
- `recall_fraud`: Recall on fraud class
- `f1_fraud`: F1-score on fraud class
- `pr_auc`: Precision-Recall AUC
- `roc_auc`: ROC AUC
- `accuracy`: Overall accuracy
- `confusion_matrix`: JSON string of [[TN, FP], [FN, TP]]
- `training_time_seconds`: Time to train
- `inference_time_seconds`: Time to predict on val set
- `val_loss`: Final validation loss
- `train_loss`: Final training loss
- `random_seed`: Seed used (always 42)
- `date_created`: Timestamp
- `notes`: Additional comments

### 10.3 Model Versioning

**Naming Convention:**
```
models/best_nn_models/mlp_{architecture}_{timestamp}.h5

Example:
mlp_128-64-32_dropout0.3_20260109_143022.h5
```

**Metadata Storage:**
```json
{
  "model_file": "mlp_128-64-32_dropout0.3_20260109_143022.h5",
  "architecture": [128, 64, 32],
  "dropout": 0.3,
  "l2_reg": 0.01,
  "batch_norm": true,
  "class_weights": "balanced",
  "validation_metrics": {
    "pr_auc": 0.82,
    "f1_fraud": 0.75,
    "recall_fraud": 0.78,
    "precision_fraud": 0.72
  }
}
```

---

## 11. Ablation Study Methodology

### 11.1 Purpose
Isolate individual component contributions to neural network performance.

### 11.2 Base Architecture
```python
base_layers = [128, 64, 32]
```

**Rationale:** Medium complexity, representative of typical MLP.

### 11.3 Ablation Experiments

| Experiment | Dropout | L2 Reg | BatchNorm | Class Weights |
|------------|---------|--------|-----------|---------------|
| Baseline   | ✗       | ✗      | ✗         | ✗             |
| +Dropout   | ✓ (0.3) | ✗      | ✗         | ✗             |
| +L2        | ✗       | ✓(0.01)| ✗         | ✗             |
| +BatchNorm | ✗       | ✗      | ✓         | ✗             |
| +Dropout+L2| ✓ (0.3) | ✓(0.01)| ✗         | ✗             |
| Full       | ✓ (0.3) | ✓(0.01)| ✓         | ✓ (balanced)  |

**Controlled Variables:**
- Same architecture
- Same optimizer (Adam)
- Same learning rate (0.001)
- Same batch size (64)
- Same random seed (42)
- Same data splits

### 11.4 Analysis Approach

**Metrics to Compare:**
- Validation loss (overfitting indicator)
- Train-val loss gap (generalization gap)
- PR-AUC (overall performance)
- F1-fraud (balanced metric)
- Training time

**Questions to Answer:**
1. Which single technique provides most benefit?
2. Do regularization techniques combine synergistically?
3. Is the full model justified, or is simpler better?

---

## 12. Final Test Set Evaluation

### 12.1 Test Set Usage Protocol

**CRITICAL RULES:**
1. Test set used **exactly once**
2. Model selection based **only on validation metrics**
3. No hyperparameter adjustments after seeing test results
4. Test results reported regardless of outcome

### 12.2 Best Model Selection

**Criteria (in order):**
1. Highest PR-AUC on validation set
2. If tied, highest F1-fraud
3. If still tied, lowest training time

**Not based on:**
- Test set performance (never seen before selection)
- Architecture complexity preferences
- Training speed alone

### 12.3 Test Evaluation Steps

```python
# 1. Load best model (determined from validation)
best_model = load_model('results/models/best_nn_model.h5')

# 2. Apply optimal threshold (from validation)
optimal_threshold = 0.35  # Example value from validation optimization

# 3. Predict on test set (ONCE)
y_test_proba = best_model.predict(X_test_scaled)
y_test_pred = (y_test_proba >= optimal_threshold).astype(int)

# 4. Compute all metrics
test_metrics = {
    'precision_fraud': precision_score(y_test, y_test_pred, pos_label=1),
    'recall_fraud': recall_score(y_test, y_test_pred, pos_label=1),
    'f1_fraud': f1_score(y_test, y_test_pred, pos_label=1),
    'pr_auc': average_precision_score(y_test, y_test_proba),
    'roc_auc': roc_auc_score(y_test, y_test_proba),
    'confusion_matrix': confusion_matrix(y_test, y_test_pred)
}

# 5. Save and report (no changes allowed)
```

### 12.4 Error Analysis (Post-Test)

**Without retraining, analyze:**
- False positives: Why did NN flag legitimate transactions?
- False negatives: What fraud patterns did NN miss?
- Feature patterns in errors
- Business implications

**Explicitly forbidden:**
- Retraining model after seeing errors
- Adjusting threshold based on test results
- Selecting different model based on test performance

---

## 13. Comparison with Baseline Models

### 13.1 Baseline Model Purpose

**Why include classical ML?**
- Establish performance floor
- Justify neural network complexity
- Provide interpretability comparison
- Demonstrate NN advantages

### 13.2 Baseline Models

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

**Expected:** Linear baseline, fast training, interpretable.

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

**Expected:** Non-linear baseline, may be competitive.

#### Dummy Classifier
```python
from sklearn.dummy import DummyClassifier

dummy = DummyClassifier(strategy='stratified', random_state=42)
dummy.fit(X_train_scaled, y_train)
```

**Expected:** Worst-case performance, statistical baseline.

### 13.3 Fair Comparison Requirements

**Ensure:**
- Same data splits
- Same preprocessing (StandardScaler)
- Same evaluation metrics
- Same class weight strategy
- Documented training times

---

## 14. Reproducibility Checklist

### Before Starting Experiments
- [ ] Random seed set to 42 in all environments
- [ ] Data splits created and saved
- [ ] StandardScaler fitted and saved
- [ ] Experiment logging CSV initialized
- [ ] Documentation files prepared

### During Experiments
- [ ] Each experiment logged immediately after completion
- [ ] Model files saved with descriptive names
- [ ] Training curves plotted and saved
- [ ] Validation metrics recorded
- [ ] Test set never accessed

### After All Experiments (Before Test Evaluation)
- [ ] Best model selected based on validation PR-AUC
- [ ] Optimal threshold determined from validation set
- [ ] All experiment results compiled
- [ ] Ablation study completed
- [ ] Documentation updated

### Final Test Evaluation
- [ ] Best model loaded
- [ ] Test set predictions made exactly once
- [ ] All metrics computed and saved
- [ ] Error analysis conducted
- [ ] Final report generated

---

## 15. Conclusion

This methodology ensures that neural network experiments are:

1. **Reproducible:** Fixed seeds, saved splits, documented configs
2. **Valid:** No data leakage, proper train/val/test separation
3. **Rigorous:** Systematic exploration, ablation studies, statistical tests
4. **Practical:** Business-relevant metrics, threshold optimization, deployment considerations
5. **Transparent:** Complete logging, error analysis, honest reporting

The result is a neural network model whose performance can be trusted and whose training procedure can be replicated.
