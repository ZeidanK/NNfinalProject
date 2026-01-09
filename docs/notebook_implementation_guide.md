# Notebook Implementation Guide

## Overview
This guide provides detailed instructions for implementing each Jupyter notebook in the Neural Networks fraud detection project. All notebooks follow a consistent structure and use the utility modules in `src/`.

---

## Common Setup for All Notebooks

### Standard Imports
```python
# Standard libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
from pathlib import Path

# Configure paths
import sys
sys.path.append('../')
import config

# Set random seeds
config.set_random_seeds()
config.ensure_directories()

# Visualization settings
plt.style.use('seaborn-v0_8-darkgrid')
%matplotlib inline
```

### Load Preprocessed Data (Notebooks 3-8)
```python
# Load split indices
train_idx = np.load('../results/train_indices.npy')
val_idx = np.load('../results/val_indices.npy')
test_idx = np.load('../results/test_indices.npy')

# Load data
df = pd.read_csv(config.DATASET_PATH)
X = df[config.FEATURE_COLUMNS].values
y = df[config.TARGET_COLUMN].values

# Apply splits
X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

# Load and apply scaler
import joblib
scaler = joblib.load(config.FITTED_SCALER_PATH)
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
```

---

## Notebook 01: Data Analysis and NN Motivation

### Purpose
Compact EDA to motivate neural network design decisions.

### Implementation Steps

1. **Load Dataset**
```python
df = pd.read_csv('../data/card_transdata.csv')
print(f"Dataset shape: {df.shape}")
print(f"Features: {df.columns.tolist()}")
```

2. **Class Distribution Analysis**
```python
class_counts = df['fraud'].value_counts()
class_percentages = df['fraud'].value_counts(normalize=True) * 100

print(f"Legitimate: {class_counts[0]:,} ({class_percentages[0]:.2f}%)")
print(f"Fraud: {class_counts[1]:,} ({class_percentages[1]:.2f}%)")
print(f"Imbalance Ratio: {class_counts[0] / class_counts[1]:.1f}:1")

# Visualize
from src.visualization_utils import plot_class_distribution
plot_class_distribution(
    df['fraud'].values,
    save_path='../results/figures/class_imbalance_severity.png'
)
```

3. **Feature Distributions**
```python
# Create subplots for all features
fig, axes = plt.subplots(4, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, feature in enumerate(config.FEATURE_COLUMNS):
    axes[idx].hist(df[feature], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    axes[idx].set_title(f'{feature}', fontweight='bold')
    axes[idx].set_xlabel('Value')
    axes[idx].set_ylabel('Frequency')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/figures/feature_distributions.png', dpi=300)
plt.show()
```

4. **Correlation Analysis**
```python
corr_matrix = df[config.FEATURE_COLUMNS + [config.TARGET_COLUMN]].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Matrix', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('../results/figures/correlation_matrix.png', dpi=300)
plt.show()
```

5. **Statistical Tests (Compact)**
```python
from scipy import stats

statistical_results = []

# Continuous features: Mann-Whitney U test (non-parametric)
continuous_features = ['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price']

for feature in continuous_features:
    legit = df[df['fraud'] == 0][feature]
    fraud = df[df['fraud'] == 1][feature]
    
    # Mann-Whitney U test
    statistic, p_value = stats.mannwhitneyu(legit, fraud, alternative='two-sided')
    
    # Effect size (rank biserial correlation)
    n1, n2 = len(legit), len(fraud)
    effect_size = 1 - (2*statistic) / (n1 * n2)
    
    statistical_results.append({
        'feature': feature,
        'test': 'Mann-Whitney U',
        'p_value': p_value,
        'effect_size': abs(effect_size),
        'significant': 'Yes' if p_value < 0.05 else 'No'
    })

# Binary features: Chi-square test
binary_features = ['repeat_retailer', 'used_chip', 'used_pin_number', 'online_order']

for feature in binary_features:
    contingency_table = pd.crosstab(df[feature], df['fraud'])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    # Cramér's V
    n = contingency_table.sum().sum()
    cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
    
    statistical_results.append({
        'feature': feature,
        'test': 'Chi-Square',
        'p_value': p_value,
        'effect_size': cramers_v,
        'significant': 'Yes' if p_value < 0.05 else 'No'
    })

# Save results
stats_df = pd.DataFrame(statistical_results)
stats_df = stats_df.sort_values('effect_size', ascending=False)
stats_df.to_csv('../results/tables/statistical_summary_table.csv', index=False)
print(stats_df.to_string())
```

6. **Neural Network Design Motivation**
```markdown
## Why Multi-Layer Perceptrons (MLPs) for This Problem?

### Data Characteristics
- **Tabular data**: 7 features (mixed continuous + binary)
- **Feature scales**: Vary significantly (requires normalization)
- **Non-linear patterns**: Fraud likely involves complex feature interactions
- **Class imbalance**: 99.2% vs 0.8% requires specialized handling

### Why Neural Networks?
1. **Non-Linear Decision Boundaries**: MLPs can learn complex, non-linear relationships between features that linear models miss.
2. **Representation Learning**: Hidden layers automatically learn fraud-indicative feature combinations.
3. **Flexible Class Weighting**: Neural networks support multiple imbalance strategies (class weights, focal loss, threshold tuning).
4. **Scalability**: Fast inference after training, suitable for real-time fraud detection.

### Architectural Considerations
- **Depth**: Multiple hidden layers for hierarchical pattern learning
- **Width**: Sufficient neurons to capture feature interactions
- **Regularization**: Dropout, L2, BatchNorm to prevent overfitting
- **Activation**: ReLU for non-linearity, Sigmoid output for probability

### Next Steps
In subsequent notebooks, we will systematically explore:
- Architecture variations (shallow vs deep vs wide)
- Regularization strategies
- Class imbalance handling techniques
- Threshold optimization for production deployment
```

---

## Notebook 02: Preprocessing and Baseline Comparison

### Purpose
Data preparation and classical ML baseline establishment.

### Implementation Steps

1. **Data Splitting**
```python
from sklearn.model_selection import train_test_split

# First split: separate test set (15%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=config.TEST_SIZE,
    stratify=y,
    random_state=config.RANDOM_SEED
)

# Second split: separate train (70%) and validation (15%)
val_size_adjusted = config.VAL_SIZE / (1 - config.TEST_SIZE)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=val_size_adjusted,
    stratify=y_temp,
    random_state=config.RANDOM_SEED
)

# Save indices
train_idx = X_train.index if hasattr(X_train, 'index') else np.arange(len(X_train))
val_idx = X_val.index if hasattr(X_val, 'index') else np.arange(len(X_val))
test_idx = X_test.index if hasattr(X_test, 'index') else np.arange(len(X_test))

np.save('../results/train_indices.npy', train_idx)
np.save('../results/val_indices.npy', val_idx)
np.save('../results/test_indices.npy', test_idx)

print(f"Train size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Val size: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
print(f"Test size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
```

2. **Feature Scaling**
```python
from sklearn.preprocessing import StandardScaler
import joblib

# Fit scaler ONLY on training data
scaler = StandardScaler()
scaler.fit(X_train)

# Transform all splits
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, config.FITTED_SCALER_PATH)
print("Scaler fitted and saved.")
```

3. **Baseline Model: Logistic Regression**
```python
from sklearn.linear_model import LogisticRegression
from src.evaluation_metrics import compute_fraud_metrics, print_classification_summary

lr = LogisticRegression(
    class_weight='balanced',
    random_state=config.RANDOM_SEED,
    max_iter=1000
)

start_time = time.time()
lr.fit(X_train_scaled, y_train)
training_time = time.time() - start_time

# Evaluate on validation set
y_val_pred = lr.predict(X_val_scaled)
y_val_proba = lr.predict_proba(X_val_scaled)[:, 1]

metrics = compute_fraud_metrics(y_val, y_val_pred, y_val_proba)
print_classification_summary(y_val, y_val_pred, y_val_proba, "Validation Set - Logistic Regression")
```

4. **Baseline Model: Random Forest**
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=config.RANDOM_SEED,
    n_jobs=-1
)

start_time = time.time()
rf.fit(X_train_scaled, y_train)
training_time = time.time() - start_time

y_val_pred = rf.predict(X_val_scaled)
y_val_proba = rf.predict_proba(X_val_scaled)[:, 1]

metrics = compute_fraud_metrics(y_val, y_val_pred, y_val_proba)
print_classification_summary(y_val, y_val_pred, y_val_proba, "Validation Set - Random Forest")
```

5. **Save Baseline Results**
```python
baseline_results = {
    'LogisticRegression': {...},  # metrics dict
    'RandomForest': {...}  # metrics dict
}

baseline_df = pd.DataFrame(baseline_results).T
baseline_df.to_csv('../results/tables/baseline_performance_targets.csv')
print("\nBaseline Performance Targets for Neural Networks:")
print(baseline_df[['precision_fraud', 'recall_fraud', 'f1_fraud', 'pr_auc']])
```

---

## Notebooks 3-8: Quick Reference

### Notebook 03: Neural Network Architectures
- Create architectures using `src.nn_architectures.create_mlp()`
- Test shallow ([32], [64,32]), medium ([128,64,32]), deep ([512,256,128,64,32,16])
- Compare wide ([1024]) vs narrow-deep ([64,64,64,64])
- Plot learning curves for each
- Log results to `nn_experiments.csv`

### Notebook 04: Regularization Experiments
- Base architecture: [128, 64, 32]
- Test dropout rates: [0.0, 0.2, 0.3, 0.4, 0.5]
- Test L2 values: [0.0, 0.001, 0.01, 0.1]
- Test with/without BatchNormalization
- Compare validation loss curves
- Identify best regularization combination

### Notebook 05: Class Imbalance Strategies
- Use best architecture from Notebook 04
- Test class weights: None, balanced, custom
- Implement focal loss
- Optimize decision threshold on validation set
- Compare PR-AUC across strategies

### Notebook 06: Neural Network Ablation Study
- Fixed architecture: [128, 64, 32]
- Baseline (no regularization)
- +Dropout only
- +L2 only
- +BatchNorm only
- +Dropout+L2
- +All combined
- Analyze component contributions

### Notebook 07: Threshold Optimization and Final Evaluation
- Select best NN from notebooks 3-6 based on validation PR-AUC
- Optimize threshold on validation set
- **Single test set evaluation**
- Error analysis (FP/FN examples)
- Confusion matrix visualization
- Business cost analysis

### Notebook 08: Neural Network Performance Analysis
- Load all experiment logs
- Rank all neural networks
- Training efficiency analysis (time vs performance)
- Feature importance using gradient methods
- Final recommendations
- Deployment readiness assessment

---

## Key Utilities Usage

### Creating Neural Networks
```python
from src.nn_architectures import create_mlp
from src.nn_training_utils import compile_model, get_callbacks, train_neural_network

# Create model
model = create_mlp(
    input_dim=7,
    hidden_layers=[128, 64, 32],
    dropout_rate=0.3,
    l2_reg=0.01,
    use_batch_norm=False
)

# Compile
model = compile_model(model, learning_rate=0.001)

# Get callbacks
callbacks = get_callbacks(
    model_save_path='../results/models/best_nn_models/model.h5',
    monitor='val_pr_auc',
    patience=15,
    experiment_name='MLP_Medium'
)

# Train
model, history = train_neural_network(
    model, X_train_scaled, y_train, X_val_scaled, y_val,
    class_weight_strategy='balanced',
    batch_size=64,
    epochs=100,
    callbacks=callbacks
)
```

### Evaluation and Visualization
```python
from src.evaluation_metrics import compute_fraud_metrics, find_optimal_threshold
from src.visualization_utils import plot_learning_curves, plot_confusion_matrix

# Evaluate
y_val_proba = model.predict(X_val_scaled).flatten()
y_val_pred = (y_val_proba > 0.5).astype(int)
metrics = compute_fraud_metrics(y_val, y_val_pred, y_val_proba)

# Plot learning curves
plot_learning_curves(
    history.history,
    metrics=['loss', 'pr_auc'],
    save_path='../results/figures/learning_curves/model_learning_curve.png'
)

# Find optimal threshold
optimal_threshold, best_f1 = find_optimal_threshold(y_val, y_val_proba, metric='f1')
```

### Logging Experiments
```python
from src.nn_training_utils import create_experiment_record, log_experiment

record = create_experiment_record(
    model_name='MLP_Medium_Dropout',
    architecture=[128, 64, 32],
    hyperparameters={'dropout_rate': 0.3, 'l2_reg': 0.01, ...},
    metrics=metrics,
    training_info={'epochs_trained': len(history.history['loss']), ...},
    notebook='03_neural_network_architectures'
)

log_experiment(record, log_file=config.NN_EXPERIMENTS_LOG)
```

---

## Troubleshooting

### Import Errors
If `src` module not found:
```python
import sys
sys.path.append('../')
```

### TensorFlow GPU Issues
```python
# Force CPU if GPU issues
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

### Memory Issues
- Reduce batch size
- Use smaller architectures
- Process data in chunks

---

This guide ensures consistent implementation across all notebooks with proper error handling, logging, and reproducibility.
