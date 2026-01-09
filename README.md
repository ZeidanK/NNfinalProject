# Neural Network Architectures for Credit Card Fraud Detection

A deep exploration of how **neural network design choices** (depth, width, regularization strategies, and class-imbalance handling) affect fraud detection performance on highly imbalanced tabular data.

## 🎯 Project Goal

**Explore how neural network design choices affect fraud detection performance under extreme class imbalance.** This project systematically compares MLP architectures and training strategies to optimize neural networks for real-world fraud detection deployment.

## 🔬 Core Research Questions

- **How does neural network depth vs width affect fraud detection capability?**
- **Which regularization strategies best prevent overfitting in fraud detection NNs?**
- **How should neural networks handle extreme class imbalance (99.2% vs 0.8%)?**
- **What architectural choices optimize the precision-recall trade-off for neural networks?**

## 📊 Dataset & Challenge

- **Source:** [Credit Card Fraud Dataset](https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud)
- **Challenge:** Severe class imbalance typical in fraud detection (fraud transactions <1%)
- **Features:** 8 tabular features suitable for MLP architecture exploration
  - `distance_from_home`
  - `distance_from_last_transaction`
  - `ratio_to_median_purchase_price`
  - `repeat_retailer`
  - `used_chip`
  - `used_pin_number`
  - `online_order`
  - `fraud` (target variable)

## 🧪 Neural Network Experiments

1. **Architecture Search:** Systematic depth vs width comparison
2. **Regularization Study:** Dropout, L2, BatchNorm effectiveness
3. **Class Imbalance:** NN-specific strategies (class weights, focal loss)
4. **Ablation Analysis:** Component contribution to NN performance
5. **Threshold Optimization:** Production deployment considerations

## 🛠️ Setup Environment

### Prerequisites
- Python 3.9+
- pip or conda package manager

### Installation

```bash
# Clone or navigate to project directory
cd NNfinalProject

# Install dependencies
pip install -r requirements.txt

# Verify installation
python config.py
```

### GPU Support (Optional)
For faster neural network training, install TensorFlow with GPU support:
```bash
pip install tensorflow-gpu>=2.10.0
```

## 📓 How to Run (Neural Network Pipeline)

Execute notebooks in the following order:

### Phase 1: Data Understanding & Preparation
1. **[01_data_analysis_and_nn_motivation.ipynb](notebooks/01_data_analysis_and_nn_motivation.ipynb)**
   - Why neural networks suit this problem
   - Data exploration and statistical analysis
   - NN design motivation

2. **[02_preprocessing_and_baseline_comparison.ipynb](notebooks/02_preprocessing_and_baseline_comparison.ipynb)**
   - Data preprocessing and split creation
   - Baseline models (targets for NNs to beat)

### Phase 2: Core Neural Network Experiments
3. **[03_neural_network_architectures.ipynb](notebooks/03_neural_network_architectures.ipynb)** 🔥
   - **Core NN architecture exploration**
   - Shallow vs medium vs deep MLPs
   - Width vs depth experiments

4. **[04_regularization_experiments.ipynb](notebooks/04_regularization_experiments.ipynb)** 🔥
   - **NN regularization strategies**
   - Dropout, L2, BatchNorm experiments
   - Overfitting prevention analysis

5. **[05_class_imbalance_strategies_for_nns.ipynb](notebooks/05_class_imbalance_strategies_for_nns.ipynb)** 🔥
   - **NN-specific imbalance handling**
   - Class weights, focal loss
   - Threshold optimization

### Phase 3: Advanced Analysis & Evaluation
6. **[06_neural_network_ablation_study.ipynb](notebooks/06_neural_network_ablation_study.ipynb)** 🔥
   - **Controlled NN component analysis**
   - Systematic ablation experiments

7. **[07_threshold_optimization_and_final_evaluation.ipynb](notebooks/07_threshold_optimization_and_final_evaluation.ipynb)** 🔥
   - **Final NN evaluation on test set**
   - Error analysis and business impact

8. **[08_neural_network_performance_analysis.ipynb](notebooks/08_neural_network_performance_analysis.ipynb)** 🔥
   - **NN performance synthesis**
   - Final recommendations

## 📁 Project Structure

```
NNfinalProject/
├── data/                           # Dataset storage
│   └── card_transdata.csv
├── notebooks/                      # Jupyter notebooks (execute in order)
│   ├── 01_data_analysis_and_nn_motivation.ipynb
│   ├── 02_preprocessing_and_baseline_comparison.ipynb
│   ├── 03_neural_network_architectures.ipynb
│   ├── 04_regularization_experiments.ipynb
│   ├── 05_class_imbalance_strategies_for_nns.ipynb
│   ├── 06_neural_network_ablation_study.ipynb
│   ├── 07_threshold_optimization_and_final_evaluation.ipynb
│   └── 08_neural_network_performance_analysis.ipynb
├── src/                            # Reusable Python modules
│   ├── nn_architectures.py        # NN architecture definitions
│   ├── nn_training_utils.py       # NN training utilities
│   ├── evaluation_metrics.py      # Evaluation functions
│   └── visualization_utils.py     # Plotting functions
├── results/                        # Output storage
│   ├── figures/                   # Plots and visualizations
│   │   ├── nn_architectures/
│   │   ├── learning_curves/
│   │   └── ablation_studies/
│   ├── models/                    # Saved models
│   │   ├── best_nn_models/
│   │   └── baseline_models/
│   └── experiment_logs/           # CSV experiment logs
│       ├── nn_experiments.csv
│       └── baseline_experiments.csv
├── reports/                        # Documentation
│   └── neural_network_methodology.md
├── docs/                           # Additional documentation
│   ├── data_leakage_prevention_checklist.md
│   └── neural_network_design_rationale.md
├── config.py                       # Project configuration
└── requirements.txt                # Python dependencies
```

## 📈 Key Deliverables

### 1. Neural Network Architecture Recommendations
Based on systematic comparison of shallow, medium, and deep MLPs with varying widths.

### 2. Regularization Strategy Guidelines
Evidence-based recommendations for Dropout, L2, and BatchNorm in imbalanced classification.

### 3. Deployment-Ready Model
Best neural network model with optimized decision threshold for production use.

### 4. Training Procedure Documentation
Complete methodology for reproducible neural network training on imbalanced data.

## 🔄 Reproducibility

This project ensures complete reproducibility through:

- **Fixed Random Seed:** All experiments use `RANDOM_SEED = 42`
- **Consistent Data Splits:** Train/val/test indices saved and reused
- **Environment Specification:** `requirements.txt` for exact package versions
- **Leakage Prevention:** StandardScaler fit only on training data
- **Comprehensive Logging:** All experiments logged to CSV with hyperparameters

See [docs/data_leakage_prevention_checklist.md](docs/data_leakage_prevention_checklist.md) for detailed guidelines.

## 📊 Evaluation Metrics

### Primary Metrics (Neural Network Focus)
- **Precision (Fraud Class):** Minimize false positives
- **Recall (Fraud Class):** Minimize false negatives
- **F1-Score (Fraud Class):** Balance precision and recall
- **PR-AUC:** Precision-Recall curve area (best for imbalanced data)
- **ROC-AUC:** Receiver Operating Characteristic curve area

### Secondary Metrics
- Accuracy (reported but not primary due to class imbalance)
- Confusion Matrix Analysis
- Training/Validation Loss Curves

**Why not Accuracy?** With 99.2% legitimate transactions, a model predicting "all legitimate" achieves 99.2% accuracy while detecting zero fraud. Hence, fraud-class specific metrics are essential.

## 🧠 Neural Network Mastery

This project demonstrates mastery of neural networks by:

1. **Systematic Architecture Exploration:** Depth vs width trade-offs with empirical evidence
2. **Regularization Analysis:** Understanding when and why each regularization technique helps
3. **Class Imbalance Handling:** NN-specific strategies beyond classical ML approaches
4. **Controlled Ablation Studies:** Isolating individual component contributions
5. **Training Dynamics:** Analyzing learning curves and convergence behavior
6. **Production Considerations:** Threshold optimization and deployment readiness

## 📚 References & Credits

- **Dataset Source:** [Kaggle - Credit Card Fraud Dataset](https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud)
- **License:** Please refer to the original dataset license on Kaggle
- **Academic Context:** Neural Networks & Deep Learning Final Project

## 👤 Author

Created as part of Neural Networks & Deep Learning coursework.

## 📝 License

This project is for educational purposes. Dataset license applies as per original source.

---

**Note:** This is a neural networks research project. Classical ML baselines (Logistic Regression, Random Forest) are included only for comparison purposes to demonstrate the advantages of neural network approaches for this problem.
