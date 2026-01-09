# Dual-Dataset Neural Network Study for Credit Card Fraud Detection

A rigorous investigation of how **neural network architectural and regularization decisions** generalize across fraud detection data regimes by conducting controlled experiments on a synthetic dataset and validating performance on a real-world, PCA-transformed dataset.

## 🎯 Project Goal

**Demonstrate that neural network design principles discovered in controlled settings transfer to production-grade fraud detection scenarios.** This dual-dataset approach ensures findings are robust and generalizable rather than artifacts of a single dataset's peculiarities.

## 🔬 Core Research Question

**Do neural network architectural choices (depth vs width, regularization strategies, threshold optimization) generalize across fraud detection data regimes?**

### Secondary Questions
- Which architectural configurations remain stable across synthetic and real-world data?
- How do regularization strategies (dropout, L2, batch normalization) adapt to different imbalance ratios?
- What design principles transfer beyond simplified benchmarks to production scenarios?
- When do baseline models (Random Forest, Logistic Regression) dominate, and why?

## 📊 Dual-Dataset Approach

### Dataset 1: card_transdata.csv (Synthetic - Exploration Phase)
- **Source:** [Kaggle - Credit Card Fraud Dataset](https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud)
- **Size:** ~1,000,000 transactions
- **Features:** 7 interpretable features (distances, ratios, binary indicators)
- **Fraud Rate:** ~0.8% (moderate imbalance, 1:124 ratio)
- **Role:** Controlled environment for architecture exploration and ablation studies
- **Expected Baseline:** Random Forest likely achieves PR-AUC ≈ 1.0 (near-perfect)
- **Value:** Clean experiments without confounding factors

### Dataset 2: creditcard.csv (Real-World - Validation Phase)
- **Source:** [Kaggle - ULB Credit Card Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** ~284,000 transactions
- **Features:** 30 PCA-transformed components + Time + Amount
- **Fraud Rate:** ~0.17% (extreme imbalance, 1:577 ratio)
- **Role:** Production-grade validation benchmark
- **Expected Baseline:** Competitive but challenging
- **Value:** Authoritative test of generalization

## 🧪 Experimental Design

### Phase 1: Architecture Exploration (card_transdata.csv)
1. **8 Architecture Comparison:** Shallow ([32]), Medium ([128,64,32]), Deep ([512,256,128,64,32,16]), Wide ([512])
2. **Ablation Study:** Isolate effects of dropout, L2, batch normalization
3. **Selection:** Best architecture by validation PR-AUC → transfer to Phase 2

### Phase 2: Regularization Optimization (creditcard.csv)
1. **Apply Best Architecture:** Use structure from Phase 1, train from scratch
2. **Regularization Grid:** Dropout [0.2-0.4], L2 [0.001-0.01], BatchNorm
3. **Threshold Optimization:** Tune on validation set for precision/recall balance
4. **ONE-TIME Test Evaluation:** Final held-out performance (no iteration)

### Phase 3: Cross-Dataset Analysis
1. Compare NN behavior across datasets (overfitting, stability)
2. Identify transferable design principles
3. Synthesize generalizable insights for fraud detection NNs

## 🛠️ Setup Environment

### Prerequisites
- Python 3.9+
- TensorFlow 2.10+
- scikit-learn, pandas, numpy, matplotlib, seaborn

### Installation

```bash
# Navigate to project directory
cd NNfinalProject

# Install dependencies
pip install -r requirements.txt

# Verify setup and create directory structure
python config.py
```

### Data Acquisition
Download both datasets and place in the `data/` directory:
1. **card_transdata.csv:** [Kaggle Link](https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud)
2. **creditcard.csv:** [Kaggle Link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## 📓 Notebook Execution Order

### Phase 1: Architecture Exploration (Synthetic Dataset)

#### 1. [01_dual_dataset_overview.ipynb](notebooks/01_dual_dataset_overview.ipynb)
**Purpose:** Comparative EDA and NN motivation  
**Key Outputs:**
- Class distribution comparison (0.8% vs 0.17% fraud)
- Feature space analysis (interpretable vs PCA)
- Dataset difficulty quantification (Cohen's d)
- Cross-dataset comparison table

#### 2. [02_card_transdata_preprocessing_and_baselines.ipynb](notebooks/02_card_transdata_preprocessing_and_baselines.ipynb)
**Purpose:** Establish preprocessing pipeline and baseline performance  
**Key Outputs:**
- Train/val/test splits (70/15/15, stratified)
- Fitted StandardScaler (training set only)
- Logistic Regression baseline
- Random Forest baseline (expected: PR-AUC ≈ 1.0)

#### 3. [03_card_transdata_nn_architecture_and_ablation.ipynb](notebooks/03_card_transdata_nn_architecture_and_ablation.ipynb) 🔥
**Purpose:** Systematic architecture exploration + ablation study  
**Key Outputs:**
- 8 architecture comparison (ARCH-01 to ARCH-08)
- Ablation study (dropout, L2, batch norm effects)
- **Best architecture selection for Phase 2**
- Architecture performance table

### Phase 2: Production Validation (Real-World Dataset)

#### 4. [04_creditcard_preprocessing_and_baselines.ipynb](notebooks/04_creditcard_preprocessing_and_baselines.ipynb)
**Purpose:** Independent preprocessing and baseline training  
**Key Outputs:**
- Separate train/val/test splits for creditcard.csv
- Separate fitted StandardScaler
- Baselines on extreme imbalance (~0.17% fraud)

#### 5. [05_creditcard_nn_training_and_regularization.ipynb](notebooks/05_creditcard_nn_training_and_regularization.ipynb) 🔥
**Purpose:** Apply best architecture + regularization optimization  
**Key Outputs:**
- Best architecture from NB03 trained on real data
- Regularization experiments (REG-01 to REG-08)
- **Best model selection by validation PR-AUC**
- Learning curves and overfitting analysis

#### 6. [06_creditcard_threshold_optimization_and_test_evaluation.ipynb](notebooks/06_creditcard_threshold_optimization_and_test_evaluation.ipynb) ⚠️
**Purpose:** Threshold tuning + ONE-TIME test evaluation  
**Key Outputs:**
- Threshold optimization on validation set
- **Final test set evaluation (EXACTLY ONCE)**
- Error analysis (FP/FN patterns)
- Business cost simulation

### Phase 3: Cross-Dataset Analysis

#### 7. [07_cross_dataset_nn_insights_and_conclusions.ipynb](notebooks/07_cross_dataset_nn_insights_and_conclusions.ipynb) 📊
**Purpose:** Synthesize findings across datasets  
**Key Outputs:**
- Generalization comparison
- Transferable design principles
- Overfitting analysis across data regimes
- Final recommendations and limitations

## 📁 Project Structure

```
NNfinalProject/
├── data/                                    # Datasets
│   ├── card_transdata.csv                   # Synthetic dataset (~1M rows)
│   └── creditcard.csv                       # Real-world ULB dataset (~284K rows)
│
├── notebooks/                               # Jupyter notebooks (7 total)
│   ├── 01_dual_dataset_overview.ipynb
│   ├── 02_card_transdata_preprocessing_and_baselines.ipynb
│   ├── 03_card_transdata_nn_architecture_and_ablation.ipynb
│   ├── 04_creditcard_preprocessing_and_baselines.ipynb
│   ├── 05_creditcard_nn_training_and_regularization.ipynb
│   ├── 06_creditcard_threshold_optimization_and_test_evaluation.ipynb
│   └── 07_cross_dataset_nn_insights_and_conclusions.ipynb
│
├── src/                                     # Reusable utility modules
│   ├── nn_architectures.py                  # MLP builders (dataset-agnostic)
│   ├── nn_training_utils.py                 # Training pipeline, logging
│   ├── evaluation_metrics.py                # Fraud-specific metrics
│   └── visualization_utils.py               # Plotting functions
│
├── results/                                 # All experimental outputs
│   ├── card_transdata/                      # Synthetic dataset results
│   │   ├── models/
│   │   │   ├── baselines/                   # LR, RF baselines
│   │   │   └── neural_networks/             # 8+ NN architectures
│   │   ├── figures/                         # Visualizations
│   │   ├── tables/                          # CSV summaries
│   │   └── logs/                            # Experiment logs
│   │
│   ├── creditcard/                          # Real-world dataset results
│   │   ├── models/
│   │   │   ├── baselines/                   # LR, RF baselines
│   │   │   └── neural_networks/             # Best NN + regularization variants
│   │   ├── figures/                         # Visualizations
│   │   ├── tables/                          # CSV summaries (incl. final test)
│   │   └── logs/                            # Experiment logs
│   │
│   └── cross_dataset_analysis/              # Comparative analysis
│       ├── figures/                         # Cross-dataset comparisons
│       └── tables/                          # Summary tables
│
├── docs/                                    # Methodology documentation
│   ├── data_leakage_prevention_checklist.md # 27 mandatory rules
│   ├── neural_network_design_rationale.md   # Architecture justification
│   └── dual_dataset_methodology.md          # Dual-dataset approach explained
│
├── reports/
│   └── neural_network_methodology.md        # Comprehensive report
│
├── config.py                                # Global configuration
├── requirements.txt                         # Python dependencies
└── README.md                                # This file
```
├── docs/                           # Additional documentation
│   ├── data_leakage_prevention_checklist.md
│   └── neural_network_design_rationale.md
├── config.py                       # Project configuration
└── requirements.txt                # Python dependencies
```

## 📈 Key Deliverables

### 1. Architecture Generalization Analysis
Systematic evidence showing which NN architectures transfer from synthetic to real-world fraud detection.

### 2. Regularization Strategy Guidelines
Dataset-specific regularization recommendations with empirical validation across imbalance ratios.

### 3. Cross-Dataset Performance Comparison
Comprehensive analysis of NN behavior (overfitting, stability, generalization) across data regimes.

### 4. Production-Ready Model with Threshold Optimization
Best NN model validated on real-world data with optimized decision threshold.

### 5. Reproducible Experimental Methodology
Complete dual-dataset pipeline for evaluating NN design principles in fraud detection.

## 🔄 Reproducibility Guarantees

This project ensures complete reproducibility through:

- **Fixed Random Seed:** All experiments use `RANDOM_SEED = 42`
- **Saved Data Splits:** Train/val/test indices stored as `.npy` files (never re-split)
- **Fitted Scalers:** Preprocessing fitted once on training data, saved as `.pkl`
- **Dataset Isolation:** Separate preprocessing pipelines for each dataset
- **Experiment Logging:** Complete CSV logs with 25+ columns per experiment
- **One-Time Test Evaluation:** Test set touched EXACTLY ONCE per dataset
- **Version Control:** All hyperparameters documented in experiment logs

See [docs/data_leakage_prevention_checklist.md](docs/data_leakage_prevention_checklist.md) for 27 mandatory rules.

## 📊 Evaluation Metrics

### Primary Metrics
- **PR-AUC:** Precision-Recall Area Under Curve (best for extreme imbalance)
- **Recall (Fraud):** Proportion of actual fraud caught
- **Precision (Fraud):** Proportion of fraud predictions that are correct
- **F1-Score (Fraud):** Harmonic mean of precision and recall
- **ROC-AUC:** Receiver Operating Characteristic curve area

### Secondary Metrics
- Accuracy (reported but not primary - can be misleading with imbalance)
- Confusion Matrix Analysis
- Training-Validation Gap (overfitting measure)

**Why PR-AUC over Accuracy?** With 0.17%-0.8% fraud rates, a model predicting "all legitimate" achieves 99%+ accuracy while detecting zero fraud. PR-AUC correctly captures fraud detection performance.

## 🎓 Why This Project Matters

### Academic Contribution
This dual-dataset approach addresses a critical gap in NN fraud detection research:

**Problem:** Most studies use single datasets, making it unclear if findings generalize or are dataset-specific artifacts.

**Solution:** By separating controlled exploration (synthetic data) from production validation (real-world data), we demonstrate that NN design principles are **transferable** rather than coincidental.

### Key Insights Expected
1. **Architecture Transfer:** Do layer configurations that work on clean data generalize to noisy data?
2. **Regularization Adaptation:** How do dropout/L2/batch norm requirements change with extreme imbalance?
3. **Baseline Dominance:** When and why do tree-based models dominate, and what does this reveal about data complexity?
4. **Production Readiness:** What NN design choices lead to stable deployment-grade models?

## 🧠 Neural Network Mastery Demonstration

This project showcases deep NN expertise through:

1. **Systematic Architecture Exploration:** 8 architectures compared with principled layer design
2. **Controlled Ablation Studies:** Isolating dropout, L2, and batch normalization effects independently
3. **Regularization Optimization:** Evidence-based selection across different imbalance ratios
4. **Cross-Dataset Validation:** Demonstrating generalization beyond single-dataset optimization
5. **Training Dynamics Analysis:** Learning curves, convergence monitoring, early stopping
6. **Threshold Optimization:** Production-grade precision/recall tradeoff tuning
7. **Data Leakage Prevention:** Rigorous experimental methodology with 27 mandatory checks

## 📚 References & Documentation

- **card_transdata.csv:** [Kaggle - Credit Card Fraud Dataset](https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud)
- **creditcard.csv:** [Kaggle - ULB Credit Card Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Methodology:** See [docs/dual_dataset_methodology.md](docs/dual_dataset_methodology.md)
- **Design Rationale:** See [docs/neural_network_design_rationale.md](docs/neural_network_design_rationale.md)
- **Academic Context:** Neural Networks & Deep Learning Final Project

## 🚨 Important Notes

### Baseline Performance Expectations
- **On card_transdata.csv:** Random Forest may achieve PR-AUC ≈ 1.0 (near-perfect). This is **expected and acceptable** - the synthetic dataset's role is enabling clean architecture studies, not challenging baselines.
- **On creditcard.csv:** Baselines provide realistic reference points. NN value lies in transferable design principles and controlled generalization.

### Test Set Usage
- Test sets are evaluated **EXACTLY ONCE** per dataset
- No model iteration after seeing test results
- Results reported as-is with honest limitations discussed

---

**Project Focus:** Demonstrating that neural network design principles generalize across fraud detection data regimes, NOT claiming NNs always beat Random Forest.
