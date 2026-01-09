"""
Project Configuration for Dual-Dataset Neural Network Fraud Detection Study

This file contains all global configuration parameters for reproducibility.
All random seeds, paths, and hyperparameter defaults are centralized here.

DUAL-DATASET STUDY:
- card_transdata.csv: Synthetic dataset for architecture exploration and ablation
- creditcard.csv: Real-world PCA-transformed dataset for validation
"""

import os
from pathlib import Path

# ============================
# PROJECT PATHS
# ============================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# ============================
# DUAL-DATASET PATHS
# ============================
# Synthetic dataset (architecture exploration)
CARD_TRANSDATA_PATH = DATA_DIR / "card_transdata.csv"
CARD_TRANSDATA_RESULTS = RESULTS_DIR / "card_transdata"
CARD_TRANSDATA_MODELS = CARD_TRANSDATA_RESULTS / "models"
CARD_TRANSDATA_FIGURES = CARD_TRANSDATA_RESULTS / "figures"
CARD_TRANSDATA_TABLES = CARD_TRANSDATA_RESULTS / "tables"
CARD_TRANSDATA_LOGS = CARD_TRANSDATA_RESULTS / "logs"
CARD_TRANSDATA_TRAIN_IDX = CARD_TRANSDATA_RESULTS / "train_indices.npy"
CARD_TRANSDATA_VAL_IDX = CARD_TRANSDATA_RESULTS / "val_indices.npy"
CARD_TRANSDATA_TEST_IDX = CARD_TRANSDATA_RESULTS / "test_indices.npy"
CARD_TRANSDATA_SCALER = CARD_TRANSDATA_RESULTS / "fitted_scaler.pkl"

# Real-world dataset (validation)
CREDITCARD_PATH = DATA_DIR / "creditcard.csv"
CREDITCARD_RESULTS = RESULTS_DIR / "creditcard"
CREDITCARD_MODELS = CREDITCARD_RESULTS / "models"
CREDITCARD_FIGURES = CREDITCARD_RESULTS / "figures"
CREDITCARD_TABLES = CREDITCARD_RESULTS / "tables"
CREDITCARD_LOGS = CREDITCARD_RESULTS / "logs"
CREDITCARD_TRAIN_IDX = CREDITCARD_RESULTS / "train_indices.npy"
CREDITCARD_VAL_IDX = CREDITCARD_RESULTS / "val_indices.npy"
CREDITCARD_TEST_IDX = CREDITCARD_RESULTS / "test_indices.npy"
CREDITCARD_SCALER = CREDITCARD_RESULTS / "fitted_scaler.pkl"

# Cross-dataset analysis
CROSS_DATASET_RESULTS = RESULTS_DIR / "cross_dataset_analysis"
CROSS_DATASET_FIGURES = CROSS_DATASET_RESULTS / "figures"
CROSS_DATASET_TABLES = CROSS_DATASET_RESULTS / "tables"

# Legacy paths (for backward compatibility with existing code)
DATASET_PATH = CARD_TRANSDATA_PATH
TRAIN_INDICES_PATH = CARD_TRANSDATA_TRAIN_IDX
VAL_INDICES_PATH = CARD_TRANSDATA_VAL_IDX
TEST_INDICES_PATH = CARD_TRANSDATA_TEST_IDX
FITTED_SCALER_PATH = CARD_TRANSDATA_SCALER
MODELS_DIR = CARD_TRANSDATA_MODELS
FIGURES_DIR = CARD_TRANSDATA_FIGURES
EXPERIMENT_LOGS_DIR = CARD_TRANSDATA_LOGS

# ============================
# REPRODUCIBILITY
# ============================
RANDOM_SEED = 42
"""Random seed for all experiments to ensure reproducibility"""

# ============================
# DATA SPLITTING
# ============================
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15
STRATIFY = True  # Always stratify for imbalanced data

# ============================
# NEURAL NETWORK DEFAULTS
# ============================
DEFAULT_OPTIMIZER = "adam"
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_BATCH_SIZE = 64

# Dataset-specific training configs
CARD_TRANSDATA_EPOCHS = 100
CARD_TRANSDATA_EARLY_STOP_PATIENCE = 15

CREDITCARD_EPOCHS = 150  # Longer for harder dataset
CREDITCARD_EARLY_STOP_PATIENCE = 20

EARLY_STOPPING_MIN_DELTA = 0.0001

# ============================
# ARCHITECTURE EXPLORATION (card_transdata.csv)
# ============================
# 8 architectures for systematic comparison
ARCHITECTURES_TO_TEST = {
    "ARCH-01": {"name": "shallow_tiny", "layers": [32], "purpose": "Minimal capacity baseline"},
    "ARCH-02": {"name": "shallow_small", "layers": [64, 32], "purpose": "Small network"},
    "ARCH-03": {"name": "medium_base", "layers": [128, 64, 32], "purpose": "Balanced architecture"},
    "ARCH-04": {"name": "medium_deep", "layers": [256, 128, 64, 32], "purpose": "Moderate depth"},
    "ARCH-05": {"name": "deep", "layers": [512, 256, 128, 64, 32, 16], "purpose": "Maximum depth"},
    "ARCH-06": {"name": "wide_medium", "layers": [256], "purpose": "Wide-shallow"},
    "ARCH-07": {"name": "wide_large", "layers": [512], "purpose": "Very wide-shallow"},
    "ARCH-08": {"name": "balanced", "layers": [256, 128, 64], "purpose": "Depth-width balance"},
}

# ============================
# ABLATION STUDY (card_transdata.csv)
# ============================
# Use medium_base [128, 64, 32] as reference
ABLATION_EXPERIMENTS = {
    "ABL-01": {"dropout": 0.0, "l2": 0.0, "batch_norm": False, "desc": "Baseline"},
    "ABL-02": {"dropout": 0.3, "l2": 0.0, "batch_norm": False, "desc": "+Dropout"},
    "ABL-03": {"dropout": 0.0, "l2": 0.01, "batch_norm": False, "desc": "+L2"},
    "ABL-04": {"dropout": 0.0, "l2": 0.0, "batch_norm": True, "desc": "+BatchNorm"},
    "ABL-05": {"dropout": 0.3, "l2": 0.01, "batch_norm": True, "desc": "Combined"},
}

# ============================
# REGULARIZATION OPTIMIZATION (creditcard.csv)
# ============================
REGULARIZATION_EXPERIMENTS = {
    "REG-01": {"dropout": 0.0, "l2": 0.0, "batch_norm": False, "desc": "Baseline"},
    "REG-02": {"dropout": 0.2, "l2": 0.0, "batch_norm": False, "desc": "Light dropout"},
    "REG-03": {"dropout": 0.3, "l2": 0.0, "batch_norm": False, "desc": "Medium dropout"},
    "REG-04": {"dropout": 0.4, "l2": 0.0, "batch_norm": False, "desc": "Heavy dropout"},
    "REG-05": {"dropout": 0.0, "l2": 0.001, "batch_norm": False, "desc": "Light L2"},
    "REG-06": {"dropout": 0.0, "l2": 0.01, "batch_norm": False, "desc": "Medium L2"},
    "REG-07": {"dropout": 0.0, "l2": 0.0, "batch_norm": True, "desc": "BatchNorm only"},
    "REG-08": {"dropout": 0.3, "l2": 0.01, "batch_norm": True, "desc": "Combined optimal"},
}

# ============================
# THRESHOLD OPTIMIZATION
# ============================
THRESHOLD_CANDIDATES = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]

# ============================
# CLASS IMBALANCE HANDLING
# ============================
CLASS_WEIGHT_STRATEGY = "balanced"  # Fixed for all experiments

# ============================
# EVALUATION METRICS
# ============================
PRIMARY_METRICS = [
    "pr_auc",
    "roc_auc",
    "f1_fraud",
    "precision_fraud",
    "recall_fraud",
]

SECONDARY_METRICS = [
    "accuracy",
    "precision_legitimate",
    "recall_legitimate"
]

# ============================
# EXPERIMENT LOGGING
# ============================
# Log columns for reproducibility
EXPERIMENT_LOG_COLUMNS = [
    "experiment_id", "timestamp", "dataset", "architecture_name", "layer_config",
    "num_parameters", "dropout_rate", "l2_reg", "batch_norm", "optimizer", 
    "learning_rate", "batch_size", "epochs_trained", "early_stopped", 
    "train_time_seconds", "train_loss", "train_pr_auc", "train_recall_fraud", 
    "train_f1_fraud", "val_loss", "val_pr_auc", "val_recall_fraud", 
    "val_f1_fraud", "overfitting_gap_pr_auc", "notes"
]

# ============================
# VISUALIZATION SETTINGS
# ============================
FIGURE_DPI = 300
FIGURE_FORMAT = "png"
FIGURE_SIZE = (10, 6)
LARGE_FIGURE_SIZE = (14, 8)

# ============================
# DATASET INFORMATION
# ============================
# card_transdata.csv (Synthetic dataset)
CARD_TRANSDATA_SOURCE = "https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud"
CARD_TRANSDATA_TARGET = "fraud"
CARD_TRANSDATA_FEATURES = [
    "distance_from_home",
    "distance_from_last_transaction",
    "ratio_to_median_purchase_price",
    "repeat_retailer",
    "used_chip",
    "used_pin_number",
    "online_order"
]

# creditcard.csv (Real-world ULB dataset)
CREDITCARD_SOURCE = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
CREDITCARD_TARGET = "Class"
# Features V1-V28 are PCA components, plus Time and Amount
CREDITCARD_FEATURES = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]

# Legacy (backward compatibility)
TARGET_COLUMN = CARD_TRANSDATA_TARGET
FEATURE_COLUMNS = CARD_TRANSDATA_FEATURES

# ============================
# BASELINE CONFIGURATIONS
# ============================
# Logistic Regression (both datasets)
LR_CONFIG = {
    "class_weight": "balanced",
    "max_iter": 1000,
    "random_state": RANDOM_SEED,
    "n_jobs": -1
}

# Random Forest (dataset-specific)
RF_CARD_TRANSDATA_CONFIG = {
    "n_estimators": 100,
    "max_depth": 15,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "n_jobs": -1
}

RF_CREDITCARD_CONFIG = {
    "n_estimators": 100,
    "max_depth": 20,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "n_jobs": -1
}

# ============================
# HELPER FUNCTIONS
# ============================
def ensure_directories():
    """Create all necessary directories for dual-dataset study"""
    directories = [
        # card_transdata directories
        CARD_TRANSDATA_RESULTS,
        CARD_TRANSDATA_MODELS,
        CARD_TRANSDATA_MODELS / "baselines",
        CARD_TRANSDATA_MODELS / "neural_networks",
        CARD_TRANSDATA_FIGURES,
        CARD_TRANSDATA_TABLES,
        CARD_TRANSDATA_LOGS,
        
        # creditcard directories
        CREDITCARD_RESULTS,
        CREDITCARD_MODELS,
        CREDITCARD_MODELS / "baselines",
        CREDITCARD_MODELS / "neural_networks",
        CREDITCARD_FIGURES,
        CREDITCARD_TABLES,
        CREDITCARD_LOGS,
        
        # Cross-dataset analysis
        CROSS_DATASET_RESULTS,
        CROSS_DATASET_FIGURES,
        CROSS_DATASET_TABLES,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def get_dataset_config(dataset_name):
    """
    Get dataset-specific configuration
    
    Args:
        dataset_name: 'card_transdata' or 'creditcard'
    
    Returns:
        dict: Configuration dictionary with paths and settings
    """
    if dataset_name == "card_transdata":
        return {
            "data_path": CARD_TRANSDATA_PATH,
            "results_dir": CARD_TRANSDATA_RESULTS,
            "models_dir": CARD_TRANSDATA_MODELS,
            "figures_dir": CARD_TRANSDATA_FIGURES,
            "tables_dir": CARD_TRANSDATA_TABLES,
            "logs_dir": CARD_TRANSDATA_LOGS,
            "train_idx": CARD_TRANSDATA_TRAIN_IDX,
            "val_idx": CARD_TRANSDATA_VAL_IDX,
            "test_idx": CARD_TRANSDATA_TEST_IDX,
            "scaler_path": CARD_TRANSDATA_SCALER,
            "target_col": CARD_TRANSDATA_TARGET,
            "feature_cols": CARD_TRANSDATA_FEATURES,
            "epochs": CARD_TRANSDATA_EPOCHS,
            "early_stop_patience": CARD_TRANSDATA_EARLY_STOP_PATIENCE,
            "rf_config": RF_CARD_TRANSDATA_CONFIG,
        }
    elif dataset_name == "creditcard":
        return {
            "data_path": CREDITCARD_PATH,
            "results_dir": CREDITCARD_RESULTS,
            "models_dir": CREDITCARD_MODELS,
            "figures_dir": CREDITCARD_FIGURES,
            "tables_dir": CREDITCARD_TABLES,
            "logs_dir": CREDITCARD_LOGS,
            "train_idx": CREDITCARD_TRAIN_IDX,
            "val_idx": CREDITCARD_VAL_IDX,
            "test_idx": CREDITCARD_TEST_IDX,
            "scaler_path": CREDITCARD_SCALER,
            "target_col": CREDITCARD_TARGET,
            "feature_cols": CREDITCARD_FEATURES,
            "epochs": CREDITCARD_EPOCHS,
            "early_stop_patience": CREDITCARD_EARLY_STOP_PATIENCE,
            "rf_config": RF_CREDITCARD_CONFIG,
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'card_transdata' or 'creditcard'")

def set_random_seeds(seed=RANDOM_SEED):
    """Set random seeds for reproducibility"""
    import random
    import numpy as np
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        # For additional reproducibility in TensorFlow
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    except ImportError:
        pass
    
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == "__main__":
    print("Dual-Dataset Project Configuration")
    print("=" * 70)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"\nDataset 1 (Synthetic): {CARD_TRANSDATA_PATH}")
    print(f"Dataset 2 (Real-world): {CREDITCARD_PATH}")
    print(f"\nRandom Seed: {RANDOM_SEED}")
    print(f"Train/Val/Test Split: {TRAIN_SIZE}/{VAL_SIZE}/{TEST_SIZE}")
    print(f"Primary Metrics: {', '.join(PRIMARY_METRICS)}")
    print(f"\nArchitectures to test: {len(ARCHITECTURES_TO_TEST)}")
    print(f"Ablation experiments: {len(ABLATION_EXPERIMENTS)}")
    print(f"Regularization experiments: {len(REGULARIZATION_EXPERIMENTS)}")
    
    print("\nCreating directory structure...")
    ensure_directories()
    print("✓ All directories created successfully!")
