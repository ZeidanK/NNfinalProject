"""
Project Configuration for Neural Network Fraud Detection

This file contains all global configuration parameters for reproducibility.
All random seeds, paths, and hyperparameter defaults are centralized here.
"""

import os
from pathlib import Path

# ============================
# PROJECT PATHS
# ============================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
FIGURES_DIR = RESULTS_DIR / "figures"
EXPERIMENT_LOGS_DIR = RESULTS_DIR / "experiment_logs"

# ============================
# DATA FILES
# ============================
DATASET_PATH = DATA_DIR / "card_transdata.csv"
TRAIN_INDICES_PATH = RESULTS_DIR / "train_indices.npy"
VAL_INDICES_PATH = RESULTS_DIR / "val_indices.npy"
TEST_INDICES_PATH = RESULTS_DIR / "test_indices.npy"
FITTED_SCALER_PATH = RESULTS_DIR / "fitted_scaler.pkl"

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
DEFAULT_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_MIN_DELTA = 0.0001

# ============================
# NEURAL NETWORK ARCHITECTURES
# ============================
# Architectures to test systematically
SHALLOW_ARCHITECTURES = [
    [32],
    [64, 32],
]

MEDIUM_ARCHITECTURES = [
    [128, 64, 32],
    [256, 128, 64, 32],
]

DEEP_ARCHITECTURES = [
    [512, 256, 128, 64, 32, 16],
]

WIDTH_VS_DEPTH_ARCHITECTURES = [
    [1024],  # Very wide, shallow
    [64, 64, 64, 64],  # Narrow, deep
]

# ============================
# REGULARIZATION EXPERIMENTS
# ============================
DROPOUT_RATES = [0.0, 0.2, 0.3, 0.4, 0.5]
L2_REGULARIZATION_VALUES = [0.0, 0.001, 0.01, 0.1]
BATCH_NORMALIZATION_OPTIONS = [True, False]

# ============================
# CLASS IMBALANCE HANDLING
# ============================
CLASS_WEIGHT_STRATEGIES = [
    None,
    "balanced",
    "custom"  # Will be calculated from data
]

# ============================
# EVALUATION METRICS
# ============================
PRIMARY_METRICS = [
    "precision_fraud",
    "recall_fraud",
    "f1_fraud",
    "pr_auc",
    "roc_auc"
]

SECONDARY_METRICS = [
    "accuracy",
    "precision_legitimate",
    "recall_legitimate"
]

# ============================
# EXPERIMENT LOGGING
# ============================
NN_EXPERIMENTS_LOG = EXPERIMENT_LOGS_DIR / "nn_experiments.csv"
BASELINE_EXPERIMENTS_LOG = EXPERIMENT_LOGS_DIR / "baseline_experiments.csv"

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
DATASET_SOURCE = "https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud"
TARGET_COLUMN = "fraud"
FEATURE_COLUMNS = [
    "distance_from_home",
    "distance_from_last_transaction",
    "ratio_to_median_purchase_price",
    "repeat_retailer",
    "used_chip",
    "used_pin_number",
    "online_order"
]

# ============================
# HELPER FUNCTIONS
# ============================
def ensure_directories():
    """Create all necessary directories if they don't exist"""
    directories = [
        RESULTS_DIR,
        MODELS_DIR,
        FIGURES_DIR,
        FIGURES_DIR / "nn_architectures",
        FIGURES_DIR / "learning_curves",
        FIGURES_DIR / "ablation_studies",
        MODELS_DIR / "best_nn_models",
        MODELS_DIR / "baseline_models",
        EXPERIMENT_LOGS_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def set_random_seeds(seed=RANDOM_SEED):
    """Set random seeds for reproducibility"""
    import random
    import numpy as np
    import tensorflow as tf
    
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # For additional reproducibility in TensorFlow
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

if __name__ == "__main__":
    print("Project Configuration")
    print("=" * 50)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Dataset Path: {DATASET_PATH}")
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Train/Val/Test Split: {TRAIN_SIZE}/{VAL_SIZE}/{TEST_SIZE}")
    print(f"Primary Metrics: {', '.join(PRIMARY_METRICS)}")
    ensure_directories()
    print("\nAll directories created successfully!")
