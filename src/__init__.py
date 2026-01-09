"""
Neural Network Fraud Detection Project

This package contains modules for training and evaluating neural networks
on imbalanced credit card fraud detection data.
"""

__version__ = "1.0.0"
__author__ = "NNfinalProject"

# Import key functions for easy access
from .nn_architectures import (
    create_mlp,
    create_shallow_mlp,
    create_medium_mlp,
    create_deep_mlp,
    get_architecture_by_name
)

from .nn_training_utils import (
    train_neural_network,
    compile_model,
    get_class_weights,
    get_callbacks,
    log_experiment,
    create_experiment_record
)

from .evaluation_metrics import (
    compute_fraud_metrics,
    find_optimal_threshold,
    print_classification_summary,
    compute_business_cost,
    compare_models
)

from .visualization_utils import (
    plot_learning_curves,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_model_comparison,
    plot_class_distribution
)

__all__ = [
    # Architectures
    'create_mlp',
    'create_shallow_mlp',
    'create_medium_mlp',
    'create_deep_mlp',
    'get_architecture_by_name',
    
    # Training
    'train_neural_network',
    'compile_model',
    'get_class_weights',
    'get_callbacks',
    'log_experiment',
    'create_experiment_record',
    
    # Evaluation
    'compute_fraud_metrics',
    'find_optimal_threshold',
    'print_classification_summary',
    'compute_business_cost',
    'compare_models',
    
    # Visualization
    'plot_learning_curves',
    'plot_confusion_matrix',
    'plot_precision_recall_curve',
    'plot_roc_curve',
    'plot_model_comparison',
    'plot_class_distribution',
]
