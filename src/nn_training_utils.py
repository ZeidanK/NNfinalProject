"""
Neural Network Training Utilities

This module provides utilities for training neural networks with proper
monitoring, callbacks, and experiment logging for fraud detection.
"""

import time
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    EarlyStopping, 
    ModelCheckpoint, 
    ReduceLROnPlateau,
    Callback
)
from sklearn.utils.class_weight import compute_class_weight
import config


class ExperimentLogger(Callback):
    """Custom callback to log training progress"""
    
    def __init__(self, experiment_name: str):
        super().__init__()
        self.experiment_name = experiment_name
        self.epoch_times = []
        self.start_time = None
        
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        print(f"\n{'='*60}")
        print(f"Starting experiment: {self.experiment_name}")
        print(f"{'='*60}")
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()
        
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start
        self.epoch_times.append(epoch_time)
        
    def on_train_end(self, logs=None):
        total_time = time.time() - self.start_time
        print(f"\n{'='*60}")
        print(f"Training completed in {total_time:.2f} seconds")
        print(f"Average epoch time: {np.mean(self.epoch_times):.2f} seconds")
        print(f"{'='*60}\n")


def get_class_weights(y_train: np.ndarray, strategy: str = 'balanced') -> Dict[int, float]:
    """
    Compute class weights for imbalanced data.
    
    Parameters:
    -----------
    y_train : np.ndarray
        Training labels
    strategy : str
        'balanced' or 'custom' (or None for no weighting)
        
    Returns:
    --------
    Dict[int, float]
        Class weight dictionary
    """
    if strategy is None or strategy == 'none':
        return None
    
    if strategy == 'balanced':
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
        
    elif strategy == 'custom':
        # Custom strategy: ratio-based
        n_samples = len(y_train)
        n_fraud = np.sum(y_train == 1)
        n_legit = np.sum(y_train == 0)
        
        class_weight_dict = {
            0: 1.0,
            1: n_legit / n_fraud if n_fraud > 0 else 1.0
        }
    
    else:
        raise ValueError(f"Unknown class weight strategy: {strategy}")
    
    print(f"Class weights ({strategy}): {class_weight_dict}")
    return class_weight_dict


def get_callbacks(
    model_save_path: str,
    monitor: str = 'val_loss',
    patience: int = 15,
    experiment_name: str = 'NN_Experiment'
) -> List[Callback]:
    """
    Get standard callbacks for neural network training.
    
    Parameters:
    -----------
    model_save_path : str
        Path to save best model
    monitor : str
        Metric to monitor ('val_loss', 'val_pr_auc', etc.)
    patience : int
        Early stopping patience
    experiment_name : str
        Name for experiment logger
        
    Returns:
    --------
    List[Callback]
    """
    callbacks = []
    
    # Early Stopping
    early_stop = EarlyStopping(
        monitor=monitor,
        patience=patience,
        min_delta=0.0001,
        restore_best_weights=True,
        verbose=1,
        mode='auto'
    )
    callbacks.append(early_stop)
    
    # Model Checkpoint
    checkpoint = ModelCheckpoint(
        model_save_path,
        monitor=monitor,
        save_best_only=True,
        mode='auto',
        verbose=1
    )
    callbacks.append(checkpoint)
    
    # Reduce Learning Rate on Plateau
    reduce_lr = ReduceLROnPlateau(
        monitor=monitor,
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1,
        mode='auto'
    )
    callbacks.append(reduce_lr)
    
    # Custom Experiment Logger
    exp_logger = ExperimentLogger(experiment_name)
    callbacks.append(exp_logger)
    
    return callbacks


def train_neural_network(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weight_strategy: str = 'balanced',
    batch_size: int = 64,
    epochs: int = 100,
    callbacks: Optional[List[Callback]] = None,
    verbose: int = 1
) -> Tuple[keras.Model, keras.callbacks.History]:
    """
    Train a neural network with proper configuration.
    
    Parameters:
    -----------
    model : keras.Model
        Compiled Keras model
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    X_val : np.ndarray
        Validation features
    y_val : np.ndarray
        Validation labels
    class_weight_strategy : str
        'balanced', 'custom', or None
    batch_size : int
        Batch size for training
    epochs : int
        Maximum epochs
    callbacks : List[Callback]
        Keras callbacks
    verbose : int
        Verbosity level
        
    Returns:
    --------
    Tuple[keras.Model, keras.callbacks.History]
        Trained model and training history
    """
    # Compute class weights
    class_weights = get_class_weights(y_train, class_weight_strategy)
    
    # Train model
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose
    )
    
    training_time = time.time() - start_time
    
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Epochs trained: {len(history.history['loss'])}")
    
    return model, history


def compile_model(
    model: keras.Model,
    learning_rate: float = 0.001,
    optimizer: str = 'adam',
    loss: str = 'binary_crossentropy',
    metrics: Optional[List] = None
) -> keras.Model:
    """
    Compile a Keras model with standard configuration.
    
    Parameters:
    -----------
    model : keras.Model
        Uncompiled Keras model
    learning_rate : float
        Learning rate for optimizer
    optimizer : str
        Optimizer name ('adam', 'rmsprop', 'sgd')
    loss : str
        Loss function
    metrics : List
        List of metrics
        
    Returns:
    --------
    keras.Model
        Compiled model
    """
    if metrics is None:
        metrics = [
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
            keras.metrics.AUC(name='pr_auc', curve='PR')
        ]
    
    # Configure optimizer
    if optimizer.lower() == 'adam':
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer.lower() == 'rmsprop':
        opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer.lower() == 'sgd':
        opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")
    
    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=metrics
    )
    
    return model


def log_experiment(
    experiment_data: Dict,
    log_file: str = None
):
    """
    Log experiment results to CSV.
    
    Parameters:
    -----------
    experiment_data : Dict
        Dictionary with experiment information
    log_file : str
        Path to CSV file
    """
    if log_file is None:
        log_file = config.NN_EXPERIMENTS_LOG
    
    # Create directory if needed
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing log or create new
    if log_path.exists():
        df = pd.read_csv(log_file)
        df = pd.concat([df, pd.DataFrame([experiment_data])], ignore_index=True)
    else:
        df = pd.DataFrame([experiment_data])
    
    # Save
    df.to_csv(log_file, index=False)
    print(f"Experiment logged to {log_file}")


def create_experiment_record(
    model_name: str,
    architecture: List[int],
    hyperparameters: Dict,
    metrics: Dict,
    training_info: Dict,
    notebook: str = None
) -> Dict:
    """
    Create a standardized experiment record.
    
    Parameters:
    -----------
    model_name : str
        Model identifier
    architecture : List[int]
        Hidden layer sizes
    hyperparameters : Dict
        Hyperparameters used
    metrics : Dict
        Performance metrics
    training_info : Dict
        Training metadata
    notebook : str
        Source notebook name
        
    Returns:
    --------
    Dict
        Experiment record
    """
    from src.nn_architectures import architecture_to_string
    
    record = {
        'experiment_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'date_created': datetime.now().isoformat(),
        'notebook': notebook or 'unknown',
        'model_name': model_name,
        'architecture': architecture_to_string(architecture),
        'layers': json.dumps(architecture),
        'random_seed': config.RANDOM_SEED,
    }
    
    # Add hyperparameters
    record.update({
        'dropout_rate': hyperparameters.get('dropout_rate', 0.0),
        'l2_reg': hyperparameters.get('l2_reg', 0.0),
        'batch_norm': hyperparameters.get('batch_norm', False),
        'class_weight_strategy': hyperparameters.get('class_weight_strategy', 'balanced'),
        'learning_rate': hyperparameters.get('learning_rate', 0.001),
        'batch_size': hyperparameters.get('batch_size', 64),
        'optimizer': hyperparameters.get('optimizer', 'adam'),
    })
    
    # Add metrics
    record.update({
        'precision_fraud': metrics.get('precision_fraud'),
        'recall_fraud': metrics.get('recall_fraud'),
        'f1_fraud': metrics.get('f1_fraud'),
        'pr_auc': metrics.get('pr_auc'),
        'roc_auc': metrics.get('roc_auc'),
        'accuracy': metrics.get('accuracy'),
        'confusion_matrix': json.dumps(metrics.get('confusion_matrix', [])),
    })
    
    # Add training info
    record.update({
        'epochs_trained': training_info.get('epochs_trained'),
        'training_time_seconds': training_info.get('training_time'),
        'inference_time_seconds': training_info.get('inference_time'),
        'val_loss': training_info.get('val_loss'),
        'train_loss': training_info.get('train_loss'),
    })
    
    # Add notes
    record['notes'] = hyperparameters.get('notes', '')
    
    return record


def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss for addressing class imbalance.
    
    Parameters:
    -----------
    gamma : float
        Focusing parameter (higher = more focus on hard examples)
    alpha : float
        Balancing parameter
        
    Returns:
    --------
    function
        Loss function
    """
    def focal_loss_fixed(y_true, y_pred):
        epsilon = keras.backend.epsilon()
        y_pred = keras.backend.clip(y_pred, epsilon, 1. - epsilon)
        
        cross_entropy = -y_true * keras.backend.log(y_pred)
        weight = alpha * y_true * keras.backend.pow((1 - y_pred), gamma)
        
        loss = weight * cross_entropy
        return keras.backend.mean(keras.backend.sum(loss, axis=1))
    
    return focal_loss_fixed


if __name__ == "__main__":
    print("Neural Network Training Utilities Module")
    print("=" * 50)
    
    # Example: Create experiment record
    example_record = create_experiment_record(
        model_name="MLP_Medium_Dropout",
        architecture=[128, 64, 32],
        hyperparameters={
            'dropout_rate': 0.3,
            'l2_reg': 0.01,
            'batch_norm': False,
            'class_weight_strategy': 'balanced',
            'learning_rate': 0.001,
            'batch_size': 64,
        },
        metrics={
            'precision_fraud': 0.75,
            'recall_fraud': 0.80,
            'f1_fraud': 0.77,
            'pr_auc': 0.82,
            'roc_auc': 0.95,
            'accuracy': 0.98,
            'confusion_matrix': [[9800, 20], [40, 160]],
        },
        training_info={
            'epochs_trained': 45,
            'training_time': 123.4,
            'inference_time': 0.5,
            'val_loss': 0.15,
            'train_loss': 0.12,
        },
        notebook='03_neural_network_architectures'
    )
    
    print("\nExample experiment record:")
    for key, value in example_record.items():
        print(f"  {key}: {value}")
    
    print("\n✓ Training utilities loaded successfully!")
