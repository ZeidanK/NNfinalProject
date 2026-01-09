"""
Neural Network Architecture Definitions

This module contains reusable MLP architecture builders for fraud detection.
All architectures are designed for binary classification on tabular data.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from typing import List, Optional
import config


def create_mlp(
    input_dim: int,
    hidden_layers: List[int],
    dropout_rate: float = 0.0,
    l2_reg: float = 0.0,
    use_batch_norm: bool = False,
    activation: str = 'relu',
    output_activation: str = 'sigmoid',
    name: str = 'MLP'
) -> keras.Model:
    """
    Create a Multi-Layer Perceptron for binary classification.
    
    Parameters:
    -----------
    input_dim : int
        Number of input features
    hidden_layers : List[int]
        Number of units in each hidden layer, e.g., [128, 64, 32]
    dropout_rate : float
        Dropout rate (0.0 = no dropout)
    l2_reg : float
        L2 regularization strength (0.0 = no regularization)
    use_batch_norm : bool
        Whether to use Batch Normalization
    activation : str
        Activation function for hidden layers
    output_activation : str
        Activation function for output layer
    name : str
        Model name
        
    Returns:
    --------
    keras.Model
        Compiled Keras model
    """
    model = keras.Sequential(name=name)
    
    # Input layer
    model.add(layers.Input(shape=(input_dim,)))
    
    # Hidden layers
    for i, units in enumerate(hidden_layers):
        # Dense layer with optional L2 regularization
        if l2_reg > 0:
            model.add(layers.Dense(
                units,
                activation=activation,
                kernel_regularizer=regularizers.l2(l2_reg),
                name=f'hidden_{i+1}'
            ))
        else:
            model.add(layers.Dense(
                units,
                activation=activation,
                name=f'hidden_{i+1}'
            ))
        
        # Batch Normalization (if enabled)
        if use_batch_norm:
            model.add(layers.BatchNormalization(name=f'bn_{i+1}'))
        
        # Dropout (if enabled)
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate, name=f'dropout_{i+1}'))
    
    # Output layer
    model.add(layers.Dense(1, activation=output_activation, name='output'))
    
    return model


def create_shallow_mlp(input_dim: int, **kwargs) -> keras.Model:
    """Create shallow MLP: [32] or [64, 32]"""
    hidden_layers = kwargs.get('hidden_layers', [64, 32])
    return create_mlp(input_dim, hidden_layers, name='Shallow_MLP', **kwargs)


def create_medium_mlp(input_dim: int, **kwargs) -> keras.Model:
    """Create medium MLP: [128, 64, 32] (recommended)"""
    hidden_layers = kwargs.get('hidden_layers', [128, 64, 32])
    return create_mlp(input_dim, hidden_layers, name='Medium_MLP', **kwargs)


def create_deep_mlp(input_dim: int, **kwargs) -> keras.Model:
    """Create deep MLP: [512, 256, 128, 64, 32, 16]"""
    hidden_layers = kwargs.get('hidden_layers', [512, 256, 128, 64, 32, 16])
    return create_mlp(input_dim, hidden_layers, name='Deep_MLP', **kwargs)


def create_wide_mlp(input_dim: int, **kwargs) -> keras.Model:
    """Create wide shallow MLP: [1024]"""
    hidden_layers = kwargs.get('hidden_layers', [1024])
    return create_mlp(input_dim, hidden_layers, name='Wide_MLP', **kwargs)


def create_narrow_deep_mlp(input_dim: int, **kwargs) -> keras.Model:
    """Create narrow deep MLP: [64, 64, 64, 64]"""
    hidden_layers = kwargs.get('hidden_layers', [64, 64, 64, 64])
    return create_mlp(input_dim, hidden_layers, name='Narrow_Deep_MLP', **kwargs)


def get_architecture_by_name(arch_name: str, input_dim: int, **kwargs) -> keras.Model:
    """
    Get architecture by name.
    
    Parameters:
    -----------
    arch_name : str
        One of: 'shallow', 'medium', 'deep', 'wide', 'narrow_deep'
    input_dim : int
        Number of input features
    **kwargs : dict
        Additional parameters for create_mlp
        
    Returns:
    --------
    keras.Model
    """
    architectures = {
        'shallow': create_shallow_mlp,
        'medium': create_medium_mlp,
        'deep': create_deep_mlp,
        'wide': create_wide_mlp,
        'narrow_deep': create_narrow_deep_mlp,
    }
    
    if arch_name not in architectures:
        raise ValueError(f"Unknown architecture: {arch_name}. "
                        f"Choose from: {list(architectures.keys())}")
    
    return architectures[arch_name](input_dim, **kwargs)


def get_model_summary_string(model: keras.Model) -> str:
    """
    Get model summary as string.
    
    Parameters:
    -----------
    model : keras.Model
        Keras model
        
    Returns:
    --------
    str
        Model summary
    """
    from io import StringIO
    import sys
    
    old_stdout = sys.stdout
    sys.stdout = buffer = StringIO()
    
    model.summary()
    
    sys.stdout = old_stdout
    summary = buffer.getvalue()
    
    return summary


def count_parameters(model: keras.Model) -> dict:
    """
    Count trainable and non-trainable parameters.
    
    Parameters:
    -----------
    model : keras.Model
        Keras model
        
    Returns:
    --------
    dict
        Dictionary with parameter counts
    """
    trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    
    return {
        'trainable': int(trainable),
        'non_trainable': int(non_trainable),
        'total': int(trainable + non_trainable)
    }


def architecture_to_string(hidden_layers: List[int]) -> str:
    """
    Convert architecture list to string representation.
    
    Parameters:
    -----------
    hidden_layers : List[int]
        List of hidden layer sizes
        
    Returns:
    --------
    str
        String representation, e.g., "128-64-32"
    """
    return "-".join(map(str, hidden_layers))


if __name__ == "__main__":
    # Example usage
    print("Neural Network Architectures Module")
    print("=" * 50)
    
    # Create example models
    input_features = 7
    
    print("\n1. Shallow MLP:")
    model = create_shallow_mlp(input_features)
    print(f"   Layers: {[64, 32]}")
    print(f"   Parameters: {count_parameters(model)['total']:,}")
    
    print("\n2. Medium MLP (Recommended):")
    model = create_medium_mlp(input_features, dropout_rate=0.3, l2_reg=0.01)
    print(f"   Layers: {[128, 64, 32]}")
    print(f"   Parameters: {count_parameters(model)['total']:,}")
    print(f"   Regularization: Dropout=0.3, L2=0.01")
    
    print("\n3. Deep MLP:")
    model = create_deep_mlp(input_features)
    print(f"   Layers: {[512, 256, 128, 64, 32, 16]}")
    print(f"   Parameters: {count_parameters(model)['total']:,}")
    
    print("\n✓ All architectures defined successfully!")
