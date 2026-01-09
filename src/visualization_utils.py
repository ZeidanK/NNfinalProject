"""
Visualization Utilities for Neural Network Experiments

This module provides visualization functions for neural network training,
evaluation, and analysis in fraud detection tasks.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
import config


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = config.FIGURE_DPI


def plot_learning_curves(
    history: dict,
    metrics: List[str] = ['loss', 'pr_auc'],
    save_path: Optional[str] = None,
    title: str = "Learning Curves"
):
    """
    Plot training and validation learning curves.
    
    Parameters:
    -----------
    history : dict
        Training history from model.fit()
    metrics : List[str]
        Metrics to plot
    save_path : str, optional
        Path to save figure
    title : str
        Plot title
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(7*n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Plot training metric
        if metric in history:
            ax.plot(history[metric], label=f'Train {metric}', linewidth=2)
        
        # Plot validation metric
        val_metric = f'val_{metric}'
        if val_metric in history:
            ax.plot(history[val_metric], label=f'Val {metric}', linewidth=2, linestyle='--')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.set_title(f'{metric.upper()} over Epochs', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=config.FIGURE_DPI)
        print(f"Learning curves saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str] = ['Legitimate', 'Fraud'],
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix"
):
    """
    Plot confusion matrix with annotations.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    labels : List[str]
        Class labels
    save_path : str, optional
        Path to save figure
    title : str
        Plot title
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        ax=ax,
        annot_kws={'size': 14, 'weight': 'bold'}
    )
    
    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    
    # Add percentage annotations
    total = np.sum(cm)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            percentage = cm[i, j] / total * 100
            ax.text(
                j + 0.5, i + 0.7,
                f'({percentage:.2f}%)',
                ha='center',
                va='center',
                fontsize=10,
                color='gray'
            )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=config.FIGURE_DPI)
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Precision-Recall Curve"
):
    """
    Plot precision-recall curve.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred_proba : np.ndarray
        Predicted probabilities
    save_path : str, optional
        Path to save figure
    title : str
        Plot title
    """
    from sklearn.metrics import average_precision_score
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(recall, precision, linewidth=2, label=f'PR-AUC = {pr_auc:.4f}')
    ax.axhline(y=np.sum(y_true)/len(y_true), color='r', linestyle='--', 
               label='Baseline (Random)', linewidth=1.5)
    
    ax.set_xlabel('Recall (Fraud Class)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Precision (Fraud Class)', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=config.FIGURE_DPI)
        print(f"PR curve saved to {save_path}")
    
    plt.show()


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "ROC Curve"
):
    """
    Plot ROC curve.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred_proba : np.ndarray
        Predicted probabilities
    save_path : str, optional
        Path to save figure
    title : str
        Plot title
    """
    from sklearn.metrics import roc_auc_score
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, linewidth=2, label=f'ROC-AUC = {roc_auc:.4f}')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=1.5, label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=config.FIGURE_DPI)
        print(f"ROC curve saved to {save_path}")
    
    plt.show()


def plot_threshold_analysis(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Threshold Analysis"
):
    """
    Plot precision, recall, and F1 vs threshold.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred_proba : np.ndarray
        Predicted probabilities
    save_path : str, optional
        Path to save figure
    title : str
        Plot title
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    # Calculate F1 scores
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    # Find optimal threshold
    optimal_idx = np.argmax(f1_scores[:-1])
    optimal_threshold = thresholds[optimal_idx]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(thresholds, precision[:-1], label='Precision', linewidth=2)
    ax.plot(thresholds, recall[:-1], label='Recall', linewidth=2)
    ax.plot(thresholds, f1_scores[:-1], label='F1-Score', linewidth=2, linestyle='--')
    ax.axvline(x=optimal_threshold, color='red', linestyle=':', linewidth=2,
               label=f'Optimal Threshold = {optimal_threshold:.3f}')
    ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=1.5, 
               label='Default Threshold = 0.5', alpha=0.7)
    
    ax.set_xlabel('Threshold', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=config.FIGURE_DPI)
        print(f"Threshold analysis saved to {save_path}")
    
    plt.show()
    
    return optimal_threshold


def plot_model_comparison(
    results_df: pd.DataFrame,
    metric: str = 'f1_fraud',
    save_path: Optional[str] = None,
    title: str = "Model Comparison"
):
    """
    Plot bar chart comparing models by metric.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with model results (index=model names)
    metric : str
        Metric column to plot
    save_path : str, optional
        Path to save figure
    title : str
        Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    results_sorted = results_df.sort_values(by=metric, ascending=True)
    
    bars = ax.barh(
        results_sorted.index,
        results_sorted[metric],
        color=sns.color_palette("viridis", len(results_sorted))
    )
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width,
            bar.get_y() + bar.get_height()/2,
            f'{width:.4f}',
            ha='left',
            va='center',
            fontsize=10,
            fontweight='bold'
        )
    
    ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=13, fontweight='bold')
    ax.set_ylabel('Model', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=config.FIGURE_DPI)
        print(f"Model comparison saved to {save_path}")
    
    plt.show()


def plot_architecture_comparison(
    architectures_results: Dict[str, Dict],
    save_path: Optional[str] = None
):
    """
    Plot comprehensive architecture comparison.
    
    Parameters:
    -----------
    architectures_results : Dict[str, Dict]
        Dictionary of {architecture_name: metrics_dict}
    save_path : str, optional
        Path to save figure
    """
    metrics_to_plot = ['precision_fraud', 'recall_fraud', 'f1_fraud', 'pr_auc']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    df = pd.DataFrame(architectures_results).T
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        if metric in df.columns:
            df_sorted = df.sort_values(by=metric, ascending=False)
            
            bars = ax.bar(
                range(len(df_sorted)),
                df_sorted[metric],
                color=sns.color_palette("rocket", len(df_sorted))
            )
            
            ax.set_xticks(range(len(df_sorted)))
            ax.set_xticklabels(df_sorted.index, rotation=45, ha='right', fontsize=9)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            ax.set_title(f'{metric.replace("_", " ").title()} by Architecture', 
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    height,
                    f'{height:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=8
                )
    
    fig.suptitle('Neural Network Architecture Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=config.FIGURE_DPI)
        print(f"Architecture comparison saved to {save_path}")
    
    plt.show()


def plot_class_distribution(
    y: np.ndarray,
    labels: List[str] = ['Legitimate', 'Fraud'],
    save_path: Optional[str] = None,
    title: str = "Class Distribution"
):
    """
    Plot class distribution.
    
    Parameters:
    -----------
    y : np.ndarray
        Labels
    labels : List[str]
        Class names
    save_path : str, optional
        Path to save figure
    title : str
        Plot title
    """
    unique, counts = np.unique(y, return_counts=True)
    percentages = counts / len(y) * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart
    bars = ax1.bar(labels, counts, color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Count', fontsize=13, fontweight='bold')
    ax1.set_title('Class Counts', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add count labels
    for bar, count in zip(bars, counts):
        ax1.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height(),
            f'{count:,}',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )
    
    # Pie chart
    colors = ['#2ecc71', '#e74c3c']
    explode = (0, 0.1)  # Explode fraud slice
    ax2.pie(
        counts,
        labels=[f'{label}\n({pct:.2f}%)' for label, pct in zip(labels, percentages)],
        autopct='%d',
        colors=colors,
        explode=explode,
        shadow=True,
        startangle=90
    )
    ax2.set_title('Class Distribution', fontsize=14, fontweight='bold')
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=config.FIGURE_DPI)
        print(f"Class distribution saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    print("Visualization Utilities Module")
    print("=" * 50)
    
    # Example: Generate synthetic data and plot
    np.random.seed(42)
    y_true = np.array([0]*990 + [1]*10)
    y_pred_proba = np.random.rand(1000)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    print("\nExample: Plotting class distribution")
    plot_class_distribution(y_true, title="Example Class Distribution")
    
    print("\n✓ Visualization utilities module loaded successfully!")
