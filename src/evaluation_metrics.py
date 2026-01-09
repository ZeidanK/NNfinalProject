"""
Evaluation Metrics for Fraud Detection

This module provides comprehensive evaluation metrics specifically
designed for imbalanced binary classification (fraud detection).
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve
)


def compute_fraud_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for fraud detection.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels (binary)
    y_pred_proba : np.ndarray, optional
        Predicted probabilities (for AUC metrics)
        
    Returns:
    --------
    Dict[str, float]
        Dictionary of metrics
    """
    metrics = {}
    
    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)
    
    # Fraud class metrics (primary)
    metrics['precision_fraud'] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    metrics['recall_fraud'] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    metrics['f1_fraud'] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    # Legitimate class metrics (secondary)
    metrics['precision_legitimate'] = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    metrics['recall_legitimate'] = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    metrics['f1_legitimate'] = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    
    # Overall accuracy (less important for imbalanced data)
    metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
    
    # AUC metrics (if probabilities provided)
    if y_pred_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['pr_auc'] = average_precision_score(y_true, y_pred_proba)
        except:
            metrics['roc_auc'] = np.nan
            metrics['pr_auc'] = np.nan
    
    # Additional derived metrics
    metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    metric: str = 'f1',
    beta: float = 1.0
) -> Tuple[float, float]:
    """
    Find optimal classification threshold.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred_proba : np.ndarray
        Predicted probabilities
    metric : str
        Metric to optimize ('f1', 'f_beta', 'youden')
    beta : float
        Beta value for F-beta score (only used if metric='f_beta')
        
    Returns:
    --------
    Tuple[float, float]
        (optimal_threshold, best_metric_value)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    if metric == 'f1':
        # F1 score
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        optimal_idx = np.argmax(f1_scores[:-1])  # Exclude last element
        best_metric = f1_scores[optimal_idx]
        
    elif metric == 'f_beta':
        # F-beta score
        f_beta_scores = (1 + beta**2) * (precisions * recalls) / (beta**2 * precisions + recalls + 1e-10)
        optimal_idx = np.argmax(f_beta_scores[:-1])
        best_metric = f_beta_scores[optimal_idx]
        
    elif metric == 'youden':
        # Youden's J statistic
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = roc_thresholds[optimal_idx]
        best_metric = j_scores[optimal_idx]
        return optimal_threshold, best_metric
        
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold, best_metric


def print_classification_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    dataset_name: str = "Dataset"
):
    """
    Print a comprehensive classification summary.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_pred_proba : np.ndarray, optional
        Predicted probabilities
    dataset_name : str
        Name of the dataset (for display)
    """
    print(f"\n{'='*70}")
    print(f"Classification Summary - {dataset_name}")
    print(f"{'='*70}")
    
    # Compute metrics
    metrics = compute_fraud_metrics(y_true, y_pred, y_pred_proba)
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    print(f"                 Predicted Legitimate    Predicted Fraud")
    print(f"Actual Legitimate    {metrics['true_negatives']:>10}        {metrics['false_positives']:>10}")
    print(f"Actual Fraud         {metrics['false_negatives']:>10}        {metrics['true_positives']:>10}")
    
    # Primary Metrics (Fraud Class)
    print("\n" + "="*70)
    print("PRIMARY METRICS (Fraud Class):")
    print("="*70)
    print(f"Precision (Fraud): {metrics['precision_fraud']:.4f}")
    print(f"Recall (Fraud):    {metrics['recall_fraud']:.4f}")
    print(f"F1-Score (Fraud):  {metrics['f1_fraud']:.4f}")
    
    if y_pred_proba is not None:
        print(f"\nPR-AUC:            {metrics['pr_auc']:.4f}  ← Best for imbalanced data")
        print(f"ROC-AUC:           {metrics['roc_auc']:.4f}")
    
    # Secondary Metrics
    print("\n" + "="*70)
    print("SECONDARY METRICS:")
    print("="*70)
    print(f"Accuracy:          {metrics['accuracy']:.4f}  ← Misleading for imbalanced data")
    print(f"FPR:               {metrics['false_positive_rate']:.4f}")
    print(f"FNR:               {metrics['false_negative_rate']:.4f}")
    
    # Business Interpretation
    print("\n" + "="*70)
    print("BUSINESS INTERPRETATION:")
    print("="*70)
    total_fraud = metrics['true_positives'] + metrics['false_negatives']
    total_legit = metrics['true_negatives'] + metrics['false_positives']
    
    print(f"Total Fraud Cases:       {total_fraud:>6}")
    print(f"Fraud Detected:          {metrics['true_positives']:>6} ({metrics['recall_fraud']*100:.1f}%)")
    print(f"Fraud Missed:            {metrics['false_negatives']:>6} ({metrics['false_negative_rate']*100:.1f}%)")
    print(f"\nTotal Legitimate Cases:  {total_legit:>6}")
    print(f"False Alarms:            {metrics['false_positives']:>6} ({metrics['false_positive_rate']*100:.1f}%)")
    
    print(f"\n{'='*70}\n")


def get_classification_report_df(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """
    Get classification report as DataFrame.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
        
    Returns:
    --------
    pd.DataFrame
        Classification report
    """
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=['Legitimate', 'Fraud'],
        output_dict=True
    )
    
    df = pd.DataFrame(report).transpose()
    return df


def compute_business_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_fn: float = 100.0,
    cost_fp: float = 1.0
) -> Dict[str, float]:
    """
    Compute business cost of predictions.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    cost_fn : float
        Cost of false negative (missed fraud)
    cost_fp : float
        Cost of false positive (false alarm)
        
    Returns:
    --------
    Dict[str, float]
        Cost analysis
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    total_cost = (fn * cost_fn) + (fp * cost_fp)
    
    # Cost if we predicted all legitimate (baseline)
    baseline_cost = np.sum(y_true == 1) * cost_fn
    
    # Cost savings
    cost_savings = baseline_cost - total_cost
    savings_percentage = (cost_savings / baseline_cost * 100) if baseline_cost > 0 else 0
    
    return {
        'false_negative_cost': fn * cost_fn,
        'false_positive_cost': fp * cost_fp,
        'total_cost': total_cost,
        'baseline_cost': baseline_cost,
        'cost_savings': cost_savings,
        'savings_percentage': savings_percentage,
        'cost_per_fn': cost_fn,
        'cost_per_fp': cost_fp
    }


def compare_models(
    models_dict: Dict[str, Dict],
    metric: str = 'f1_fraud',
    ascending: bool = False
) -> pd.DataFrame:
    """
    Compare multiple models by metrics.
    
    Parameters:
    -----------
    models_dict : Dict[str, Dict]
        Dictionary of {model_name: metrics_dict}
    metric : str
        Metric to sort by
    ascending : bool
        Sort order
        
    Returns:
    --------
    pd.DataFrame
        Comparison table
    """
    df = pd.DataFrame(models_dict).T
    
    if metric in df.columns:
        df = df.sort_values(by=metric, ascending=ascending)
    
    return df


def get_error_samples(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    feature_names: list,
    error_type: str = 'false_positive',
    n_samples: int = 10
) -> pd.DataFrame:
    """
    Extract samples of specific error types for analysis.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_pred_proba : np.ndarray
        Predicted probabilities
    feature_names : list
        Feature names
    error_type : str
        'false_positive' or 'false_negative'
    n_samples : int
        Number of samples to return
        
    Returns:
    --------
    pd.DataFrame
        Error samples with features and predictions
    """
    if error_type == 'false_positive':
        # Legitimate flagged as fraud
        error_mask = (y_true == 0) & (y_pred == 1)
    elif error_type == 'false_negative':
        # Fraud missed
        error_mask = (y_true == 1) & (y_pred == 0)
    else:
        raise ValueError(f"Unknown error type: {error_type}")
    
    error_indices = np.where(error_mask)[0]
    
    if len(error_indices) == 0:
        print(f"No {error_type} errors found!")
        return pd.DataFrame()
    
    # Sample random errors
    sample_size = min(n_samples, len(error_indices))
    sampled_indices = np.random.choice(error_indices, size=sample_size, replace=False)
    
    # Create DataFrame
    df = pd.DataFrame(X[sampled_indices], columns=feature_names)
    df['true_label'] = y_true[sampled_indices]
    df['predicted_label'] = y_pred[sampled_indices]
    df['predicted_probability'] = y_pred_proba[sampled_indices]
    df['error_type'] = error_type
    
    return df


if __name__ == "__main__":
    print("Evaluation Metrics Module")
    print("=" * 50)
    
    # Example usage with synthetic data
    np.random.seed(42)
    y_true = np.array([0]*990 + [1]*10)  # Imbalanced
    y_pred_proba = np.random.rand(1000)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    print("\nExample: Computing fraud detection metrics")
    metrics = compute_fraud_metrics(y_true, y_pred, y_pred_proba)
    
    print(f"\nPrecision (Fraud): {metrics['precision_fraud']:.4f}")
    print(f"Recall (Fraud):    {metrics['recall_fraud']:.4f}")
    print(f"F1-Score (Fraud):  {metrics['f1_fraud']:.4f}")
    print(f"PR-AUC:            {metrics['pr_auc']:.4f}")
    
    # Find optimal threshold
    optimal_thresh, best_f1 = find_optimal_threshold(y_true, y_pred_proba, metric='f1')
    print(f"\nOptimal Threshold: {optimal_thresh:.4f} (F1={best_f1:.4f})")
    
    print("\n✓ Evaluation metrics module loaded successfully!")
