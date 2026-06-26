"""
evaluation.py
-------------
Loads trained models and test data, computes comprehensive evaluation metrics
(accuracy, precision, recall, F1, ROC-AUC), generates confusion matrix and
feature importance visualizations, and saves them to notebooks/images/.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
import os


def load_data_and_models():
    """Load test data and trained models.
    
    Returns:
        Tuple of (X_test, y_test, baseline_model, lgb_model, feature_names).
    """
    processed_dir = os.path.join('data', 'processed')
    
    X_test = pd.read_csv(os.path.join(processed_dir, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(processed_dir, 'y_test.csv')).squeeze('columns')
    
    baseline_model = joblib.load(os.path.join('models', 'baseline_model.pkl'))
    lgb_model = joblib.load(os.path.join('models', 'lightgbm_model.pkl'))
    
    with open(os.path.join(processed_dir, 'feature_names.txt'), 'r') as f:
        feature_names = f.read().splitlines()
    
    return X_test, y_test, baseline_model, lgb_model, feature_names


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate and print comprehensive metrics for a given model.
    
    Args:
        model: Trained classifier with predict() and predict_proba().
        X_test: Test feature matrix.
        y_test: True test labels.
        model_name: Display name for the model.
    
    Returns:
        Dictionary with metric values.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall':    recall_score(y_test, y_pred),
        'f1':        f1_score(y_test, y_pred),
        'roc_auc':   roc_auc_score(y_test, y_prob),
    }
    
    print(f"--- {model_name} Performance ---")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print()
    
    return metrics, y_pred, y_prob


def plot_confusion_matrix(y_test, y_pred, model_name, save_dir):
    """Generate and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.title(f'Confusion Matrix — {model_name}', fontsize=14)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    filename = f"confusion_matrix_{model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  Saved: {filepath}")


def plot_roc_curves(models_data, save_dir):
    """Plot ROC curves for all models on a single figure."""
    plt.figure(figsize=(8, 6))
    
    for name, y_test, y_prob in models_data:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve Comparison', fontsize=14)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    filepath = os.path.join(save_dir, 'roc_curve_comparison.png')
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  Saved: {filepath}")


def plot_feature_importance(model, feature_names, save_dir, top_n=15):
    """Plot top feature importances for the LightGBM model."""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:top_n]
    
    plt.figure(figsize=(10, 7))
    sns.barplot(
        x=importance[indices],
        y=np.array(feature_names)[indices],
        hue=np.array(feature_names)[indices],
        palette='viridis',
        legend=False
    )
    plt.title(f'Top {top_n} Feature Importances (LightGBM)', fontsize=14)
    plt.xlabel('Relative Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    
    filepath = os.path.join(save_dir, 'feature_importance.png')
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  Saved: {filepath}")


if __name__ == "__main__":
    model_check = os.path.join('models', 'lightgbm_model.pkl')
    
    if not os.path.exists(model_check):
        print("Error: Models not found.")
        print("Please run 'python src/model_training.py' first.")
    else:
        X_test, y_test, baseline_model, lgb_model, feature_names = load_data_and_models()
        
        # --- Evaluate Both Models ---
        print("=" * 50)
        print("MODEL EVALUATION ON TEST SET")
        print("=" * 50 + "\n")
        
        base_metrics, base_pred, base_prob = evaluate_model(
            baseline_model, X_test, y_test, "Logistic Regression (Baseline)"
        )
        lgb_metrics, lgb_pred, lgb_prob = evaluate_model(
            lgb_model, X_test, y_test, "LightGBM"
        )
        
        # --- Detailed Classification Report ---
        print("=" * 50)
        print("DETAILED CLASSIFICATION REPORT (LightGBM)")
        print("=" * 50)
        print(classification_report(y_test, lgb_pred, target_names=['No Churn', 'Churn']))
        
        # --- Generate Visualizations ---
        save_dir = os.path.join('notebooks', 'images')
        os.makedirs(save_dir, exist_ok=True)
        
        print("Generating visualizations...")
        plot_confusion_matrix(y_test, base_pred, "Logistic Regression", save_dir)
        plot_confusion_matrix(y_test, lgb_pred, "LightGBM", save_dir)
        
        plot_roc_curves([
            ("Logistic Regression", y_test, base_prob),
            ("LightGBM", y_test, lgb_prob),
        ], save_dir)
        
        plot_feature_importance(lgb_model, feature_names, save_dir)
        
        print("\nEvaluation complete.")
