"""
model_training.py
-----------------
Loads preprocessed data, trains a Logistic Regression baseline and a
LightGBM classifier, reports test accuracy, and saves the trained
models to the models/ directory.
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.metrics import accuracy_score


def load_processed_data():
    """Load the processed train and test sets."""
    processed_dir = os.path.join('data', 'processed')

    X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
    X_test  = pd.read_csv(os.path.join(processed_dir, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(processed_dir, 'y_train.csv')).squeeze('columns')
    y_test  = pd.read_csv(os.path.join(processed_dir, 'y_test.csv')).squeeze('columns')

    with open(os.path.join(processed_dir, 'feature_names.txt'), 'r') as f:
        feature_names = f.read().splitlines()

    return X_train, X_test, y_train, y_test, feature_names


def train_baseline_model(X_train, y_train):
    """Train a baseline Logistic Regression model."""
    print("  Training Baseline Model (Logistic Regression)...")
    model = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
    model.fit(X_train, y_train)
    return model


def train_lightgbm_model(X_train, y_train):
    """Train a LightGBM Classifier."""
    print("  Training LightGBM Model...")
    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    processed_check = os.path.join('data', 'processed', 'X_train.csv')

    if not os.path.exists(processed_check):
        print("Error: Processed data not found.")
        print("Please run 'python src/data_preprocessing.py' first.")
    else:
        X_train, X_test, y_train, y_test, feature_names = load_processed_data()
        print(f"Loaded data: {X_train.shape[0]} train / {X_test.shape[0]} test samples")

        # 1. Baseline Model
        baseline_model = train_baseline_model(X_train, y_train)
        y_pred_base_test = baseline_model.predict(X_test)
        print(f"  Baseline Test Accuracy: {accuracy_score(y_test, y_pred_base_test):.4f}")

        # 2. LightGBM Model
        lgb_model = train_lightgbm_model(X_train, y_train)
        y_pred_lgb_test = lgb_model.predict(X_test)
        print(f"  LightGBM Test Accuracy: {accuracy_score(y_test, y_pred_lgb_test):.4f}")

        # Save Models
        os.makedirs('models', exist_ok=True)
        joblib.dump(baseline_model, os.path.join('models', 'baseline_model.pkl'))
        joblib.dump(lgb_model, os.path.join('models', 'lightgbm_model.pkl'))

        print("\nModels trained and saved to models/ directory.")
