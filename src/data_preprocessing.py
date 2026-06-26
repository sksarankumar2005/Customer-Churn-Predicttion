"""
data_preprocessing.py
---------------------
Loads the raw Telco Customer Churn dataset, cleans it, encodes categorical
features, splits into train/test sets, scales numerical features, and saves
all processed artifacts to data/processed/.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os


def load_and_clean_data(filepath):
    """Loads the dataset and handles missing/incorrect values.
    
    Args:
        filepath: Path to the raw CSV file.
    
    Returns:
        Cleaned pandas DataFrame.
    """
    df = pd.read_csv(filepath)
    
    # TotalCharges is stored as object (string) — convert to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Drop rows with missing TotalCharges (~11 rows where tenure=0)
    df.dropna(inplace=True)
    
    # Drop customerID — it's a unique identifier, not a feature
    df.drop('customerID', axis=1, inplace=True)
    
    return df


def preprocess_data(df):
    """Encodes categorical features, splits data, and scales numerical features.
    
    Args:
        df: Cleaned pandas DataFrame.
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names_list, scaler).
    """
    # Target variable encoding (Churn: Yes -> 1, No -> 0)
    df = df.copy()
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    y = df['Churn']
    X = df.drop('Churn', axis=1)
    
    # Identify numerical and categorical columns
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_cols = X.drop(columns=numerical_cols).columns.tolist()
    
    # One-Hot Encoding for categorical features (drop_first to avoid multicollinearity)
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Train-test split (80-20, stratified by target to preserve class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Cast numerical columns to float (tenure is int64, scaler returns float64)
    X_train[numerical_cols] = X_train[numerical_cols].astype(float)
    X_test[numerical_cols] = X_test[numerical_cols].astype(float)

    # Scale numerical features using StandardScaler
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    return X_train, X_test, y_train, y_test, X.columns.tolist(), scaler


if __name__ == "__main__":
    data_path = os.path.join('data', 'Telco_Customer_Churn.csv')
    
    if not os.path.exists(data_path):
        print(f"Error: '{data_path}' not found.")
        print("Please place the Telco Customer Churn CSV in the data/ directory.")
    else:
        print("Loading and cleaning data...")
        df = load_and_clean_data(data_path)
        print(f"  Dataset shape after cleaning: {df.shape}")
        
        print("Preprocessing data (encoding, splitting, scaling)...")
        X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_data(df)
        
        print(f"  Training set: {X_train.shape[0]} samples")
        print(f"  Test set:     {X_test.shape[0]} samples")
        print(f"  Features:     {X_train.shape[1]}")
        
        # Save processed data for downstream tasks
        output_dir = os.path.join('data', 'processed')
        os.makedirs(output_dir, exist_ok=True)
        
        X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
        X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
        y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
        y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
        
        with open(os.path.join(output_dir, 'feature_names.txt'), 'w') as f:
            f.write('\n'.join(feature_names))
        
        # Save the scaler for potential future inference
        joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
        
        print(f"\nData preprocessing complete. Files saved to {output_dir}/")
