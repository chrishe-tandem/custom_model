"""
XGBoost model training, evaluation, and persistence module.
"""

import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
import pickle
import json
import os


def train_xgboost(X_train, y_train, X_val, y_val, params, 
                 early_stopping_rounds=50, verbose=True):
    """
    Train XGBoost model with early stopping.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        params: XGBoost hyperparameters
        early_stopping_rounds: Early stopping rounds (default: 50)
        verbose: Whether to print training progress (default: True)
        
    Returns:
        Trained model and training history
    """
    # Handle NaN values
    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)
    
    # Create model
    # Handle early_stopping_rounds if present in params
    model_params = params.copy()
    if 'early_stopping_rounds' in model_params:
        early_stopping_rounds = model_params.pop('early_stopping_rounds')
    else:
        early_stopping_rounds = None
    
    model = xgb.XGBRegressor(**model_params)
    
    # Train with early stopping
    # Note: In XGBoost 2.0+, early_stopping_rounds is passed in __init__ or via callbacks
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=verbose
    )
    
    # Get training history
    try:
        evals_result = model.evals_result()
        train_rmse = evals_result.get('validation_0', {}).get('rmse', [])
    except:
        train_rmse = []
    
    try:
        best_iter = model.best_iteration
        n_estimators = best_iter + 1 if best_iter is not None else params.get('n_estimators', 100)
    except:
        n_estimators = params.get('n_estimators', 100)
    
    history = {
        'train_rmse': train_rmse,
        'n_estimators': n_estimators
    }
    
    return model, history


def evaluate_model(model, X, y):
    """
    Evaluate model performance.
    
    Args:
        model: Trained XGBoost model
        X: Features
        y: True targets
        
    Returns:
        Dictionary with R², MAE, RMSE
    """
    X = X.fillna(0)
    y_pred = model.predict(X)
    
    metrics = {
        'r2': r2_score(y, y_pred),
        'mae': mean_absolute_error(y, y_pred),
        'rmse': np.sqrt(mean_squared_error(y, y_pred))
    }
    
    return metrics


def cross_validate_model(X, y, params, cv_folds=5, random_state=42):
    """
    Run cross-validation on model.
    
    Args:
        X: Features
        y: Targets
        params: XGBoost hyperparameters
        cv_folds: Number of CV folds (default: 5)
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary with mean and std of CV scores
    """
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    r2_scores = []
    mae_scores = []
    rmse_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train model
        model, _ = train_xgboost(
            X_train_fold, y_train_fold,
            X_val_fold, y_val_fold,
            params,
            verbose=False
        )
        
        # Evaluate
        metrics = evaluate_model(model, X_val_fold, y_val_fold)
        r2_scores.append(metrics['r2'])
        mae_scores.append(metrics['mae'])
        rmse_scores.append(metrics['rmse'])
        
        print(f"  Fold {fold}/{cv_folds}: R²={metrics['r2']:.4f}, MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}")
    
    results = {
        'r2_mean': np.mean(r2_scores),
        'r2_std': np.std(r2_scores),
        'mae_mean': np.mean(mae_scores),
        'mae_std': np.std(mae_scores),
        'rmse_mean': np.mean(rmse_scores),
        'rmse_std': np.std(rmse_scores),
        'r2_scores': r2_scores,
        'mae_scores': mae_scores,
        'rmse_scores': rmse_scores
    }
    
    return results


def save_model(model, filepath):
    """
    Save trained model to file.
    
    Args:
        model: Trained XGBoost model
        filepath: Path to save model
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """
    Load saved model from file.
    
    Args:
        filepath: Path to saved model
        
    Returns:
        Loaded XGBoost model
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Model loaded from {filepath}")
    return model


def get_feature_importance(model, feature_names=None, importance_type='gain', top_n=50):
    """
    Extract and rank feature importance.
    
    Args:
        model: Trained XGBoost model
        feature_names: List of feature names (optional)
        importance_type: Type of importance ('gain', 'weight', 'cover')
        top_n: Number of top features to return (default: 50)
        
    Returns:
        DataFrame with feature importance rankings
    """
    # Get feature importance
    importance = model.get_booster().get_score(importance_type=importance_type)
    
    # Convert to DataFrame
    if feature_names is None:
        # Use feature indices
        df_importance = pd.DataFrame({
            'feature': [f'feature_{i}' for i in range(len(importance))],
            'importance': [importance.get(f'f{i}', 0) for i in range(len(importance))]
        })
    else:
        # Use feature names
        df_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': [importance.get(f'f{i}', 0) for i in range(len(feature_names))]
        })
    
    # Sort by importance
    df_importance = df_importance.sort_values('importance', ascending=False)
    
    # Get top N
    df_importance_top = df_importance.head(top_n)
    
    return df_importance, df_importance_top

