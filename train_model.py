"""
Main training script for XGBoost model with Optuna hyperparameter optimization.

Combines all molecular features and trains XGBoost model with cross-validation.
"""

import pandas as pd
import numpy as np
import argparse
import json
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import local modules
from featurization import calculate_all_features
from optuna_tuning import optimize_hyperparameters
from xgboost_trainer import (
    train_xgboost, evaluate_model, cross_validate_model,
    save_model, get_feature_importance
)


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train XGBoost model with Optuna optimization')
    parser.add_argument('--data', type=str, default='../train_val_features.csv',
                       help='Path to training data CSV file')
    parser.add_argument('--n_trials', type=int, default=100,
                       help='Number of Optuna trials (default: 100)')
    parser.add_argument('--cv_folds', type=int, default=5,
                       help='Number of CV folds (default: 5)')
    parser.add_argument('--metric', type=str, default='r2', choices=['r2', 'mae', 'rmse'],
                       help='Metric to optimize (default: r2)')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results (default: results)')
    parser.add_argument('--reduced_features', type=str, 
                       default='../reduced_mordred_features.json',
                       help='Path to reduced Mordred features JSON file')
    parser.add_argument('--include_map4', action='store_true', default=True,
                       help='Include MAP4 fingerprints (default: True)')
    parser.add_argument('--map4_dimensions', type=int, default=1024,
                       help='MAP4 fingerprint dimensions (default: 1024)')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("XGBoost Training Pipeline")
    print("=" * 80)
    print(f"Data: {args.data}")
    print(f"Optuna trials: {args.n_trials}")
    print(f"CV folds: {args.cv_folds}")
    print(f"Metric: {args.metric}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)
    
    # Step 1: Load data
    print("\n1. Loading data...")
    try:
        df = pd.read_csv(args.data)
        print(f"   Loaded {len(df)} samples")
    except FileNotFoundError:
        print(f"Error: Data file not found: {args.data}")
        return
    
    # Extract SMILES and labels
    if 'smiles' not in df.columns or 'label' not in df.columns:
        print("Error: Data must contain 'smiles' and 'label' columns")
        return
    
    smiles_list = df['smiles'].dropna().tolist()
    y = df['label'].dropna()
    
    # Align indices
    valid_indices = df.dropna(subset=['smiles', 'label']).index
    smiles_list = [df.loc[i, 'smiles'] for i in valid_indices]
    y = df.loc[valid_indices, 'label']
    
    print(f"   Valid samples: {len(smiles_list)}")
    print(f"   Target range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"   Target mean: {y.mean():.2f} ± {y.std():.2f}")
    
    # Step 2: Calculate features
    print("\n2. Calculating features...")
    try:
        X = calculate_all_features(
            smiles_list,
            reduced_features_path=args.reduced_features,
            include_map4=args.include_map4,
            map4_dimensions=args.map4_dimensions
        )
        print(f"   Features calculated: {X.shape[1]}")
    except Exception as e:
        print(f"Error calculating features: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Split data
    print("\n3. Splitting data...")
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    
    print(f"   Train set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    # Step 4: Optuna optimization
    print("\n4. Optuna hyperparameter optimization...")
    try:
        optuna_results = optimize_hyperparameters(
            X_train, y_train,
            n_trials=args.n_trials,
            cv_folds=args.cv_folds,
            metric=args.metric,
            random_state=args.random_state
        )
        
        best_params = optuna_results['best_params']
        best_value = optuna_results['best_value']
        
        # Save best hyperparameters
        hyperparams_path = os.path.join(args.output_dir, 'best_hyperparameters.json')
        with open(hyperparams_path, 'w') as f:
            json.dump(best_params, f, indent=2)
        print(f"   Best hyperparameters saved to {hyperparams_path}")
        
    except Exception as e:
        print(f"Error in Optuna optimization: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 5: Train final model
    print("\n5. Training final model...")
    try:
        # Split train into train/val for early stopping
        X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
            X_train, y_train, test_size=0.2, random_state=args.random_state
        )
        
        # Add early stopping rounds to params
        final_params = best_params.copy()
        final_params['early_stopping_rounds'] = 50
        
        model, history = train_xgboost(
            X_train_final, y_train_final,
            X_val_final, y_val_final,
            final_params,
            verbose=True
        )
        
        print(f"   Model trained with {history['n_estimators']} estimators")
        
    except Exception as e:
        print(f"Error training model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 6: Evaluate on test set
    print("\n6. Evaluating on test set...")
    test_metrics = evaluate_model(model, X_test, y_test)
    print(f"   Test R²: {test_metrics['r2']:.4f}")
    print(f"   Test MAE: {test_metrics['mae']:.4f}")
    print(f"   Test RMSE: {test_metrics['rmse']:.4f}")
    
    # Step 7: Cross-validation evaluation
    print("\n7. Cross-validation evaluation...")
    cv_results = cross_validate_model(
        X_train, y_train,
        best_params,
        cv_folds=args.cv_folds,
        random_state=args.random_state
    )
    
    print(f"\n   CV Results:")
    print(f"     R²: {cv_results['r2_mean']:.4f} ± {cv_results['r2_std']:.4f}")
    print(f"     MAE: {cv_results['mae_mean']:.4f} ± {cv_results['mae_std']:.4f}")
    print(f"     RMSE: {cv_results['rmse_mean']:.4f} ± {cv_results['rmse_std']:.4f}")
    
    # Step 8: Feature importance
    print("\n8. Extracting feature importance...")
    feature_names = X.columns.tolist()
    df_importance, df_importance_top = get_feature_importance(
        model, feature_names, top_n=50
    )
    
    importance_path = os.path.join(args.output_dir, 'feature_importance.csv')
    df_importance.to_csv(importance_path, index=False)
    print(f"   Feature importance saved to {importance_path}")
    
    print(f"\n   Top 10 features:")
    for idx, row in df_importance_top.head(10).iterrows():
        print(f"     {row['feature']}: {row['importance']:.4f}")
    
    # Step 9: Save model
    print("\n9. Saving model...")
    model_path = os.path.join(args.output_dir, 'best_model.pkl')
    save_model(model, model_path)
    
    # Step 10: Save results
    print("\n10. Saving results...")
    results = {
        'timestamp': datetime.now().isoformat(),
        'data_file': args.data,
        'n_samples': len(smiles_list),
        'n_features': X.shape[1],
        'test_size': args.test_size,
        'n_trials': args.n_trials,
        'cv_folds': args.cv_folds,
        'metric': args.metric,
        'best_hyperparameters': best_params,
        'best_optuna_value': best_value,
        'test_metrics': test_metrics,
        'cv_metrics': {
            'r2_mean': cv_results['r2_mean'],
            'r2_std': cv_results['r2_std'],
            'mae_mean': cv_results['mae_mean'],
            'mae_std': cv_results['mae_std'],
            'rmse_mean': cv_results['rmse_mean'],
            'rmse_std': cv_results['rmse_std']
        },
        'model_info': {
            'n_estimators': history['n_estimators'],
            'model_path': model_path
        }
    }
    
    results_path = os.path.join(args.output_dir, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   Results saved to {results_path}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"\nBest hyperparameters: {hyperparams_path}")
    print(f"Model: {model_path}")
    print(f"Results: {results_path}")
    print(f"Feature importance: {importance_path}")
    print(f"\nFinal Performance:")
    print(f"  Test R²: {test_metrics['r2']:.4f}")
    print(f"  CV R²: {cv_results['r2_mean']:.4f} ± {cv_results['r2_std']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()

