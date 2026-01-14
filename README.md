# XGBoost Training Module with Optuna Hyperparameter Optimization

This module provides a complete pipeline for training XGBoost models on molecular data using multiple feature types and Optuna for hyperparameter optimization.

## Features

- **Multiple Feature Types**: Combines reduced Mordred descriptors, Morgan fingerprints, RDKit fingerprints, MACCS keys, and MAP4 fingerprints
- **Optuna Optimization**: Uses cross-validation for robust hyperparameter selection
- **Comprehensive Evaluation**: Includes test set evaluation and cross-validation
- **Feature Importance**: Extracts and ranks feature importance
- **Model Persistence**: Saves trained models and results

## Installation

### Option 1: Conda Environment (Recommended)

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate xgboost_training

# Or use the setup script (Linux/Mac)
./setup_conda.sh

# Or use the setup script (Windows)
setup_conda.bat
```

### Option 2: Pip Installation

```bash
pip install -r requirements.txt
```

**Note**: Some packages (like `rdkit-pypi` and `mordred`) may be easier to install via conda.

## Usage

### Basic Usage

```bash
python train_model.py \
    --data ../train_val_features.csv \
    --n_trials 100 \
    --cv_folds 5 \
    --metric r2 \
    --output_dir results/
```

### Command Line Arguments

- `--data`: Path to training data CSV file (default: `../train_val_features.csv`)
- `--n_trials`: Number of Optuna trials (default: 100)
- `--cv_folds`: Number of CV folds (default: 5)
- `--metric`: Metric to optimize - `r2`, `mae`, or `rmse` (default: `r2`)
- `--output_dir`: Output directory for results (default: `results`)
- `--reduced_features`: Path to reduced Mordred features JSON (default: `../reduced_mordred_features.json`)
- `--include_map4`: Include MAP4 fingerprints (default: True)
- `--map4_dimensions`: MAP4 fingerprint dimensions (default: 1024)
- `--test_size`: Test set size (default: 0.2)
- `--random_state`: Random state for reproducibility (default: 42)

## Module Structure

### `featurization.py`
Calculates and combines all molecular features:
- Reduced Mordred descriptors
- Morgan fingerprints (radius=2, 2048 bits)
- RDKit fingerprints (Atom Pair, 2048 bits)
- MACCS keys (167 bits)
- MAP4 fingerprints (configurable dimensions)

### `optuna_tuning.py`
Optuna hyperparameter optimization with cross-validation:
- Defines hyperparameter search space
- Uses CV scores for robust optimization
- Supports R², MAE, and RMSE metrics

### `xgboost_trainer.py`
XGBoost model training and evaluation:
- Model training with early stopping
- Model evaluation (R², MAE, RMSE)
- Cross-validation
- Model persistence (save/load)
- Feature importance extraction

### `train_model.py`
Main orchestration script that:
1. Loads data
2. Calculates features
3. Splits data
4. Runs Optuna optimization
5. Trains final model
6. Evaluates on test set
7. Runs cross-validation
8. Extracts feature importance
9. Saves model and results

## Output Files

The training pipeline generates the following files in the output directory:

- `best_model.pkl`: Trained XGBoost model
- `best_hyperparameters.json`: Optimal hyperparameters from Optuna
- `training_results.json`: Complete training results and metrics
- `feature_importance.csv`: Feature importance rankings

## Example

```bash
# Run with default settings
python train_model.py

# Run with custom settings
python train_model.py \
    --data ../train_val_features.csv \
    --n_trials 200 \
    --cv_folds 10 \
    --metric mae \
    --output_dir my_results \
    --map4_dimensions 2048
```

## Dependencies

- pandas
- numpy
- scikit-learn
- xgboost
- optuna
- rdkit-pypi
- mordred
- map4

## Notes

- Ensure `reduced_mordred_features.json` exists in the parent directory (or specify path with `--reduced_features`)
- MAP4 requires the `map4` package (install with `pip install map4`)
- The pipeline handles missing values by filling with 0
- Early stopping is used during training to prevent overfitting

