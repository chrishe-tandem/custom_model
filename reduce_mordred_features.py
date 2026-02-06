"""
Script to reduce Mordred features based on correlation and enable faster featurization
by calculating only the selected subset of descriptors.
"""

import pandas as pd
import numpy as np
from mordred import Calculator, descriptors
from rdkit import Chem
from typing import List, Set, Dict
import pickle
import json


def calculate_correlation_reduction(df: pd.DataFrame, threshold: float = 0.7) -> Dict:
    """
    Calculate correlation matrix and identify redundant features.
    
    Args:
        df: DataFrame with mordred features
        threshold: Correlation threshold above which features are considered redundant
        
    Returns:
        Dictionary with features to keep, drop, and statistics
    """
    # Calculate correlation matrix
    corr_matrix = df.corr().abs()
    
    # Get upper triangle of correlation matrix
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Find features to drop (highly correlated with others)
    to_drop = []
    for column in upper.columns:
        if any(upper[column] > threshold):
            to_drop.append(column)
    
    # Features to keep
    to_keep = [col for col in df.columns if col not in to_drop]
    
    return {
        'features_to_keep': to_keep,
        'features_to_drop': to_drop,
        'original_count': len(df.columns),
        'reduced_count': len(to_keep),
        'reduction_ratio': len(to_keep) / len(df.columns),
        'correlation_matrix': corr_matrix
    }


def map_feature_names_to_descriptors(feature_names: List[str]) -> tuple:
    """
    Map mordred feature names (with 'mordred_' prefix) to actual descriptor objects.
    
    Uses a test calculation to get actual descriptor keys, which is more reliable
    than trying to match string representations.
    
    Args:
        feature_names: List of feature names (e.g., ['mordred_ABC', 'mordred_nAcid'])
        
    Returns:
        Tuple of (feature_to_descriptor dict, missing_descriptors list)
    """
    # Remove 'mordred_' prefix to get descriptor names
    descriptor_names = [name.replace('mordred_', '') for name in feature_names]
    
    # Get all available descriptors
    all_calc = Calculator(descriptors, ignore_3D=False)
    all_descriptors = all_calc.descriptors
    
    # Create a test molecule to see actual descriptor keys
    test_mol = Chem.MolFromSmiles("CCO")  # Simple test molecule
    if test_mol:
        test_results = all_calc(test_mol)
        # Create mapping from result keys to descriptor objects
        key_to_descriptor = {}
        for desc_obj, key in zip(all_descriptors, test_results.keys()):
            key_to_descriptor[key] = desc_obj
            # Also map without any prefix variations
            key_clean = str(key).replace('mordred.', '').replace('mordred_', '')
            key_to_descriptor[key_clean] = desc_obj
    
    # Map feature names to descriptor objects
    feature_to_descriptor = {}
    missing_descriptors = []
    
    for feature_name, desc_name in zip(feature_names, descriptor_names):
        found = False
        
        # Try exact match with various key formats
        for key_format in [desc_name, f"mordred.{desc_name}", f"mordred_{desc_name}"]:
            if key_format in key_to_descriptor:
                feature_to_descriptor[feature_name] = key_to_descriptor[key_format]
                found = True
                break
        
        # Try case-insensitive match
        if not found:
            desc_name_lower = desc_name.lower()
            for key, desc_obj in key_to_descriptor.items():
                if desc_name_lower in str(key).lower() or str(key).lower() in desc_name_lower:
                    feature_to_descriptor[feature_name] = desc_obj
                    found = True
                    break
        
        # Try searching by class name
        if not found:
            for desc_obj in all_descriptors:
                class_name = desc_obj.__class__.__name__
                if desc_name == class_name or desc_name in class_name:
                    feature_to_descriptor[feature_name] = desc_obj
                    found = True
                    break
        
        if not found:
            missing_descriptors.append(feature_name)
            print(f"Warning: Could not find descriptor for {feature_name} (looking for {desc_name})")
    
    if missing_descriptors:
        print(f"\nWarning: {len(missing_descriptors)} descriptors could not be automatically mapped.")
        print("You may need to manually map these descriptors.")
    
    return feature_to_descriptor, missing_descriptors


def create_reduced_calculator(feature_names: List[str]) -> Calculator:
    """
    Create a Mordred Calculator that calculates only the specified features.
    
    Args:
        feature_names: List of feature names to calculate (e.g., ['mordred_ABC', 'mordred_nAcid'])
        
    Returns:
        Calculator configured to calculate only the specified descriptors
    """
    feature_to_descriptor, missing = map_feature_names_to_descriptors(feature_names)
    
    if missing:
        print(f"Warning: {len(missing)} features could not be mapped to descriptors.")
        print(f"Calculator will include {len(feature_to_descriptor)} descriptors.")
    
    # Get list of descriptor objects
    selected_descriptors = list(feature_to_descriptor.values())
    
    # Create calculator with only selected descriptors
    calc = Calculator(selected_descriptors, ignore_3D=False)
    
    return calc, feature_to_descriptor


def calculate_features_fast(smiles_list: List[str], calculator: Calculator) -> pd.DataFrame:
    """
    Calculate mordred features for a list of SMILES strings using a reduced calculator.
    
    Args:
        smiles_list: List of SMILES strings
        calculator: Pre-configured Calculator with selected descriptors
        
    Returns:
        DataFrame with calculated features
    """
    results = []
    
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                results.append({})
                continue
            
            # Calculate descriptors
            desc_values = calculator(mol)
            
            # Convert to dictionary
            desc_dict = {}
            for desc, value in desc_values.items():
                # Handle error values
                if isinstance(value, Exception):
                    desc_dict[f"mordred_{desc}"] = np.nan
                else:
                    desc_dict[f"mordred_{desc}"] = value
            
            results.append(desc_dict)
        except Exception as e:
            print(f"Error processing {smiles}: {e}")
            results.append({})
    
    return pd.DataFrame(results)


def save_reduced_feature_list(feature_names: List[str], filepath: str):
    """Save the list of reduced features to a file for future use."""
    with open(filepath, 'w') as f:
        json.dump(feature_names, f, indent=2)
    print(f"Saved {len(feature_names)} features to {filepath}")


def load_reduced_feature_list(filepath: str) -> List[str]:
    """Load a previously saved list of reduced features."""
    with open(filepath, 'r') as f:
        return json.load(f)


def list_available_descriptors(sample_size: int = 100) -> List[str]:
    """
    List all available descriptor names by calculating them on a test molecule.
    This helps identify the exact naming convention used.
    
    Args:
        sample_size: Number of descriptors to list (default: first 100)
        
    Returns:
        List of descriptor names as they appear in calculation results
    """
    test_mol = Chem.MolFromSmiles("CCO")
    calc = Calculator(descriptors, ignore_3D=False)
    
    if test_mol:
        results = calc(test_mol)
        descriptor_names = list(results.keys())[:sample_size]
        return descriptor_names
    return []


def main():
    """
    Main workflow:
    1. Load existing features
    2. Perform correlation analysis
    3. Map features to descriptors
    4. Create reduced calculator
    5. Save reduced feature list
    """
    print("=" * 60)
    print("Mordred Feature Reduction Script")
    print("=" * 60)
    
    # Step 1: Load your existing features (adjust path as needed)
    print("\n1. Loading existing features...")
    try:
        df = pd.read_csv("train_val_features.csv")
        print(f"   Loaded {len(df)} rows with {len(df.columns)} features")
    except FileNotFoundError:
        print("   Warning: train_val_features.csv not found.")
        print("   Please provide a CSV file with mordred features.")
        return
    
    # Remove non-mordred columns if present
    mordred_cols = [col for col in df.columns if col.startswith('mordred_')]
    if not mordred_cols:
        print("   Error: No mordred features found (columns should start with 'mordred_')")
        return
    
    df_mordred = df[mordred_cols].copy()
    
    # Step 2: Perform correlation analysis
    print("\n2. Performing correlation analysis...")
    threshold = 0.7
    reduction_result = calculate_correlation_reduction(df_mordred, threshold=threshold)
    
    print(f"   Original features: {reduction_result['original_count']}")
    print(f"   Reduced features: {reduction_result['reduced_count']}")
    print(f"   Reduction ratio: {reduction_result['reduction_ratio']:.2%}")
    print(f"   Features dropped: {len(reduction_result['features_to_drop'])}")
    
    # Step 3: Map features to descriptors
    print("\n3. Mapping feature names to Mordred descriptors...")
    feature_to_descriptor, missing = map_feature_names_to_descriptors(
        reduction_result['features_to_keep']
    )
    
    print(f"   Successfully mapped: {len(feature_to_descriptor)} descriptors")
    if missing:
        print(f"   Could not map: {len(missing)} descriptors")
    
    # Step 4: Create reduced calculator
    print("\n4. Creating reduced Calculator...")
    calculator, mapping = create_reduced_calculator(reduction_result['features_to_keep'])
    print(f"   Calculator configured with {len(calculator.descriptors)} descriptors")
    
    # Step 5: Save reduced feature list
    print("\n5. Saving reduced feature list...")
    save_reduced_feature_list(
        reduction_result['features_to_keep'],
        "reduced_mordred_features.json"
    )
    
    # Step 6: Example usage
    print("\n6. Example usage:")
    print("   " + "=" * 56)
    print("   # Load reduced feature list")
    print("   reduced_features = load_reduced_feature_list('reduced_mordred_features.json')")
    print("   ")
    print("   # Create calculator")
    print("   calc, _ = create_reduced_calculator(reduced_features)")
    print("   ")
    print("   # Calculate features for new molecules")
    print("   smiles_list = ['CCO', 'CC(=O)O']")
    print("   df_features = calculate_features_fast(smiles_list, calc)")
    print("   " + "=" * 56)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
    
    return reduction_result, calculator, mapping


if __name__ == "__main__":
    result, calc, mapping = main()

