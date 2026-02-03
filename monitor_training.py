#!/usr/bin/env python3
"""
Monitor training progress and estimate completion time.
"""

import os
import time
import subprocess
import json
from datetime import datetime, timedelta

def check_process_running():
    """Check if training process is still running."""
    try:
        result = subprocess.run(
            ['ps', 'aux'], 
            capture_output=True, 
            text=True
        )
        return 'train_model.py' in result.stdout and 'peptide_molecular_combined.csv' in result.stdout
    except:
        return False

def check_results_directory():
    """Check what files exist in results directory."""
    results_dir = 'peptide_full_results'
    if not os.path.exists(results_dir):
        return None, []
    
    files = os.listdir(results_dir)
    file_info = []
    for f in files:
        filepath = os.path.join(results_dir, f)
        if os.path.isfile(filepath):
            size = os.path.getsize(filepath)
            mtime = os.path.getmtime(filepath)
            file_info.append({
                'name': f,
                'size': size,
                'modified': datetime.fromtimestamp(mtime)
            })
    
    return results_dir, sorted(file_info, key=lambda x: x['modified'], reverse=True)

def estimate_progress(files):
    """Estimate training progress based on files created."""
    file_names = [f['name'] for f in files]
    
    progress_stages = {
        'selected_features.json': 'Feature Selection Complete',
        'feature_selection_scores.csv': 'Feature Selection Complete',
        'best_hyperparameters.json': 'Optuna Optimization Complete',
        'best_model.pkl': 'Model Training Complete',
        'training_results.json': 'Training Complete'
    }
    
    completed_stages = []
    for stage_file, stage_name in progress_stages.items():
        if stage_file in file_names:
            completed_stages.append(stage_name)
    
    return completed_stages

def estimate_time_remaining(files, start_time):
    """Estimate time remaining based on progress."""
    elapsed = time.time() - start_time
    
    file_names = [f['name'] for f in files]
    
    # Estimate based on stages
    if 'selected_features.json' in file_names:
        if 'best_hyperparameters.json' in file_names:
            # Optuna done, final training
            if 'best_model.pkl' in file_names:
                return 0, "Training complete!"
            else:
                return max(0, 30 * 60 - elapsed), "Final model training"
        else:
            # Feature selection done, Optuna in progress
            # Check if we can estimate from elapsed time
            # Assume feature selection took ~20% of total time
            estimated_total = elapsed / 0.2
            remaining = estimated_total - elapsed
            return max(0, remaining), "Optuna optimization in progress"
    else:
        # Still in feature selection
        # Assume feature selection takes 15-30 minutes
        estimated_feature_selection = 20 * 60  # 20 minutes
        if elapsed < estimated_feature_selection:
            return estimated_feature_selection - elapsed, "Feature selection in progress"
        else:
            # Feature selection taking longer than expected
            return None, "Feature selection taking longer than expected"

def main():
    print("=" * 80)
    print("TRAINING PROGRESS MONITOR")
    print("=" * 80)
    print(f"Started monitoring at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check if process is running
    is_running = check_process_running()
    print(f"Training process running: {'Yes' if is_running else 'No'}")
    
    if not is_running:
        print("\n⚠️  Training process not found. It may have completed or failed.")
        print("Checking for results...")
    
    # Check results directory
    results_dir, files = check_results_directory()
    
    if results_dir:
        print(f"\nResults directory: {results_dir}")
        print(f"Files found: {len(files)}")
        
        if files:
            print("\nRecent files:")
            for f in files[:5]:
                size_kb = f['size'] / 1024
                size_mb = f['size'] / (1024 * 1024)
                if size_mb >= 1:
                    size_str = f"{size_mb:.2f} MB"
                else:
                    size_str = f"{size_kb:.2f} KB"
                print(f"  {f['name']}: {size_str} (modified: {f['modified'].strftime('%H:%M:%S')})")
        
        # Estimate progress
        completed = estimate_progress(files)
        if completed:
            print("\n✓ Completed stages:")
            for stage in completed:
                print(f"  - {stage}")
        
        # Estimate time remaining
        # Use file modification times to estimate
        if files:
            latest_file_time = max(f['modified'] for f in files)
            time_since_last_file = (datetime.now() - latest_file_time).total_seconds()
            
            if 'training_results.json' in [f['name'] for f in files]:
                print("\n" + "=" * 80)
                print("🎉 TRAINING COMPLETE!")
                print("=" * 80)
            elif 'best_model.pkl' in [f['name'] for f in files]:
                print(f"\n⏱️  Status: Finalizing results...")
                print(f"   Last file updated {time_since_last_file/60:.1f} minutes ago")
            elif 'best_hyperparameters.json' in [f['name'] for f in files]:
                print(f"\n⏱️  Status: Training final model...")
                print(f"   Last file updated {time_since_last_file/60:.1f} minutes ago")
            elif 'selected_features.json' in [f['name'] for f in files]:
                print(f"\n⏱️  Status: Optuna optimization in progress...")
                print(f"   Feature selection completed {time_since_last_file/60:.1f} minutes ago")
                print(f"   Estimated time remaining: 1-3 hours (depending on hardware)")
            else:
                print(f"\n⏱️  Status: Feature selection in progress...")
                print(f"   Last file updated {time_since_last_file/60:.1f} minutes ago")
                print(f"   Estimated time remaining: 10-30 minutes for feature selection, then 1-3 hours for Optuna")
    else:
        print("\n⏳ Results directory not created yet - training still initializing...")
        print("   This stage typically takes 5-15 minutes")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()

