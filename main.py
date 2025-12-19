import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
SCRIPTS_DIR = ROOT / "scripts"

def run_script(script_name: str):
    """Run a script as a module so imports work correctly."""
    module_name = f"scripts.{script_name.replace('.py', '')}"

    print(f"Running: {module_name}")

    subprocess.run(
        [sys.executable, "-m", module_name],
        check=True
    )


print("=" * 60)
print("PIPELINE START")
print("=" * 60)

# 1 - run data preparation to extract features from raw data
print("\n" + "=" * 60)
print("STEP 1: Data Preparation")
print("Extracting features from raw data...")
print("=" * 60)
#run_script("data_preparation.py")
print("✓ Data preparation complete!")

# 2 - run LOSO model training to train models on extracted features
print("\n" + "=" * 60)
print("STEP 2: LOSO (Leave-One-Subject-Out) Model Training")
print("=" * 60)

# 2a - task-agnostic training: train multiple models on different tasks
print("\n--- STEP 2a: Task-Agnostic Training ---")
print("Training models across different tasks...")
run_script("LOSO_task_agnostic_train.py")
print("✓ Task-agnostic training complete!")

# 2b - task-specific training: train a single model on a specific task
print("\n--- STEP 2b: Task-Specific Training ---")
print("Training models for specific tasks...")
run_script("LOSO_task_specific_train.py")
print("✓ Task-specific training complete!")

# After training, generate average plots across tasks
print("\n--- Generating Average Plots ---")
print("Creating summary visualizations...")
run_script("plot_average.py")
print("✓ Average plots generated!")

# 2c - run LOSO top performing model for feature importance and SHAP analysis
print("\n--- STEP 2c: Feature Importance Analysis ---")
print("Analyzing feature importance for LOSO models...")
run_script("LOSO_feature_importance.py")
print("✓ Feature importance analysis complete!")

# 3a - run Time split model training
print("\n" + "=" * 60)
print("STEP 3: Global (Time-Split) Model Training")
print("=" * 60)
print("Training global models with time-based splits...")
run_script("GLOBAL_train.py")
print("✓ Global training complete!")

# 3b - Feature importance and SHAP for time split models
print("\n--- STEP 3b: Global Feature Importance ---")
print("Analyzing feature importance for global models...")
run_script("GLOBAL_feature_importance.py")
print("✓ Global feature importance analysis complete!")

print("\n" + "=" * 60)
print("PIPELINE COMPLETE!")
print("All steps executed successfully.")
print("=" * 60)