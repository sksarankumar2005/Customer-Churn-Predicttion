"""
run_pipeline.py
---------------
Single entry point to execute the entire Customer Churn Prediction pipeline:
  1. Data Preprocessing
  2. Model Training
  3. Model Evaluation
  4. Generate EDA Notebook

Usage:
    python run_pipeline.py
"""

import subprocess
import sys
import os


def run_step(step_name, script_path):
    """Run a pipeline step as a subprocess and check for errors."""
    print("=" * 60)
    print(f"  STEP: {step_name}")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        capture_output=False
    )

    if result.returncode != 0:
        print(f"\n[FAILED] {step_name}")
        print(f"  Script '{script_path}' exited with code {result.returncode}")
        sys.exit(1)

    print()


def main():
    print("\n" + "=" * 60)
    print("  CUSTOMER CHURN PREDICTION -- FULL PIPELINE")
    print("=" * 60 + "\n")

    # Step 1: Data Preprocessing
    run_step("Data Preprocessing", os.path.join("src", "data_preprocessing.py"))

    # Step 2: Model Training
    run_step("Model Training", os.path.join("src", "model_training.py"))

    # Step 3: Model Evaluation
    run_step("Model Evaluation", os.path.join("src", "evaluation.py"))

    # Step 4: Generate EDA Notebook
    run_step("Generate EDA Notebook", os.path.join("src", "create_eda_notebook.py"))

    print("=" * 60)
    print("  PIPELINE COMPLETE -- All steps executed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("  - View EDA notebook:  jupyter notebook notebooks/01_EDA_and_Insights.ipynb")
    print("  - Check saved models: models/")
    print("  - View plots:         notebooks/images/")
    print()


if __name__ == "__main__":
    main()
