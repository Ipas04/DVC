# run_pipeline.py - Windows Compatible
import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*50}")
    print(f"RUNNING: {description}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print("[SUCCESS]")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAILED]: {e}")
        if e.stdout:
            print("Output:", e.stdout)
        if e.stderr:
            print("Error:", e.stderr)
        return False

def main():
    """Run complete ML pipeline"""
    print("STARTING COMPLETE ML PIPELINE")
    
    steps = [
        ("python src/data_preprocessing_robust.py", "Data Preprocessing"),
        ("python src/model_training.py", "Model Training"),
        ("python src/evaluation.py", "Model Evaluation")
    ]
    
    for command, description in steps:
        if not run_command(command, description):
            print(f"\n[PIPELINE FAILED] at: {description}")
            return False
    
    print(f"\n[PIPELINE COMPLETED SUCCESSFULLY]")
    print("\nResults:")
    print("- Processed data: data/processed/processed_data.csv")
    print("- Trained models: models/")
    print("- Evaluation plots: confusion_matrices.png, model_comparison.png")
    print("- Results: training_results.json, evaluation_results.json")
    
    # Try to show MLflow UI command
    try:
        import mlflow
        print("\nTo view MLflow experiments:")
        print("mlflow ui")
    except ImportError:
        pass
    
    return True

if __name__ == "__main__":
    main()