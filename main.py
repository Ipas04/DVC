import subprocess
import sys
import os

def run_pipeline():
    """Run the complete ML pipeline"""
    try:
        print("Starting ML Pipeline...")
        
        # Run DVC pipeline
        print("Running DVC pipeline...")
        subprocess.run(["dvc", "repro"], check=True)
        
        print("Pipeline completed successfully!")
        
        # Show MLflow UI command
        print("\nTo view MLflow results, run:")
        print("mlflow ui")
        
    except subprocess.CalledProcessError as e:
        print(f"Pipeline failed with error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_pipeline()