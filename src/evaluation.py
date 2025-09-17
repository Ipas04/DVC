# src/evaluation.py - Fixed Syntax
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split

def load_processed_data():
    """Load processed data"""
    data = pd.read_csv('data/processed/processed_data.csv')
    return data

def load_models():
    """Load trained models"""
    models = {}
    model_files = {
        'random_forest': 'models/random_forest_model.pkl',
        'logistic_regression': 'models/logistic_regression_model.pkl',
        'svm': 'models/svm_model.pkl'
    }
    
    for name, path in model_files.items():
        if os.path.exists(path):
            try:
                models[name] = joblib.load(path)
                print(f"[OK] Loaded {name}")
            except Exception as e:
                print(f"[ERROR] Failed to load {name}: {e}")
        else:
            print(f"[WARNING] Model not found: {path}")
    
    return models

def evaluate_models(models, X_test, y_test):
    """Evaluate all models and create visualizations"""
    results = {}
    
    if not models:
        print("[ERROR] No models to evaluate")
        return results
    
    # Create figure for confusion matrices
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, model) in enumerate(models.items()):
        print(f"Evaluating {model_name}...")
        
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Classification report
            class_report = classification_report(y_test, y_pred, output_dict=True)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
            
            # Plot confusion matrix
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
            disp.plot(ax=axes[idx], cmap='Blues')
            axes[idx].set_title(f'{model_name}\nAccuracy: {accuracy:.4f}')
            
            results[model_name] = {
                "accuracy": accuracy,
                "classification_report": class_report,
                "confusion_matrix": cm.tolist()
            }
            
            print(f"[OK] {model_name} - Accuracy: {accuracy:.4f}")
            
        except Exception as e:
            print(f"[ERROR] Failed to evaluate {model_name}: {e}")
            continue
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

def create_comparison_plots(results):
    """Create comparison visualizations"""
    if not results:
        print("[WARNING] No results to plot")
        return
    
    # Accuracy comparison
    plt.figure(figsize=(10, 6))
    
    model_names = list(results.keys())
    accuracies = [results[name]["accuracy"] for name in model_names]
    
    bars = plt.bar(model_names, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.title('Model Accuracy Comparison', fontsize=16, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main evaluation function"""
    try:
        print("=== Model Evaluation Started ===")
        
        # Load data
        data = load_processed_data()
        X = data.iloc[:, :-1]
        y = data['label']
        
        # Split data (same split as training)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"Test set size: {len(X_test)}")
        
        # Load models
        models = load_models()
        
        if not models:
            print("[ERROR] No models found! Please run model training first.")
            return False
        
        # Evaluate models
        results = evaluate_models(models, X_test, y_test)
        
        if not results:
            print("[ERROR] No models could be evaluated")
            return False
        
        # Create comparison plots
        create_comparison_plots(results)
        
        # Save evaluation results
        with open('evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # MLflow logging if available
        try:
            import mlflow
            mlflow.set_experiment("activity_recognition")
            
            with mlflow.start_run(run_name="model_evaluation"):
                for model_name, result in results.items():
                    mlflow.log_metric(f"{model_name}_accuracy", result["accuracy"])
                
                # Log artifacts
                mlflow.log_artifact('confusion_matrices.png')
                mlflow.log_artifact('model_comparison.png')
                mlflow.log_artifact('evaluation_results.json')
                
                # Find and log best model
                best_model = max(results.items(), key=lambda x: x[1]["accuracy"])
                mlflow.log_param("best_model", best_model[0])
                mlflow.log_metric("best_accuracy", best_model[1]["accuracy"])
            
            print("[OK] MLflow logging completed")
        except ImportError:
            print("[WARNING] MLflow not available, skipping experiment logging")
        except Exception as e:
            print(f"[WARNING] MLflow error: {e}")
        
        # Print summary
        print(f"\n[SUCCESS] Evaluation completed successfully!")
        print("\n=== RESULTS SUMMARY ===")
        for model_name, result in results.items():
            print(f"{model_name}: {result['accuracy']:.4f}")
        
        best_model = max(results.items(), key=lambda x: x[1]["accuracy"])
        print(f"\nBest Model: {best_model[0]} ({best_model[1]['accuracy']:.4f})")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error in evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n[SUCCESS] All done! Check the generated plots and results.")
    else:
        print("\n[FAILED] Please fix the errors above and try again.")