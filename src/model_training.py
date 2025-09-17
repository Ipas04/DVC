# src/model_training.py - Fixed Syntax
import pandas as pd
import numpy as np
import os
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_params():
    """Load parameters with fallback"""
    default_params = {
        'data': {
            'test_size': 0.3,
            'random_state': 42
        },
        'models': {
            'random_forest': {
                'n_estimators': 100,
                'random_state': 42
            },
            'logistic_regression': {
                'multi_class': 'multinomial',
                'solver': 'lbfgs',
                'max_iter': 1000
            },
            'svm': {
                'kernel': 'linear',
                'probability': True
            }
        },
        'mlflow': {
            'experiment_name': 'activity_recognition'
        }
    }
    
    try:
        import yaml
        with open('params.yaml', 'r') as file:
            params = yaml.safe_load(file)
        print("[OK] Loaded parameters from params.yaml")
        return params
    except Exception as e:
        print(f"[WARNING] Using default parameters: {e}")
        return default_params

def load_processed_data():
    """Load preprocessed data"""
    data_path = 'data/processed/processed_data.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data not found at {data_path}")
    
    data = pd.read_csv(data_path)
    print(f"[OK] Loaded processed data: {data.shape}")
    return data

def create_models(params):
    """Create model instances"""
    models = {
        'random_forest': RandomForestClassifier(
            n_estimators=params['models']['random_forest']['n_estimators'],
            random_state=params['models']['random_forest']['random_state']
        ),
        'logistic_regression': LogisticRegression(
            multi_class=params['models']['logistic_regression']['multi_class'],
            solver=params['models']['logistic_regression']['solver'],
            max_iter=params['models']['logistic_regression']['max_iter']
        ),
        'svm': SVC(
            kernel=params['models']['svm']['kernel'],
            probability=params['models']['svm']['probability']
        )
    }
    return models

def train_and_evaluate_models(X_train, X_test, y_train, y_test, models, params):
    """Train and evaluate all models"""
    results = {}
    
    # Try to use MLflow if available
    use_mlflow = False
    try:
        import mlflow
        import mlflow.sklearn
        mlflow.set_experiment(params['mlflow']['experiment_name'])
        use_mlflow = True
        print("[OK] MLflow available")
    except ImportError:
        print("[WARNING] MLflow not available, continuing without experiment tracking")
    except Exception as e:
        print(f"[WARNING] MLflow error: {e}")
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        
        try:
            if use_mlflow:
                with mlflow.start_run(run_name=f"training_{model_name}"):
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    # Get classification report
                    class_report = classification_report(y_test, y_pred, output_dict=True)
                    
                    # Log parameters
                    mlflow.log_params(params['models'][model_name])
                    mlflow.log_param("model_type", model_name)
                    
                    # Log metrics
                    mlflow.log_metric("accuracy", accuracy)
                    mlflow.log_metric("train_size", len(X_train))
                    mlflow.log_metric("test_size", len(X_test))
                    
                    # Log detailed metrics
                    for label, metrics in class_report.items():
                        if isinstance(metrics, dict):
                            for metric_name, value in metrics.items():
                                mlflow.log_metric(f"{label}_{metric_name}", value)
                    
                    # Save and log model
                    model_path = f'models/{model_name}_model.pkl'
                    joblib.dump(model, model_path)
                    mlflow.sklearn.log_model(model, f"{model_name}_model")
                    mlflow.log_artifact(model_path)
            else:
                # Train without MLflow
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                class_report = classification_report(y_test, y_pred, output_dict=True)
                
                # Save model
                model_path = f'models/{model_name}_model.pkl'
                joblib.dump(model, model_path)
            
            # Store results
            results[model_name] = {
                "accuracy": accuracy,
                "classification_report": class_report,
                "model_path": model_path
            }
            
            print(f"[OK] {model_name} - Accuracy: {accuracy:.4f}")
            
        except Exception as e:
            print(f"[ERROR] Failed to train {model_name}: {e}")
            continue
    
    return results

def main():
    """Main training function"""
    try:
        print("=== Model Training Started ===")
        
        # Ensure models directory exists
        os.makedirs('models', exist_ok=True)
        
        # Load parameters
        params = load_params()
        
        # Load processed data
        data = load_processed_data()
        
        # Prepare features and labels
        X = data.iloc[:, :-1]  # All columns except last (label)
        y = data['label']      # Label column
        
        print(f"Features shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Unique labels: {sorted(y.unique())}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=params['data']['test_size'],
            random_state=params['data']['random_state'],
            stratify=y
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Create models
        models = create_models(params)
        print(f"[OK] Created {len(models)} models: {list(models.keys())}")
        
        # Train and evaluate models
        results = train_and_evaluate_models(X_train, X_test, y_train, y_test, models, params)
        
        if not results:
            print("[ERROR] No models were trained successfully")
            return False
        
        # Save training results
        training_results = {}
        for model_name, result in results.items():
            training_results[model_name] = {
                "accuracy": result["accuracy"],
                "model_path": result["model_path"]
            }
        
        with open('training_results.json', 'w') as f:
            json.dump(training_results, f, indent=2)
        
        # Find best model
        best_model = max(results.items(), key=lambda x: x[1]["accuracy"])
        
        print(f"\n[SUCCESS] Training completed successfully!")
        print(f"Best model: {best_model[0]} with accuracy: {best_model[1]['accuracy']:.4f}")
        print(f"Results saved to: training_results.json")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error in model training: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n[SUCCESS] Ready for model evaluation!")
    else:
        print("\n[FAILED] Please fix the errors above and try again.")