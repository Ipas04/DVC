import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import json

def load_params():
    with open('params.yaml', 'r') as file:
        params = yaml.safe_load(file)
    return params

def load_processed_data():
    return pd.read_csv('data/processed/processed_data.csv')

def create_models(params):
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

def main():
    params = load_params()
    
    # Set MLflow experiment
    mlflow.set_experiment(params['mlflow']['experiment_name'])
    
    # Load data
    data = load_processed_data()
    X = data.iloc[:, :-1]
    y = data['label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=params['data']['test_size'],
        random_state=params['data']['random_state'],
        stratify=y
    )
    
    # Create models
    models = create_models(params)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    metrics = {}
    
    for model_name, model in models.items():
        with mlflow.start_run(run_name=f"training_{model_name}"):
            # Log model parameters
            mlflow.log_params(params['models'][model_name])
            mlflow.log_param("model_type", model_name)
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("train_size", len(X_train))
            mlflow.log_metric("test_size", len(X_test))
            
            # Log classification report
            class_report = classification_report(y_test, y_pred, output_dict=True)
            for label, metrics_dict in class_report.items():
                if isinstance(metrics_dict, dict):
                    for metric, value in metrics_dict.items():
                        mlflow.log_metric(f"{label}_{metric}", value)
            
            # Save model
            model_path = f'models/{model_name}_model.pkl'
            joblib.dump(model, model_path)
            
            # Log model
            mlflow.sklearn.log_model(model, f"{model_name}_model")
            mlflow.log_artifact(model_path)
            
            metrics[model_name] = {
                "accuracy": accuracy,
                "classification_report": class_report
            }
            
            print(f"{model_name} - Accuracy: {accuracy:.4f}")
    
    # Save metrics
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("Model training completed!")

if __name__ == "__main__":
    main()