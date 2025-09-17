import pandas as pd
import yaml
import mlflow
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import os

def load_params():
    with open('params.yaml', 'r') as file:
        params = yaml.safe_load(file)
    return params

def load_models():
    models = {}
    model_files = {
        'random_forest': 'models/random_forest_model.pkl',
        'logistic_regression': 'models/logistic_regression_model.pkl',
        'svm': 'models/svm_model.pkl'
    }
    
    for name, path in model_files.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
    
    return models

def main():
    params = load_params()
    
    # Set MLflow experiment
    mlflow.set_experiment(params['mlflow']['experiment_name'])
    
    # Load data
    data = pd.read_csv('data/processed/processed_data.csv')
    X = data.iloc[:, :-1]
    y = data['label']
    
    # Split data (same as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=params['data']['test_size'],
        random_state=params['data']['random_state'],
        stratify=y
    )
    
    # Load models
    models = load_models()
    
    evaluation_metrics = {}
    
    with mlflow.start_run(run_name="model_evaluation"):
        for model_name, model in models.items():
            # Make predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Log metrics
            mlflow.log_metric(f"{model_name}_accuracy", accuracy)
            
            # Create confusion matrix
            cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
            disp.plot(cmap='Blues')
            plt.title(f'Confusion Matrix - {model_name}')
            
            # Save plot
            plot_path = f'confusion_matrix_{model_name}.png'
            plt.savefig(plot_path)
            mlflow.log_artifact(plot_path)
            plt.close()
            
            evaluation_metrics[model_name] = {
                "accuracy": accuracy,
                "confusion_matrix": cm.tolist()
            }
        
        # Create comparison plot
        plt.figure(figsize=(10, 6))
        accuracies = [evaluation_metrics[name]["accuracy"] for name in models.keys()]
        sns.barplot(x=list(models.keys()), y=accuracies, palette='viridis')
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        
        comparison_plot_path = 'model_comparison.png'
        plt.savefig(comparison_plot_path)
        mlflow.log_artifact(comparison_plot_path)
        plt.close()
        
        # Find best model
        best_model = max(evaluation_metrics.items(), key=lambda x: x[1]["accuracy"])
        mlflow.log_param("best_model", best_model[0])
        mlflow.log_metric("best_accuracy", best_model[1]["accuracy"])
    
    # Save evaluation metrics
    with open('evaluation_metrics.json', 'w') as f:
        json.dump(evaluation_metrics, f, indent=2)
    
    print("Model evaluation completed!")
    print(f"Best model: {best_model[0]} with accuracy: {best_model[1]['accuracy']:.4f}")

if __name__ == "__main__":
    main()