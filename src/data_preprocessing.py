import os
import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import mode

def load_params():
    with open('params.yaml', 'r') as file:
        params = yaml.safe_load(file)
    return params

def load_and_combine_data():
    """Load and combine all CSV files"""
    data_files = [f'50{i}.csv' for i in range(1, 11)]
    dataframes = []
    
    for file in data_files:
        if os.path.exists(f'data/raw/{file}'):
            df = pd.read_csv(f'data/raw/{file}')
            dataframes.append(df)
    
    if not dataframes:
        raise FileNotFoundError("No data files found in data/raw/")
    
    combined_data = pd.concat(dataframes, ignore_index=True)
    return combined_data

def preprocess_data(data, params):
    """Preprocess the data according to parameters"""
    # Remove timestamp column
    data = data.drop(['timestamp'], axis=1)
    
    # Average every 20 rows
    window_size = params['data']['window_size']
    average_results = []
    
    for i in range(0, len(data), window_size):
        subset = data.iloc[i:i+window_size]
        averages = subset.iloc[:, :6].mean()
        label_mode = mode(subset.iloc[:, 6])[0]
        average_results.append(list(averages) + [label_mode])
    
    processed_data = pd.DataFrame(
        average_results, 
        columns=['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z', 'label']
    )
    
    # Filter selected labels
    selected_labels = params['data']['selected_labels']
    processed_data = processed_data[processed_data['label'].isin(selected_labels)]
    
    # Undersampling if enabled
    if params['preprocessing']['undersampling']:
        features = processed_data.iloc[:, :-1]
        labels = processed_data['label']
        
        undersampler = RandomUnderSampler(
            random_state=params['preprocessing']['undersampling_random_state']
        )
        X_resampled, y_resampled = undersampler.fit_resample(features, labels)
        
        processed_data = pd.DataFrame(X_resampled, columns=features.columns)
        processed_data['label'] = y_resampled
    
    return processed_data

def main():
    params = load_params()
    
    # Set MLflow experiment
    mlflow.set_experiment(params['mlflow']['experiment_name'])
    
    with mlflow.start_run(run_name="data_preprocessing"):
        # Log parameters
        mlflow.log_params(params['data'])
        mlflow.log_params(params['preprocessing'])
        
        # Load and preprocess data
        raw_data = load_and_combine_data()
        mlflow.log_metric("raw_data_size", len(raw_data))
        
        processed_data = preprocess_data(raw_data, params)
        mlflow.log_metric("processed_data_size", len(processed_data))
        
        # Log class distribution
        class_distribution = processed_data['label'].value_counts().to_dict()
        for label, count in class_distribution.items():
            mlflow.log_metric(f"class_{label}_count", count)
        
        # Save processed data
        os.makedirs('data/processed', exist_ok=True)
        processed_data.to_csv('data/processed/processed_data.csv', index=False)
        
        # Log as artifact
        mlflow.log_artifact('data/processed/processed_data.csv')
        
        print(f"Data preprocessing completed. Final dataset size: {len(processed_data)}")

if __name__ == "__main__":
    main()