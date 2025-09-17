# src/data_preprocessing_robust.py - Windows Compatible
import os
import sys
import pandas as pd
import numpy as np

def ensure_directories():
    """Ensure all necessary directories exist"""
    dirs = ['data/raw', 'data/processed', 'models', 'mlruns']
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"[OK] Directory ensured: {directory}")

def load_params():
    """Load parameters with fallback to defaults"""
    default_params = {
        'data': {
            'window_size': 20,
            'selected_labels': [1, 6, 7],
            'test_size': 0.3,
            'random_state': 42
        },
        'preprocessing': {
            'undersampling': True,
            'undersampling_random_state': 42
        },
        'mlflow': {
            'experiment_name': 'activity_recognition'
        }
    }
    
    try:
        import yaml
        with open('params.yaml', 'r') as file:
            params = yaml.safe_load(file)
        print("[OK] Loaded params from params.yaml")
        return params
    except (ImportError, FileNotFoundError) as e:
        print(f"[WARNING] Using default parameters: {e}")
        return default_params

def download_from_kaggle():
    """Try to download data from Kaggle"""
    try:
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        # Setup API
        api = KaggleApi()
        api.authenticate()
        
        dataset_name = "anshtanwar/adult-subjects-70-95-years-activity-recognition"
        download_path = "data/raw"
        
        print(f"Downloading {dataset_name}...")
        api.dataset_download_files(dataset_name, path=download_path, unzip=True)
        
        # Check if files exist
        expected_files = [f'50{i}.csv' for i in range(1, 11)]
        found_files = []
        
        for file in expected_files:
            if os.path.exists(f'data/raw/{file}'):
                found_files.append(file)
        
        if found_files:
            print(f"[OK] Downloaded {len(found_files)} files from Kaggle")
            return True
        else:
            print("[WARNING] No expected files found after download")
            return False
            
    except Exception as e:
        print(f"[WARNING] Kaggle download failed: {e}")
        return False

def generate_realistic_sample_data():
    """Generate realistic sample data"""
    print("Generating sample data...")
    
    np.random.seed(42)
    
    # Generate data for each subject (501-510)
    for subject_id in range(501, 511):
        n_samples = np.random.randint(4000, 6000)  # Vary sample sizes
        
        data = []
        
        for i in range(n_samples):
            # Generate timestamp
            timestamp = i
            
            # Generate realistic activity patterns
            if i < n_samples * 0.3:
                activity = 1  # Walking
                back_x = np.random.normal(0.1, 0.3)
                back_y = np.random.normal(0.8, 0.2)
                back_z = np.random.normal(0.2, 0.1)
                thigh_x = np.random.normal(0.2, 0.4)
                thigh_y = np.random.normal(0.6, 0.3)
                thigh_z = np.random.normal(0.1, 0.2)
            elif i < n_samples * 0.6:
                activity = 6  # Standing
                back_x = np.random.normal(0.0, 0.1)
                back_y = np.random.normal(1.0, 0.1)
                back_z = np.random.normal(0.0, 0.1)
                thigh_x = np.random.normal(0.0, 0.1)
                thigh_y = np.random.normal(1.0, 0.1)
                thigh_z = np.random.normal(0.0, 0.1)
            elif i < n_samples * 0.9:
                activity = 7  # Sitting
                back_x = np.random.normal(0.0, 0.2)
                back_y = np.random.normal(0.7, 0.2)
                back_z = np.random.normal(0.7, 0.2)
                thigh_x = np.random.normal(0.0, 0.1)
                thigh_y = np.random.normal(0.0, 0.1)
                thigh_z = np.random.normal(1.0, 0.1)
            else:
                # Other activities (2, 3, 4, 5, 8)
                activity = np.random.choice([2, 3, 4, 5, 8])
                back_x = np.random.normal(0, 0.5)
                back_y = np.random.normal(0, 0.5)
                back_z = np.random.normal(0, 0.5)
                thigh_x = np.random.normal(0, 0.5)
                thigh_y = np.random.normal(0, 0.5)
                thigh_z = np.random.normal(0, 0.5)
            
            data.append([
                timestamp, back_x, back_y, back_z, 
                thigh_x, thigh_y, thigh_z, activity
            ])
        
        # Create DataFrame and save
        df = pd.DataFrame(data, columns=[
            'timestamp', 'back_x', 'back_y', 'back_z',
            'thigh_x', 'thigh_y', 'thigh_z', 'label'
        ])
        
        filename = f'data/raw/{subject_id}.csv'
        df.to_csv(filename, index=False)
        print(f"[OK] Generated {filename} with {len(df)} samples")
    
    return True

def load_and_combine_data():
    """Load and combine data files"""
    # Try Kaggle download first
    if not download_from_kaggle():
        # Generate sample data if download fails
        generate_realistic_sample_data()
    
    # Load data files
    data_files = [f'50{i}.csv' for i in range(1, 11)]
    dataframes = []
    
    for file in data_files:
        file_path = f'data/raw/{file}'
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                dataframes.append(df)
                print(f"[OK] Loaded {file}: {len(df)} rows")
            except Exception as e:
                print(f"[WARNING] Error loading {file}: {e}")
    
    if not dataframes:
        raise FileNotFoundError("No data files could be loaded!")
    
    combined_data = pd.concat(dataframes, ignore_index=True)
    print(f"[OK] Combined data: {combined_data.shape}")
    
    return combined_data

def preprocess_data(data, params):
    """Preprocess the data"""
    print("Starting preprocessing...")
    
    # Remove timestamp if present
    if 'timestamp' in data.columns:
        data = data.drop(['timestamp'], axis=1)
    
    print(f"Data shape: {data.shape}")
    print(f"Label distribution:")
    print(data['label'].value_counts().sort_index())
    
    # Windowing
    window_size = params['data']['window_size']
    average_results = []
    
    print(f"Applying windowing (size={window_size})...")
    
    for i in range(0, len(data), window_size):
        subset = data.iloc[i:i+window_size]
        if len(subset) < window_size:
            continue
        
        # Calculate averages for sensor data
        averages = subset.iloc[:, :6].mean()
        
        # Get most common label in window
        try:
            from scipy.stats import mode
            label_mode = mode(subset.iloc[:, 6], keepdims=False)[0]
        except ImportError:
            # Fallback without scipy
            label_mode = subset.iloc[:, 6].mode().iloc[0]
        
        average_results.append(list(averages) + [label_mode])
    
    processed_data = pd.DataFrame(
        average_results,
        columns=['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z', 'label']
    )
    
    print(f"After windowing: {processed_data.shape}")
    
    # Filter selected labels
    selected_labels = params['data']['selected_labels']
    processed_data = processed_data[processed_data['label'].isin(selected_labels)]
    
    print(f"After label filtering {selected_labels}: {processed_data.shape}")
    print(f"Final label distribution:")
    print(processed_data['label'].value_counts().sort_index())
    
    # Undersampling if available and enabled
    if params['preprocessing']['undersampling']:
        try:
            from imblearn.under_sampling import RandomUnderSampler
            
            features = processed_data.iloc[:, :-1]
            labels = processed_data['label']
            
            undersampler = RandomUnderSampler(
                random_state=params['preprocessing']['undersampling_random_state']
            )
            X_resampled, y_resampled = undersampler.fit_resample(features, labels)
            
            processed_data = pd.DataFrame(X_resampled, columns=features.columns)
            processed_data['label'] = y_resampled
            
            print(f"After undersampling: {processed_data.shape}")
            print(f"Balanced label distribution:")
            print(processed_data['label'].value_counts().sort_index())
            
        except ImportError:
            print("[WARNING] imbalanced-learn not available, skipping undersampling")
    
    return processed_data

def main():
    """Main preprocessing function"""
    try:
        print("=== Data Preprocessing Started ===")
        
        # Ensure directories exist
        ensure_directories()
        
        # Load parameters
        params = load_params()
        
        # Load and combine data
        raw_data = load_and_combine_data()
        
        # Preprocess data
        processed_data = preprocess_data(raw_data, params)
        
        # Save processed data
        output_path = 'data/processed/processed_data.csv'
        processed_data.to_csv(output_path, index=False)
        print(f"[OK] Saved processed data to: {output_path}")
        
        # Try MLflow logging if available
        try:
            import mlflow
            mlflow.set_experiment(params['mlflow']['experiment_name'])
            
            with mlflow.start_run(run_name="data_preprocessing"):
                mlflow.log_metric("raw_data_size", len(raw_data))
                mlflow.log_metric("processed_data_size", len(processed_data))
                
                # Log class distribution
                class_dist = processed_data['label'].value_counts().to_dict()
                for label, count in class_dist.items():
                    mlflow.log_metric(f"class_{label}_count", count)
                
                mlflow.log_artifact(output_path)
            
            print("[OK] MLflow logging completed")
            
        except ImportError:
            print("[WARNING] MLflow not available, skipping experiment logging")
        
        print(f"\n[SUCCESS] Preprocessing completed successfully!")
        print(f"Final dataset: {len(processed_data)} samples")
        print(f"Features: {list(processed_data.columns[:-1])}")
        print(f"Labels: {sorted(processed_data['label'].unique())}")
        
    except Exception as e:
        print(f"[ERROR] Error in preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n[SUCCESS] You can now run the next step!")
    else:
        print("\n[FAILED] Please fix the errors above and try again.")