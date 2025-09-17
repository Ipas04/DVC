# setup_project.py
import os
import sys

def create_project_structure():
    """Create necessary project directories"""
    directories = [
        'data/raw',
        'data/processed', 
        'models',
        'mlruns',
        'src',
        '.github/workflows'
    ]
    
    print("Creating project structure...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created: {directory}")
        
        # Create .gitkeep files to preserve empty directories
        gitkeep_path = os.path.join(directory, '.gitkeep')
        if not os.path.exists(gitkeep_path):
            with open(gitkeep_path, 'w') as f:
                f.write('')

def create_params_file():
    """Create params.yaml if it doesn't exist"""
    params_content = """
data:
  test_size: 0.3
  random_state: 42
  selected_labels: [1, 6, 7]
  window_size: 20

preprocessing:
  undersampling: true
  undersampling_random_state: 42

models:
  random_forest:
    n_estimators: 100
    random_state: 42
  
  logistic_regression:
    multi_class: "multinomial"
    solver: "lbfgs"
    max_iter: 1000
  
  svm:
    kernel: "linear"
    probability: true

mlflow:
  experiment_name: "activity_recognition"
  tracking_uri: "mlruns"
"""
    
    if not os.path.exists('params.yaml'):
        with open('params.yaml', 'w') as f:
            f.write(params_content.strip())
        print("✓ Created: params.yaml")
    else:
        print("✓ params.yaml already exists")

def test_imports():
    """Test if required packages are installed"""
    required_packages = [
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('sklearn', 'sklearn'),
        ('yaml', 'yaml'),
        ('scipy', 'scipy'),
        ('matplotlib', 'plt'),
        ('seaborn', 'sns')
    ]
    
    optional_packages = [
        ('mlflow', 'mlflow'),
        ('imblearn', 'imblearn'),
        ('kaggle', 'kaggle')
    ]
    
    print("\nTesting required packages...")
    missing_required = []
    
    for package, alias in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
            elif package == 'matplotlib':
                import matplotlib.pyplot as plt
            else:
                exec(f"import {package}")
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing_required.append(package)
    
    print("\nTesting optional packages...")
    missing_optional = []
    
    for package, alias in optional_packages:
        try:
            exec(f"import {package}")
            print(f"✓ {package}")
        except ImportError:
            print(f"⚠ {package} - OPTIONAL, not installed")
            missing_optional.append(package)
    
    if missing_required:
        print(f"\n❌ Missing required packages: {missing_required}")
        print("Please install them using:")
        for pkg in missing_required:
            if pkg == 'sklearn':
                print("pip install --user scikit-learn")
            else:
                print(f"pip install --user {pkg}")
        return False
    
    if missing_optional:
        print(f"\n⚠ Missing optional packages: {missing_optional}")
        print("Install them for full functionality:")
        for pkg in missing_optional:
            print(f"pip install --user {pkg}")
    
    return True

def main():
    print("=== Project Setup ===")
    create_project_structure()
    create_params_file()
    
    if test_imports():
        print("\n✅ Setup completed successfully!")
        print("You can now run: python src/data_preprocessing.py")
    else:
        print("\n❌ Setup incomplete. Please install missing packages first.")

if __name__ == "__main__":
    main()