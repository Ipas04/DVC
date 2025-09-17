# debug_setup.py
import os
import sys

def check_everything():
    print("=== Debugging Setup ===")
    
    print(f"Current directory: {os.getcwd()}")
    print(f"Python version: {sys.version}")
    
    # Check directories
    dirs_to_check = ['data', 'data/raw', 'data/processed', 'src', 'models']
    for directory in dirs_to_check:
        exists = os.path.exists(directory)
        print(f"Directory {directory}: {'✓' if exists else '✗'}")
        
        if exists:
            try:
                files = os.listdir(directory)
                print(f"  Files: {files[:5]}{'...' if len(files) > 5 else ''}")
            except PermissionError:
                print(f"  Permission denied")
    
    # Check files
    files_to_check = ['params.yaml', 'setup_project.py', 'src/data_preprocessing_robust.py']
    for file in files_to_check:
        exists = os.path.exists(file)
        print(f"File {file}: {'✓' if exists else '✗'}")
    
    # Test imports
    packages = ['pandas', 'numpy', 'os', 'sys']
    for package in packages:
        try:
            __import__(package)
            print(f"Import {package}: ✓")
        except ImportError as e:
            print(f"Import {package}: ✗ ({e})")

if __name__ == "__main__":
    check_everything()