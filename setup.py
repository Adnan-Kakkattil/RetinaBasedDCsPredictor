"""
Setup script for the Retina-Based Heart Disease Predictor project
"""
import os
import sys
import subprocess

def check_python_version():
    """Check if Python version is 3.8 or higher"""
    if sys.version_info < (3, 8):
        print("ERROR: Python 3.8 or higher is required!")
        print(f"Current version: {sys.version}")
        return False
    print(f"[OK] Python version: {sys.version.split()[0]}")
    return True

def install_requirements():
    """Install required packages"""
    print("\nInstalling required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("[OK] Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("[ERROR] Failed to install requirements")
        return False

def create_directories():
    """Create necessary directories"""
    print("\nCreating project directories...")
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'logs',
        'uploads',
        'templates',
        'static'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"[OK] Created {directory}/")
    
    return True

def main():
    """Main setup function"""
    print("=" * 60)
    print("Retina-Based Heart Disease Predictor - Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Create directories
    if not create_directories():
        return
    
    # Install requirements
    if not install_requirements():
        print("\n[WARNING] Some packages may not have installed correctly.")
        print("You can try manually: pip install -r requirements.txt")
    
    print("\n" + "=" * 60)
    print("Setup completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Download dataset: python utils/download_dataset.py")
    print("2. Organize your data in data/raw/ directory")
    print("3. Train model: python src/train.py")
    print("4. Evaluate model: python src/evaluate.py")
    print("5. Run web app: python app.py")
    print("=" * 60)

if __name__ == "__main__":
    main()

