"""
Test script to validate all imports and check for common issues
"""
import sys
import os

print("=" * 60)
print("Testing Project Imports and Setup")
print("=" * 60)

errors = []
warnings = []

# Test 1: Check Python version
print("\n[1] Checking Python version...")
if sys.version_info < (3, 8):
    errors.append(f"Python 3.8+ required. Current: {sys.version}")
    print("[ERROR] Python version too old")
else:
    print(f"[OK] Python {sys.version.split()[0]}")

# Test 2: Check project structure
print("\n[2] Checking project structure...")
required_dirs = ['data', 'src', 'utils', 'models', 'templates']
for dir_name in required_dirs:
    if os.path.exists(dir_name):
        print(f"[OK] Directory exists: {dir_name}/")
    else:
        errors.append(f"Missing directory: {dir_name}/")
        print(f"[ERROR] Missing directory: {dir_name}/")

# Test 3: Check config import
print("\n[3] Testing config import...")
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from src.config import *
    print("[OK] Config imported successfully")
except Exception as e:
    errors.append(f"Config import failed: {str(e)}")
    print(f"[ERROR] Config import failed: {str(e)}")

# Test 4: Check Python syntax
print("\n[4] Checking Python syntax...")
python_files = [
    'src/config.py',
    'src/data_preprocessing.py',
    'src/model_builder.py',
    'src/train.py',
    'src/evaluate.py',
    'app.py',
    'setup.py',
    'utils/download_dataset.py'
]

for file_path in python_files:
    if os.path.exists(file_path):
        try:
            compile(open(file_path).read(), file_path, 'exec')
            print(f"[OK] Syntax check passed: {file_path}")
        except SyntaxError as e:
            errors.append(f"Syntax error in {file_path}: {str(e)}")
            print(f"[ERROR] Syntax error in {file_path}: {str(e)}")
    else:
        warnings.append(f"File not found: {file_path}")
        print(f"[WARNING] File not found: {file_path}")

# Test 5: Check dependencies (if available)
print("\n[5] Checking dependencies...")
dependencies = {
    'numpy': 'numpy',
    'pandas': 'pandas',
    'cv2': 'opencv-python',
    'sklearn': 'scikit-learn',
    'tensorflow': 'tensorflow',
    'flask': 'flask'
}

for module_name, package_name in dependencies.items():
    try:
        __import__(module_name)
        print(f"[OK] {package_name} is installed")
    except ImportError:
        warnings.append(f"{package_name} is not installed")
        print(f"[WARNING] {package_name} is not installed (run: pip install -r requirements.txt)")

# Test 6: Check file permissions
print("\n[6] Checking file permissions...")
if os.access('.', os.W_OK):
    print("[OK] Write permissions: OK")
else:
    errors.append("No write permissions in current directory")
    print("[ERROR] No write permissions")

# Summary
print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)

if errors:
    print(f"\n[ERRORS] Found {len(errors)} error(s):")
    for i, error in enumerate(errors, 1):
        print(f"  {i}. {error}")
else:
    print("\n[OK] No errors found!")

if warnings:
    print(f"\n[WARNINGS] Found {len(warnings)} warning(s):")
    for i, warning in enumerate(warnings, 1):
        print(f"  {i}. {warning}")
else:
    print("\n[OK] No warnings!")

print("\n" + "=" * 60)

if errors:
    print("\nPlease fix the errors before proceeding.")
    sys.exit(1)
elif warnings:
    print("\nProject structure is OK. Some dependencies may need installation.")
    sys.exit(0)
else:
    print("\nAll checks passed! Project is ready to use.")
    sys.exit(0)

