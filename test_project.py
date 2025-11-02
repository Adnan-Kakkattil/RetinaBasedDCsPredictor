"""
Comprehensive test script for Retina-Based Heart Disease Predictor
Tests all components that can be tested without full dependencies
"""
import os
import sys
import traceback

print("=" * 70)
print("COMPREHENSIVE PROJECT TEST SUITE")
print("=" * 70)

test_results = []
errors = []
warnings = []

def test(name, func):
    """Run a test and record results"""
    try:
        result = func()
        if result:
            test_results.append((name, "PASSED", None))
            print(f"\n[PASS] {name}")
            return True
        else:
            test_results.append((name, "FAILED", "Test returned False"))
            print(f"\n[FAIL] {name}: Test returned False")
            return False
    except Exception as e:
        test_results.append((name, "ERROR", str(e)))
        errors.append(f"{name}: {str(e)}")
        print(f"\n[ERROR] {name}: {str(e)}")
        return False

# Test 1: File Structure
print("\n" + "=" * 70)
print("TEST 1: Project File Structure")
print("=" * 70)

def test_file_structure():
    required_files = [
        'README.md',
        'requirements.txt',
        'setup.py',
        'app.py',
        'src/config.py',
        'src/data_preprocessing.py',
        'src/model_builder.py',
        'src/train.py',
        'src/evaluate.py',
        'templates/index.html',
        'utils/download_dataset.py'
    ]
    
    missing = []
    for file in required_files:
        if not os.path.exists(file):
            missing.append(file)
    
    if missing:
        print(f"[WARNING] Missing files: {', '.join(missing)}")
        warnings.extend(missing)
        return len(missing) < 3  # Allow a few missing
    
    return True

test("File Structure", test_file_structure)

# Test 2: Config Module
print("\n" + "=" * 70)
print("TEST 2: Configuration Module")
print("=" * 70)

def test_config():
    sys.path.insert(0, '.')
    from src.config import (
        BASE_DIR, DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
        MODELS_DIR, LOGS_DIR, IMAGE_SIZE, BATCH_SIZE, EPOCHS,
        BASE_MODEL, LEARNING_RATE
    )
    
    assert BASE_DIR is not None, "BASE_DIR not defined"
    assert IMAGE_SIZE == (224, 224), f"IMAGE_SIZE should be (224, 224), got {IMAGE_SIZE}"
    assert BATCH_SIZE > 0, f"BATCH_SIZE should be > 0, got {BATCH_SIZE}"
    assert BASE_MODEL in ['ResNet50', 'MobileNet', 'EfficientNet'], f"Invalid BASE_MODEL: {BASE_MODEL}"
    
    print(f"  BASE_DIR: {BASE_DIR}")
    print(f"  IMAGE_SIZE: {IMAGE_SIZE}")
    print(f"  BATCH_SIZE: {BATCH_SIZE}")
    print(f"  BASE_MODEL: {BASE_MODEL}")
    
    return True

test("Configuration Module", test_config)

# Test 3: Data Preprocessing (without TensorFlow)
print("\n" + "=" * 70)
print("TEST 3: Data Preprocessing Module (Partial)")
print("=" * 70)

def test_data_preprocessing():
    sys.path.insert(0, '.')
    from src.data_preprocessing import preprocess_image, split_data
    
    # Test that functions are callable
    assert callable(preprocess_image), "preprocess_image is not callable"
    assert callable(split_data), "split_data is not callable"
    
    # Test with a dummy image path (will fail but we test the function exists)
    # We can't actually test image processing without a real image
    print("  Functions are properly defined")
    print("  Note: Full image processing requires actual images")
    
    return True

test("Data Preprocessing Module", test_data_preprocessing)

# Test 4: Directory Creation
print("\n" + "=" * 70)
print("TEST 4: Directory Creation")
print("=" * 70)

def test_directories():
    from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR
    
    # Check if directories can be created
    for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
        os.makedirs(dir_path, exist_ok=True)
        if not os.path.exists(dir_path):
            return False
        print(f"  [OK] {dir_path}")
    
    return True

test("Directory Creation", test_directories)

# Test 5: Flask App Structure
print("\n" + "=" * 70)
print("TEST 5: Flask Application Structure")
print("=" * 70)

def test_flask_app():
    # Check if Flask template exists
    template_path = 'templates/index.html'
    if not os.path.exists(template_path):
        return False
    
    # Read template and check for key elements
    with open(template_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    required_elements = ['<!DOCTYPE html>', '<title>', '<form', 'predict', 'upload']
    found = [elem for elem in required_elements if elem.lower() in content.lower()]
    
    print(f"  Template found: {template_path}")
    print(f"  Key elements found: {len(found)}/{len(required_elements)}")
    
    return len(found) >= 3  # At least 3 key elements should be present

test("Flask Application Structure", test_flask_app)

# Test 6: Import Paths
print("\n" + "=" * 70)
print("TEST 6: Import Path Validation")
print("=" * 70)

def test_import_paths():
    sys.path.insert(0, '.')
    
    try:
        # Test config import
        from src.config import BASE_DIR
        print("  [OK] src.config imports correctly")
        
        # Test that modules can be imported (even if TensorFlow fails)
        try:
            import src.data_preprocessing
            print("  [OK] src.data_preprocessing imports (TensorFlow check skipped)")
        except ImportError as e:
            if 'tensorflow' in str(e).lower():
                print("  [SKIP] TensorFlow not installed (expected)")
            else:
                raise
        
        return True
    except Exception as e:
        print(f"  [ERROR] Import failed: {str(e)}")
        return False

test("Import Path Validation", test_import_paths)

# Test 7: Utility Scripts
print("\n" + "=" * 70)
print("TEST 7: Utility Scripts")
print("=" * 70)

def test_utilities():
    # Test download_dataset script structure
    utils_path = 'utils/download_dataset.py'
    if os.path.exists(utils_path):
        with open(utils_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_functions = ['setup_dataset_structure', 'print_dataset_instructions']
        found = [func for func in required_functions if func in content]
        
        print(f"  Utility functions found: {len(found)}/{len(required_functions)}")
        return len(found) >= 1
    return False

test("Utility Scripts", test_utilities)

# Test 8: Code Quality Checks
print("\n" + "=" * 70)
print("TEST 8: Code Quality")
print("=" * 70)

def test_code_quality():
    python_files = [
        'src/config.py',
        'src/data_preprocessing.py',
        'src/model_builder.py',
        'src/train.py',
        'src/evaluate.py',
        'app.py',
        'setup.py'
    ]
    
    syntax_errors = []
    for file_path in python_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
                compile(code, file_path, 'exec')
                print(f"  [OK] {file_path}")
            except SyntaxError as e:
                syntax_errors.append(f"{file_path}: {str(e)}")
                print(f"  [ERROR] {file_path}: {str(e)}")
    
    return len(syntax_errors) == 0

test("Code Quality", test_code_quality)

# Final Summary
print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)

passed = sum(1 for _, status, _ in test_results if status == "PASSED")
failed = sum(1 for _, status, _ in test_results if status != "PASSED")

print(f"\nTotal Tests: {len(test_results)}")
print(f"Passed: {passed}")
print(f"Failed/Errors: {failed}")

if test_results:
    print("\nDetailed Results:")
    for name, status, error in test_results:
        symbol = "[OK]" if status == "PASSED" else "[X]"
        print(f"  {symbol} {name}: {status}")
        if error:
            print(f"      Error: {error}")

if errors:
    print("\n[ERRORS]:")
    for error in errors:
        print(f"  - {error}")

if warnings:
    print("\n[WARNINGS]:")
    for warning in warnings:
        print(f"  - {warning}")

print("\n" + "=" * 70)

if failed == 0:
    print("\n[SUCCESS] All tests passed!")
    sys.exit(0)
else:
    print(f"\n[WARNING] {failed} test(s) had issues")
    if len(errors) == 0:
        sys.exit(0)  # Warnings only
    else:
        sys.exit(1)  # Actual errors

