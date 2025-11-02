"""
Production server launcher for Retina-Based Heart Disease Predictor
"""
import os
import sys
import webbrowser
from threading import Timer

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def open_browser():
    """Open browser after a delay"""
    webbrowser.open('http://localhost:5000')

def main():
    """Start production server"""
    print("=" * 70)
    print("Retina-Based Heart Disease Predictor - Production Server")
    print("=" * 70)
    
    # Check if model exists
    from src.config import MODELS_DIR, MODEL_NAME
    model_path = os.path.join(MODELS_DIR, MODEL_NAME)
    
    if not os.path.exists(model_path):
        print("\n[WARNING] Trained model not found!")
        print(f"Model path: {model_path}")
        print("\nPlease train the model first:")
        print("  python src/train.py")
        print("\nOr use the demo mode to create sample data and train.")
        return
    
    print(f"\n[OK] Model found: {model_path}")
    print(f"    Size: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")
    
    # Check GPU
    print("\n" + "=" * 70)
    print("System Configuration")
    print("=" * 70)
    
    try:
        from src.gpu_utils import setup_gpu
        device_info = setup_gpu()
        print(f"Device: {device_info['device_name']}")
    except:
        print("Device: CPU/GPU check skipped")
    
    # Start Flask app
    print("\n" + "=" * 70)
    print("Starting Production Server")
    print("=" * 70)
    print("\nServer will be available at: http://localhost:5000")
    print("Press CTRL+C to stop the server\n")
    
    # Open browser after 2 seconds
    Timer(2.0, open_browser).start()
    
    # Import and run Flask app
    from app import app
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)

if __name__ == "__main__":
    main()

