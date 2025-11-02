"""
Flask web application for retina-based heart disease prediction
"""
import os
import sys
import numpy as np
import tensorflow as tf
import cv2
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.config import *

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model
model = None
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_NAME)

def load_model():
    """Load the trained model"""
    global model
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}...")
        try:
            # Try loading with custom objects for focal loss
            from src.model_builder import focal_loss
            custom_objects = {'focal_loss_fn': focal_loss(gamma=2.0, alpha=0.25)}
            model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
        except:
            try:
                # Try standard load
                model = tf.keras.models.load_model(MODEL_PATH)
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                print("Attempting to rebuild model...")
                from src.model_builder import build_model
                from src.config import BASE_MODEL, IMAGE_SIZE, IMAGE_CHANNELS, USE_FOCAL_LOSS
                model = build_model(
                    base_model_name=BASE_MODEL,
                    input_shape=(*IMAGE_SIZE, IMAGE_CHANNELS),
                    num_classes=1,
                    use_focal_loss=USE_FOCAL_LOSS
                )
                # Try loading weights
                weights_path = MODEL_PATH.replace('.h5', '_weights.h5')
                if os.path.exists(weights_path):
                    model.load_weights(weights_path)
                    print("Model weights loaded!")
                else:
                    print("WARNING: Could not load model weights")
                    model = None
                    return
        print("Model loaded successfully!")
    else:
        print(f"WARNING: Model not found at {MODEL_PATH}")
        print("Please train the model first using: python src/train.py")

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    """Preprocess image for prediction"""
    try:
        # Read and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, IMAGE_SIZE)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict heart disease risk from uploaded image"""
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG.'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess image
        processed_image = preprocess_image(filepath)
        
        if processed_image is None:
            return jsonify({'error': 'Failed to process image'}), 400
        
        # Make prediction
        prediction = model.predict(processed_image, verbose=0)[0][0]
        
        # Calculate risk percentage
        risk_percentage = float(prediction * 100)
        has_disease = prediction > 0.5
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'heart_disease_risk': risk_percentage,
            'has_disease': bool(has_disease),
            'prediction': float(prediction),
            'message': f"{'High risk' if has_disease else 'Low risk'} of heart disease detected. Risk level: {risk_percentage:.2f}%"
        })
    
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    model_status = 'loaded' if model is not None else 'not loaded'
    return jsonify({
        'status': 'healthy',
        'model_status': model_status,
        'model_path': MODEL_PATH
    })

if __name__ == '__main__':
    import sys
    # Check if running in production mode
    production = '--production' in sys.argv or 'production' in sys.argv
    
    print("=" * 70)
    print("Retina-Based Heart Disease Predictor - Web Application")
    print("=" * 70)
    
    print("\nLoading model...")
    load_model()
    
    if production:
        print("\n[PRODUCTION MODE]")
        print("Server running at: http://0.0.0.0:5000")
        print("Access from: http://localhost:5000")
        print("Press CTRL+C to stop\n")
        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
    else:
        print("\n[DEVELOPMENT MODE]")
        print("Server running at: http://localhost:5000")
        print("Debug mode: Enabled")
        print("Press CTRL+C to stop\n")
        app.run(debug=True, host='127.0.0.1', port=5000)

