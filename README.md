# Retina-Based Heart Disease Predictor 🫀

A college project for predicting heart disease risk from retinal fundus images using deep learning and transfer learning techniques.

## Overview

This project leverages the established correlation between retinal images (retinopathy signs) and cardiovascular disease risk. Using machine learning, especially convolutional neural networks (CNNs), retinal fundus images are analyzed for features such as vessel narrowing, microaneurysms, and hemorrhages, which are early indicators of heart disease.

The project implements:
- **Transfer Learning**: Uses pretrained models (ResNet50, MobileNetV2, EfficientNetB0)
- **Deep Learning**: CNN-based binary classification for heart disease risk
- **Web Interface**: Flask-based web application for real-time predictions
- **Complete Pipeline**: From data preprocessing to model deployment

## Project Structure

```
RetinaBasedDCsPredictor/
├── data/
│   ├── raw/              # Raw retinal fundus images
│   │   ├── normal/       # Normal images
│   │   └── disease/      # Disease images
│   └── processed/        # Preprocessed data (numpy arrays)
├── models/               # Trained models and checkpoints
├── src/
│   ├── config.py         # Configuration parameters
│   ├── data_preprocessing.py  # Data loading and preprocessing
│   ├── model_builder.py  # Model architecture
│   ├── train.py          # Training script
│   └── evaluate.py       # Evaluation script
├── utils/
│   └── download_dataset.py  # Dataset download helper
├── templates/
│   └── index.html        # Web interface
├── app.py                # Flask web application
├── setup.py              # Setup script
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Features

- ✅ Transfer learning with multiple base models (ResNet50, MobileNetV2, EfficientNetB0)
- ✅ Comprehensive data preprocessing with augmentation
- ✅ Automatic train/validation/test split
- ✅ Model checkpointing and early stopping
- ✅ Complete evaluation metrics (Accuracy, Precision, Recall, AUC-ROC)
- ✅ Beautiful web interface for predictions
- ✅ RESTful API for predictions
- ✅ Training history visualization

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone or Download the Project

```bash
cd RetinaBasedDCsPredictor
```

### Step 2: Set Up Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

**Option A: Use the setup script (Recommended)**
```bash
python setup.py
```

**Option B: Manual installation**
```bash
pip install -r requirements.txt
```

## Dataset Setup

### Step 1: Prepare Dataset

You need retinal fundus images organized in one of two ways:

**Option A: Directory Structure**
```
data/raw/
├── normal/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── disease/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

**Option B: CSV Format**
Create `data/raw/labels.csv` with columns:
```csv
image_path,label
images/img001.jpg,0
images/img002.jpg,1
...
```
Where `label` is 0 for normal, 1 for heart disease.

### Step 2: Download Dataset Helper

Run the dataset setup script for instructions:
```bash
python utils/download_dataset.py
```

### Recommended Datasets

1. **DIARETDB1** (Diabetic Retinopathy Database)
   - Website: https://www.it.lut.fi/project/imageret/diaretdb1/
   - Free for research use
   - Contains retinal images with annotations

2. **Kaggle Datasets**
   - Diabetic Retinopathy Detection: https://www.kaggle.com/c/diabetic-retinopathy-detection
   - APTOS 2019 Blindness Detection: https://www.kaggle.com/c/aptos2019-blindness-detection
   - Note: You may need to combine with cardiovascular risk datasets

3. **EyePACS Dataset**
   - Requires registration
   - Large collection of retinal images

## Usage

### 1. Training the Model

After organizing your dataset:

```bash
python src/train.py
```

This will:
- Load and preprocess images
- Split data into train/validation/test sets
- Build model with transfer learning
- Train with early stopping and checkpointing
- Save trained model to `models/retina_heart_disease_model.h5`
- Generate training history plots

**Training Parameters** (can be modified in `src/config.py`):
- Image size: 224x224
- Batch size: 32
- Epochs: 50 (with early stopping)
- Base model: ResNet50
- Learning rate: 0.0001

### 2. Evaluating the Model

Evaluate the trained model on test data:

```bash
python src/evaluate.py
```

This generates:
- Classification report
- Confusion matrix visualization
- ROC curve
- Accuracy, Precision, Recall, AUC-ROC metrics

### 3. Running the Web Application

Start the Flask web server:

```bash
python app.py
```

Then open your browser and navigate to:
```
http://localhost:5000
```

Upload a retinal fundus image to get real-time heart disease risk prediction.

### 4. API Usage

The Flask app provides a REST API:

**Endpoint:** `POST /predict`

**Request:**
```bash
curl -X POST -F "image=@path/to/image.jpg" http://localhost:5000/predict
```

**Response:**
```json
{
  "success": true,
  "heart_disease_risk": 65.23,
  "has_disease": true,
  "prediction": 0.6523,
  "message": "High risk of heart disease detected. Risk level: 65.23%"
}
```

## Configuration

Edit `src/config.py` to customize:

- **Image Processing**: Size, channels, batch size
- **Model Architecture**: Base model, dropout rate, learning rate
- **Training**: Epochs, early stopping patience, data splits
- **Paths**: Data directories, model save locations

## Model Architecture

The model uses transfer learning:

1. **Base Model**: Pretrained CNN (ResNet50, MobileNetV2, or EfficientNetB0) on ImageNet
2. **Feature Extraction**: Global Average Pooling
3. **Classification Head**:
   - Dense(256) + BatchNorm + Dropout
   - Dense(128) + BatchNorm + Dropout
   - Dense(1) with Sigmoid activation (binary classification)

## Evaluation Metrics

The model is evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **AUC-ROC**: Area under the receiver operating characteristic curve
- **Confusion Matrix**: Visual representation of predictions

## Project Workflow

1. **Data Collection**: Download retinal fundus images with heart disease labels
2. **Preprocessing**: Resize, normalize, and augment images
3. **Feature Extraction**: Use pretrained CNN for feature extraction
4. **Fine-tuning**: Fine-tune CNN on dataset for binary classification
5. **Evaluation**: Assess model on test data with multiple metrics
6. **Deployment**: Create web interface for predictions

## Dependencies

- **numpy**: Numerical computations
- **pandas**: Data manipulation
- **matplotlib**: Plotting and visualization
- **scikit-learn**: Machine learning utilities
- **tensorflow**: Deep learning framework
- **keras**: High-level neural network API
- **opencv-python**: Image processing
- **flask**: Web framework
- **scikit-image**: Image processing
- **tqdm**: Progress bars
- **seaborn**: Statistical visualization

## Troubleshooting

### Model Not Found Error
If you get "Model not found", make sure you've trained the model first:
```bash
python src/train.py
```

### No Images Found Error
Ensure your dataset is organized correctly:
- Images in `data/raw/normal/` and `data/raw/disease/` directories, OR
- CSV file at `data/raw/labels.csv` with correct format

### Memory Issues
Reduce batch size in `src/config.py`:
```python
BATCH_SIZE = 16  # or lower
```

### GPU Not Detected
The code will automatically use CPU if GPU is not available. For GPU support:
- Install CUDA and cuDNN
- Install tensorflow-gpu instead of tensorflow

## Future Improvements

- [ ] Fine-tuning support (unfreezing base layers)
- [ ] Support for multi-class classification
- [ ] Integration with additional datasets
- [ ] Docker containerization
- [ ] Model explainability (Grad-CAM visualization)
- [ ] Batch prediction API
- [ ] Database integration for storing predictions

## References

- Research papers on heart disease prediction from retinal images
- Open access articles on retinal image analysis for cardiovascular risk assessment
- Deep learning frameworks: TensorFlow, Keras documentation

## License

This project is for educational purposes. Please ensure proper licensing when using datasets.

## Contributing

This is a college project. Contributions and suggestions are welcome!

## Contact

For questions or issues, please refer to the project documentation or create an issue in the repository.

---

**Note**: This project is for educational and research purposes. Always consult healthcare professionals for medical diagnoses.
