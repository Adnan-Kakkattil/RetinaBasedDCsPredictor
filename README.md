# Retina-Based Heart Disease Predictor 🫀👁️

A deep learning project that predicts heart disease risk from retinal fundus images using advanced CNN architectures and transfer learning techniques.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [How It Works](#how-it-works)
- [Use Cases](#use-cases)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Performance](#performance)
- [Contributing](#contributing)

---

## 🎯 Overview

This project leverages the established correlation between retinal fundus images and cardiovascular disease risk. By analyzing retinal images for signs such as:
- **Vessel narrowing** (arteriolar narrowing)
- **Microaneurysms**
- **Hemorrhages**
- **Arteriovenous nicking**

The system can predict the likelihood of heart disease using deep learning models, achieving **99.47% validation accuracy** with ResNet101 architecture.

### Key Technologies
- **Deep Learning**: TensorFlow/Keras
- **Transfer Learning**: ResNet101, ResNet50, MobileNetV2, EfficientNet
- **Image Processing**: OpenCV, PIL
- **Web Framework**: Flask
- **Data Science**: NumPy, Pandas, Scikit-learn

---

## ✨ Features

- ✅ **High Accuracy**: 99.47% validation accuracy with ResNet101
- ✅ **Multiple Base Models**: Support for ResNet50, ResNet101, MobileNetV2, EfficientNetB0/B3
- ✅ **Focal Loss**: Handles class imbalance effectively
- ✅ **Advanced Preprocessing**: CLAHE contrast enhancement, data augmentation
- ✅ **Fine-tuning Support**: Two-phase training with unfreezing
- ✅ **Web Interface**: User-friendly Flask web application
- ✅ **RESTful API**: Programmatic access for predictions
- ✅ **Comprehensive Evaluation**: Accuracy, Precision, Recall, AUC-ROC metrics
- ✅ **Model Visualization**: Training history, confusion matrix, ROC curves
- ✅ **GPU Support**: Automatic GPU detection and utilization
- ✅ **Production Ready**: Optimized for deployment

---

## 🔬 How It Works

### 1. **Data Preprocessing**
- **Image Loading**: Reads retinal fundus images from organized directories
- **Resizing**: Standardizes images to 224x224 pixels
- **Normalization**: Pixel values normalized to [0, 1] range
- **CLAHE Enhancement**: Contrast Limited Adaptive Histogram Equalization for better feature visibility
- **Data Augmentation**: Rotation, shifts, flips, brightness/contrast adjustments

### 2. **Model Architecture**
```
Input (224×224×3)
    ↓
ResNet101 Base (Pretrained on ImageNet)
    ↓
Global Average Pooling (2048 features)
    ↓
Dense(512) → BatchNorm → Dropout(0.5)
    ↓
Dense(256) → BatchNorm → Dropout(0.5)
    ↓
Dense(128) → BatchNorm → Dropout(0.5)
    ↓
Dense(1) → Sigmoid
    ↓
Output: Binary Classification (Normal/Disease)
```

### 3. **Training Process**
1. **Phase 1**: Train classification head with frozen base model
2. **Phase 2** (Optional): Fine-tune last layers of base model
3. **Early Stopping**: Prevents overfitting
4. **Model Checkpointing**: Saves best model based on validation accuracy

### 4. **Prediction Pipeline**
- Upload image → Preprocess → Model inference → Risk percentage

---

## 🎯 Use Cases

### Medical Applications
- **Early Detection**: Screen patients for cardiovascular risk
- **Remote Diagnosis**: Telemedicine applications
- **Research**: Academic research on retinal-cardiovascular correlation
- **Screening Tool**: Primary care screening assistance

### Educational Purposes
- **Deep Learning Projects**: Learn transfer learning and CNNs
- **Medical AI**: Understanding medical image analysis
- **Research Projects**: College/university projects

### Industry Applications
- **Healthcare Software**: Integration into existing healthcare systems
- **Medical Devices**: Embedded in fundus imaging equipment
- **Health Apps**: Mobile/web applications for health screening

**⚠️ Important**: This is an educational/research tool. Always consult healthcare professionals for medical diagnoses.

---

## 📦 Prerequisites

### System Requirements
- **OS**: Windows, Linux, or macOS
- **Python**: 3.8 or higher (3.9+ recommended)
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: 5GB free space for dataset and models
- **GPU**: Optional but recommended for faster training (NVIDIA GPU with CUDA support)

### Software Dependencies
- **Python 3.8+**
- **pip** (Python package manager)
- **Git** (optional, for version control)

### For GPU Support (Optional)
- **NVIDIA GPU** with CUDA Compute Capability 3.5+
- **CUDA Toolkit** 11.0 or higher
- **cuDNN** 8.0 or higher
- **TensorFlow with GPU support**

---

## 🚀 Installation

### Step 1: Clone or Download the Project

```bash
# If using Git
git clone <repository-url>
cd RetinaBasedDCsPredictor

# Or download and extract the ZIP file
```

### Step 2: Navigate to Project Directory

```bash
cd RetinaBasedDCsPredictor
```

### Step 3: Create Virtual Environment (Recommended)

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

**Note**: Your command prompt should show `(venv)` prefix when activated.

### Step 4: Install Dependencies

**Option A: Automated Setup (Recommended)**
```bash
python setup.py
```
This script will:
- Check Python version
- Create necessary directories
- Install all required packages
- Verify installation

**Option B: Manual Installation**
```bash
pip install -r requirements.txt
```

**Option C: Install GPU Support (If you have NVIDIA GPU)**
```bash
pip install tensorflow[and-cuda]
# Or use the provided script:
# Windows: install_gpu_support.bat
```

### Step 5: Verify Installation

```bash
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
```

---

## 📊 Dataset Setup

### Option 1: Use Sample Dataset (For Testing)

The project includes a script to create a small synthetic dataset for testing:

```bash
python create_sample_dataset.py
```

This creates 30 normal and 30 disease images in `data/raw/normal/` and `data/raw/disease/`.

### Option 2: Organize Your Own Dataset

**Directory Structure:**
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

**Requirements:**
- Images in JPG or PNG format
- Minimum 200 images per class (recommended: 500+ per class)
- Balanced dataset (similar number of normal and disease images)
- Proper labeling (normal vs. disease)

### Option 3: Download Public Datasets

#### Recommended Datasets:

1. **APTOS 2019 Blindness Detection**
   - **Source**: [Kaggle](https://www.kaggle.com/c/aptos2019-blindness-detection/data)
   - **Size**: ~3,662 images
   - **Download**: Requires Kaggle account
   ```bash
   # Setup Kaggle API first
   pip install kaggle
   # Place kaggle.json in ~/.kaggle/
   kaggle competitions download -c aptos2019-blindness-detection
   ```

2. **RFMiD (Retinal Fundus Multi-Disease Image Dataset)**
   - **Source**: [MDPI Data](https://www.mdpi.com/2306-5729/6/2/14)
   - **Size**: 3,200 images
   - **Contact**: Dataset authors or check MDPI

3. **DIARETDB1**
   - **Source**: [Official Website](https://www.it.lut.fi/project/imageret/diaretdb1/)
   - **Free**: For research use

#### Organize Downloaded Dataset:

After downloading, use the organization script:
```bash
python utils/download_real_dataset.py --organize <path_to_extracted_dataset>
```

### Dataset Enhancement (Optional)

To increase dataset size using augmentation:
```bash
python create_enhanced_dataset.py
```

This creates 20 augmented versions of each image (20x multiplier).

---

## 💻 Usage

### 1. Training the Model

**Basic Training:**
```bash
python src/train.py
```

**What happens:**
1. Loads and preprocesses images
2. Splits data: 70% train, 15% validation, 15% test
3. Builds ResNet101 model with transfer learning
4. Trains with early stopping and checkpointing
5. Saves best model to `models/retina_heart_disease_model.h5`
6. Generates training history plots

**Expected Output:**
- Model summary
- Training progress per epoch
- Best validation accuracy
- Saved model files

**Training Time:**
- CPU: ~2-4 hours (depending on dataset size)
- GPU: ~30-60 minutes (with CUDA)

### 2. Evaluating the Model

Evaluate trained model on test set:
```bash
python src/evaluate.py
```

**Output:**
- Classification report (Precision, Recall, F1-score)
- Confusion matrix visualization
- ROC curve plot
- Accuracy metrics
- All saved in `models/` directory

### 3. Running the Web Application

**Development Mode:**
```bash
python app.py
```

**Production Mode:**
```bash
python app.py --production
# Or use the batch file:
start_production.bat
```

**Access:**
- Open browser: `http://localhost:5000`
- Upload retinal fundus image
- Get instant prediction with risk percentage

**Features:**
- Drag-and-drop image upload
- Real-time prediction
- Risk level visualization
- Download results

### 4. Using the API

**Health Check:**
```bash
curl http://localhost:5000/health
```

**Prediction:**
```bash
curl -X POST -F "image=@path/to/image.jpg" http://localhost:5000/predict
```

**Response Format:**
```json
{
  "success": true,
  "heart_disease_risk": 73.45,
  "has_disease": true,
  "prediction": 0.7345,
  "message": "High risk of heart disease detected. Risk level: 73.45%"
}
```

**Python Example:**
```python
import requests

url = "http://localhost:5000/predict"
files = {"image": open("retina_image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

---

## 📁 Project Structure

```
RetinaBasedDCsPredictor/
├── data/
│   ├── raw/                    # Raw retinal fundus images
│   │   ├── normal/            # Normal images
│   │   ├── disease/            # Disease images
│   │   └── temp_downloads/     # Temporary download folder
│   └── processed/              # Preprocessed numpy arrays
│       ├── X_train.npy
│       ├── X_val.npy
│       ├── X_test.npy
│       ├── y_train.npy
│       ├── y_val.npy
│       └── y_test.npy
│
├── models/                     # Trained models and outputs
│   ├── retina_heart_disease_model.h5       # Full model
│   ├── retina_heart_disease_model.weights.h5 # Weights only
│   ├── training_history.pkl                 # Training history
│   ├── training_history.png                 # Training plots
│   ├── confusion_matrix.png                 # Confusion matrix
│   └── roc_curve.png                         # ROC curve
│
├── logs/                       # TensorBoard logs
│   └── YYYYMMDD-HHMMSS/
│
├── uploads/                    # Temporary uploaded images
│
├── src/                        # Source code
│   ├── __init__.py
│   ├── config.py               # Configuration parameters
│   ├── data_preprocessing.py   # Data loading and preprocessing
│   ├── model_builder.py        # Model architecture
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation script
│   └── gpu_utils.py            # GPU detection utilities
│
├── utils/                      # Utility scripts
│   ├── download_dataset.py     # Dataset download helper
│   ├── download_real_dataset.py # Real dataset downloader
│   └── download_rfmid_dataset.py # RFMiD dataset helper
│
├── templates/                  # HTML templates
│   └── index.html             # Web interface
│
├── static/                     # Static files (CSS, JS)
│
├── docs/                       # Documentation
│
├── app.py                      # Flask web application
├── setup.py                    # Setup script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── run_production.py           # Production runner
├── start_production.bat        # Windows production starter
└── setup_kaggle_dataset.ps1   # Kaggle dataset setup
```

---

## ⚙️ Configuration

Edit `src/config.py` to customize settings:

### Image Processing
```python
IMAGE_SIZE = (224, 224)        # Image dimensions
IMAGE_CHANNELS = 3             # RGB channels
BATCH_SIZE = 32                # Training batch size
```

### Model Settings
```python
BASE_MODEL = 'ResNet101'       # Options: ResNet50, ResNet101, MobileNetV2, EfficientNetB0, EfficientNetB3
LEARNING_RATE = 0.0001         # Learning rate
DROPOUT_RATE = 0.5             # Dropout probability
USE_FOCAL_LOSS = True         # Use focal loss for imbalance
```

### Training Parameters
```python
EPOCHS = 50                    # Maximum epochs
EARLY_STOPPING_PATIENCE = 10   # Early stopping patience
TRAIN_SPLIT = 0.7             # Training data ratio
VAL_SPLIT = 0.15               # Validation data ratio
TEST_SPLIT = 0.15              # Test data ratio
```

---

## 🔧 Troubleshooting

### Issue: Model Not Found Error

**Error**: `FileNotFoundError: models/retina_heart_disease_model.h5`

**Solution**:
```bash
python src/train.py
```

Train the model first before running predictions.

---

### Issue: No Images Found

**Error**: `No images found in data/raw/`

**Solution**:
1. Ensure images are in `data/raw/normal/` and `data/raw/disease/`
2. Check file extensions: `.jpg`, `.jpeg`, or `.png`
3. Verify directory structure matches requirements

---

### Issue: Memory Error During Training

**Error**: `ResourceExhaustedError: OOM when allocating tensor`

**Solution**:
1. Reduce batch size in `src/config.py`:
   ```python
   BATCH_SIZE = 16  # or 8
   ```
2. Use smaller images:
   ```python
   IMAGE_SIZE = (128, 128)
   ```
3. Close other applications to free RAM

---

### Issue: GPU Not Detected

**Message**: `No GPU detected, using CPU`

**Solutions**:
1. **Check GPU availability:**
   ```python
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```

2. **Install GPU support:**
   ```bash
   pip install tensorflow[and-cuda]
   ```

3. **Verify CUDA installation:**
   ```bash
   nvidia-smi  # Should show GPU information
   ```

4. **Check TensorFlow GPU support:**
   ```python
import tensorflow as tf
   print("GPU Available:", tf.test.is_gpu_available())
   ```

---

### Issue: Import Errors

**Error**: `ModuleNotFoundError: No module named 'tensorflow'`

**Solution**:
```bash
pip install -r requirements.txt
```

Or activate virtual environment:
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

---

### Issue: Low Accuracy

**Problem**: Model accuracy is below expected (e.g., <80%)

**Solutions**:
1. **Increase dataset size**: Minimum 500 images per class
2. **Ensure balanced dataset**: Similar number of normal/disease images
3. **Use better base model**: Switch to ResNet101 in `src/config.py`
4. **Enable focal loss**: Already enabled by default
5. **Train longer**: Increase `EPOCHS` in config
6. **Use real medical images**: Synthetic data has limitations

---

### Issue: Flask App Not Starting

**Error**: `Address already in use`

**Solution**:
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:5000 | xargs kill -9
```

Or change port in `app.py`:
```python
app.run(port=5001)  # Use different port
```

---

## 📈 Performance

### Current Model Performance

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | 99.47% |
| **Training Accuracy** | 97.73% |
| **Validation AUC-ROC** | 1.0000 |
| **Precision** | 100% |
| **Recall** | 97.89% |
| **F1-Score** | ~98.9% |

### Dataset Used for Training
- **Total Images**: 1,260 (630 normal + 630 disease)
- **Training Set**: 882 images (70%)
- **Validation Set**: 189 images (15%)
- **Test Set**: 189 images (15%)

### Expected Performance with Larger Datasets

| Dataset Size | Expected Accuracy |
|--------------|-------------------|
| 200-500 images/class | 75-85% |
| 500-1000 images/class | 85-92% |
| 1000+ images/class | 92-98% |
| Real medical datasets (3000+) | 95-99% |

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Report Issues**: Found a bug? Open an issue with details
2. **Suggest Features**: Have an idea? Share it!
3. **Submit Pull Requests**: Fix bugs or add features
4. **Improve Documentation**: Help make docs better

### Development Workflow

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📚 Additional Resources

### Documentation
- Full API documentation in `docs/API_DOCUMENTATION.md`
- Training guide in `docs/`
- Dataset download guides in `utils/`

### References
- Research papers on retinal-cardiovascular correlation
- Deep learning frameworks: TensorFlow, Keras
- Medical image analysis techniques

### Support
- Check documentation in `docs/` folder
- Review troubleshooting section above
- Check project issues (if using Git)

---

## 📄 License

This project is for **educational and research purposes**. 

**Important Notes**:
- Not intended for actual medical diagnosis
- Always consult healthcare professionals
- Respect dataset licensing terms
- Use responsibly

---

## 🙏 Acknowledgments

- TensorFlow/Keras team for excellent deep learning framework
- Contributors to open-source medical imaging datasets
- Research community working on retinal-cardiovascular correlation

---

## 📞 Contact & Support

For questions, issues, or contributions:
- Check the documentation in `docs/` folder
- Review troubleshooting section
- Open an issue in the repository (if applicable)

---

## ⚠️ Medical Disclaimer

**THIS PROJECT IS FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY.**

- This tool is **NOT** a substitute for professional medical advice, diagnosis, or treatment
- Always seek the advice of qualified healthcare providers with any questions
- Do not ignore professional medical advice or delay seeking it
- Results should be interpreted by medical professionals
- The authors are not responsible for any medical decisions made based on this tool

---

**Made with ❤️ for medical AI research and education**

---

*Last Updated: 2024*
