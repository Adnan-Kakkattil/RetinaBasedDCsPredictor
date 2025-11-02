"""
Configuration file for the Retina-Based Heart Disease Predictor project
"""
import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Image processing parameters
IMAGE_SIZE = (224, 224)
IMAGE_CHANNELS = 3
BATCH_SIZE = 32
EPOCHS = 50

# Data split
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Model parameters
BASE_MODEL = 'ResNet101'  # Options: ResNet50, ResNet101, MobileNetV2, EfficientNetB0, EfficientNetB3
LEARNING_RATE = 0.0001
DROPOUT_RATE = 0.5
USE_FOCAL_LOSS = True  # Use focal loss for class imbalance

# Training parameters
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5

# Output
MODEL_NAME = 'retina_heart_disease_model.h5'
HISTORY_NAME = 'training_history.pkl'

