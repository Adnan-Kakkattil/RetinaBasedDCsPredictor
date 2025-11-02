"""
Data preprocessing module for retinal fundus images
"""
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle

def preprocess_image(image_path, target_size=(224, 224), enhance_contrast=True):
    """
    Preprocess a single image with advanced techniques
    
    Args:
        image_path: Path to the image file
        target_size: Tuple of (height, width) for resizing
        enhance_contrast: Whether to apply contrast enhancement
    
    Returns:
        Preprocessed image as numpy array
    """
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert BGR to RGB (OpenCV reads as BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if enhance_contrast:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels and convert back to RGB
            lab = cv2.merge([l, a, b])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Resize image with high-quality interpolation
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        
        # Normalize pixel values to [0, 1] using ImageNet normalization
        # Alternative: simple normalization
        image = image.astype(np.float32) / 255.0
        
        # Optional: Apply ImageNet normalization statistics
        # mean = np.array([0.485, 0.456, 0.406])
        # std = np.array([0.229, 0.224, 0.225])
        # image = (image - mean) / std
        
        return image
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {str(e)}")
        return None

def load_dataset_from_directory(data_dir, labels_file=None):
    """
    Load dataset from directory structure or labels file
    
    Args:
        data_dir: Directory containing images
        labels_file: Optional CSV file with image paths and labels
    
    Returns:
        Tuple of (images, labels)
    """
    images = []
    labels = []
    image_paths = []
    
    if labels_file and os.path.exists(labels_file):
        # Load from CSV file
        df = pd.read_csv(labels_file)
        for idx, row in df.iterrows():
            img_path = os.path.join(data_dir, row['image_path'])
            if os.path.exists(img_path):
                image = preprocess_image(img_path)
                if image is not None:
                    images.append(image)
                    labels.append(row['label'])  # Assuming binary: 0=no disease, 1=disease
                    image_paths.append(img_path)
    else:
        # Load from directory structure (assuming structure: data/raw/class_name/image.jpg)
        for class_name in ['normal', 'disease']:
            class_dir = os.path.join(data_dir, class_name)
            if os.path.exists(class_dir):
                for filename in os.listdir(class_dir):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_dir, filename)
                        image = preprocess_image(img_path)
                        if image is not None:
                            images.append(image)
                            labels.append(0 if class_name == 'normal' else 1)
                            image_paths.append(img_path)
    
    return np.array(images), np.array(labels), image_paths

def create_data_generators(train_dir, val_dir, test_dir=None, batch_size=32):
    """
    Create data generators with advanced augmentation for training
    
    Args:
        train_dir: Training data directory
        val_dir: Validation data directory
        test_dir: Test data directory (optional)
        batch_size: Batch size for training
    
    Returns:
        Tuple of (train_gen, val_gen, test_gen)
    """
    # Advanced data augmentation for training
    train_datagen = ImageDataGenerator(
        rotation_range=30,  # Increased rotation
        width_shift_range=0.25,  # Increased shift
        height_shift_range=0.25,
        shear_range=0.2,  # Added shear transformation
        zoom_range=0.25,  # Increased zoom
        horizontal_flip=True,
        vertical_flip=True,  # Enabled vertical flip
        fill_mode='reflect',  # Better fill mode
        brightness_range=[0.7, 1.3],  # Increased brightness range
        channel_shift_range=20.0,  # Color augmentation
        preprocessing_function=lambda x: x / 255.0  # Normalization
    )
    
    # Minimal augmentation for validation (only normalization)
    val_test_datagen = ImageDataGenerator(
        preprocessing_function=lambda x: x / 255.0
    )
    
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary',
        color_mode='rgb'
    )
    
    val_gen = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary',
        color_mode='rgb'
    )
    
    test_gen = None
    if test_dir and os.path.exists(test_dir):
        test_gen = val_test_datagen.flow_from_directory(
            test_dir,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='binary',
            color_mode='rgb',
            shuffle=False
        )
    
    return train_gen, val_gen, test_gen

def split_data(images, labels, train_split=0.7, val_split=0.15, test_split=0.15, random_state=42):
    """
    Split data into train, validation, and test sets
    
    Args:
        images: Array of images
        labels: Array of labels
        train_split: Proportion of training data
        val_split: Proportion of validation data
        test_split: Proportion of test data
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # First split: train + val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        images, labels, test_size=test_split, random_state=random_state, stratify=labels
    )
    
    # Second split: train vs val
    val_size = val_split / (train_split + val_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def save_preprocessed_data(X_train, X_val, X_test, y_train, y_val, y_test, save_dir):
    """
    Save preprocessed data to disk
    
    Args:
        X_train, X_val, X_test: Image arrays
        y_train, y_val, y_test: Label arrays
        save_dir: Directory to save preprocessed data
    """
    os.makedirs(save_dir, exist_ok=True)
    
    np.save(os.path.join(save_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(save_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(save_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(save_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(save_dir, 'y_test.npy'), y_test)
    
    print(f"Preprocessed data saved to {save_dir}")

if __name__ == "__main__":
    # Example usage
    from config import RAW_DATA_DIR, PROCESSED_DATA_DIR
    
    print("Loading and preprocessing dataset...")
    images, labels, paths = load_dataset_from_directory(RAW_DATA_DIR)
    
    if len(images) > 0:
        print(f"Loaded {len(images)} images")
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(images, labels)
        save_preprocessed_data(X_train, X_val, X_test, y_train, y_val, y_test, PROCESSED_DATA_DIR)
        print("Preprocessing completed!")
    else:
        print("No images found! Please ensure images are in the data/raw directory.")

