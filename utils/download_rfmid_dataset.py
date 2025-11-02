"""
Script to download and prepare RFMiD (Retinal Fundus Multi-Disease Image Dataset)
Based on research: RFMiD dataset with 3,200 images and 46 conditions
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import RAW_DATA_DIR

def print_rfmid_instructions():
    """
    Print instructions for downloading RFMiD dataset
    """
    instructions = """
    ============================================================
    RFMiD Dataset Download Instructions
    ============================================================
    
    The Retinal Fundus Multi-Disease Image Dataset (RFMiD) is one of the 
    most comprehensive datasets for retinal fundus image classification,
    containing 3,200 images with 46 different disease conditions.
    
    SOURCES:
    
    1. Official RFMiD Dataset:
       - Paper: https://www.mdpi.com/2306-5729/6/2/14
       - Dataset: Available through research repositories
       - Contains 3,200 images with multi-label annotations
    
    2. Kaggle Alternatives:
       - Search: "retinal fundus images"
       - "APTOS 2019 Blindness Detection"
       - "Diabetic Retinopathy Detection"
    
    3. Other High-Quality Datasets:
       - EyePACS (requires registration)
       - IDRiD (Indian Diabetic Retinopathy Image Dataset)
       - BRSET (Brazilian Multilabel Ophthalmological Dataset)
    
    ORGANIZATION:
    
    After downloading, organize your dataset as follows:
    
    data/raw/
      normal/
        image1.jpg
        image2.jpg
        ...
      disease/
        image1.jpg
        image2.jpg
        ...
    
    OR (for CSV format):
    
    data/raw/
      images/
        img001.jpg
        img002.jpg
        ...
      labels.csv (columns: image_path, label)
    
    MINIMUM REQUIREMENTS:
    
    For good accuracy, you need:
    - At least 500+ images per class
    - Balanced dataset (similar number of normal/disease)
    - High-quality, properly labeled images
    
    RECOMMENDED:
    - 1000+ images per class for production use
    - Professional medical annotation
    - Diverse patient demographics
    
    ============================================================
    """
    print(instructions)

def create_dataset_structure():
    """Create directory structure for dataset"""
    os.makedirs(os.path.join(RAW_DATA_DIR, 'normal'), exist_ok=True)
    os.makedirs(os.path.join(RAW_DATA_DIR, 'disease'), exist_ok=True)
    print(f"[OK] Created dataset directories in {RAW_DATA_DIR}")

if __name__ == "__main__":
    print("=" * 60)
    print("RFMiD Dataset Setup Helper")
    print("=" * 60)
    
    print("\n[1/2] Creating directory structure...")
    create_dataset_structure()
    
    print("\n[2/2] Dataset download instructions:")
    print_rfmid_instructions()
    
    print("\n" + "=" * 60)
    print("Next Steps:")
    print("1. Download RFMiD or similar high-quality dataset")
    print("2. Organize images in data/raw/normal/ and data/raw/disease/")
    print("3. Ensure balanced dataset (recommended: 1000+ per class)")
    print("4. Run: python src/train.py")
    print("=" * 60)

