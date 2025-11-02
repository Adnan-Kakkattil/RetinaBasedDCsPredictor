"""
Helper script to download and prepare retinal fundus image datasets
"""
import os
import sys
import requests
import zipfile
import tarfile
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import RAW_DATA_DIR

def download_file(url, destination):
    """
    Download a file from URL
    
    Args:
        url: URL of the file to download
        destination: Local path to save the file
    """
    print(f"Downloading from {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    with open(destination, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}%", end='', flush=True)
    
    print("\nDownload complete!")
    return destination

def extract_archive(archive_path, extract_to):
    """
    Extract zip or tar archive
    
    Args:
        archive_path: Path to archive file
        extract_to: Directory to extract to
    """
    print(f"Extracting {archive_path} to {extract_to}...")
    os.makedirs(extract_to, exist_ok=True)
    
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.endswith(('.tar', '.tar.gz', '.tgz')):
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            tar_ref.extractall(extract_to)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")
    
    print("Extraction complete!")

def setup_dataset_structure():
    """
    Create directory structure for dataset
    """
    os.makedirs(os.path.join(RAW_DATA_DIR, 'normal'), exist_ok=True)
    os.makedirs(os.path.join(RAW_DATA_DIR, 'disease'), exist_ok=True)
    print(f"Created dataset directories in {RAW_DATA_DIR}")

def print_dataset_instructions():
    """
    Print instructions for downloading datasets
    """
    instructions = """
    ============================================================
    DATASET DOWNLOAD INSTRUCTIONS
    ============================================================
    
    For this project, you need retinal fundus images with heart disease labels.
    Here are recommended datasets:
    
    1. DIARETDB1 (Diabetic Retinopathy Database):
       - Download from: https://www.it.lut.fi/project/imageret/diaretdb1/
       - Free for research use
       - Contains retinal images with annotations
    
    2. Kaggle Datasets:
       - Diabetic Retinopathy Detection: https://www.kaggle.com/c/diabetic-retinopathy-detection
       - APTOS 2019 Blindness Detection: https://www.kaggle.com/c/aptos2019-blindness-detection
       - You may need to combine with cardiovascular risk datasets
    
    3. EyePACS Dataset:
       - Requires registration
       - Contains large collection of retinal images
    
    DATASET STRUCTURE:
    After downloading, organize your data as follows:
    
    data/raw/
      normal/
        image1.jpg
        image2.jpg
        ...
      disease/
        image1.jpg
        image2.jpg
        ...
    
    ALTERNATIVE: CSV Format
    If you have a CSV file with image paths and labels:
    
    Create a file: data/raw/labels.csv
    With columns:
    - image_path: path to image file
    - label: 0 for normal, 1 for heart disease
    
    Example:
    image_path,label
    images/img001.jpg,0
    images/img002.jpg,1
    ...
    
    ============================================================
    """
    print(instructions)

def create_sample_structure():
    """
    Create a sample structure for demonstration
    """
    sample_dir = os.path.join(RAW_DATA_DIR, 'sample_structure.txt')
    with open(sample_dir, 'w', encoding='utf-8') as f:
        f.write("""
SAMPLE DATASET STRUCTURE:

data/raw/
  normal/
    normal_001.jpg
    normal_002.jpg
    ...
  disease/
    disease_001.jpg
    disease_002.jpg
    ...

OR

data/raw/
  images/
    img001.jpg
    img002.jpg
    ...
  labels.csv (with columns: image_path, label)

After organizing your data, run:
    python src/train.py
        """)
    print(f"Sample structure guide created at {sample_dir}")

if __name__ == "__main__":
    print("=" * 60)
    print("Retina-Based Heart Disease Predictor - Dataset Setup")
    print("=" * 60)
    
    print("\n[1/3] Creating dataset directory structure...")
    setup_dataset_structure()
    
    print("\n[2/3] Creating sample structure guide...")
    create_sample_structure()
    
    print("\n[3/3] Dataset setup instructions:")
    print_dataset_instructions()
    
    print("\n" + "=" * 60)
    print("Next steps:")
    print("1. Download a retinal fundus image dataset")
    print("2. Organize images in data/raw/normal/ and data/raw/disease/")
    print("3. OR create a labels.csv file with image paths and labels")
    print("4. Run: python src/train.py")
    print("=" * 60)

