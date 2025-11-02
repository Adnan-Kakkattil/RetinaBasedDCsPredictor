"""
Script to download real retinal fundus image datasets
Supports multiple sources including Kaggle, direct downloads
"""
import os
import sys
import requests
import zipfile
import shutil
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import RAW_DATA_DIR

def download_kaggle_dataset(dataset_name, kaggle_username=None, kaggle_key=None):
    """
    Download dataset from Kaggle
    
    Args:
        dataset_name: Kaggle dataset identifier (e.g., 'aptos2019-blindness-detection')
        kaggle_username: Kaggle username (optional, can use environment variable)
        kaggle_key: Kaggle API key (optional, can use environment variable)
    """
    try:
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        print(f"Downloading {dataset_name} from Kaggle...")
        
        api = KaggleApi()
        api.authenticate()
        
        # Download dataset
        api.dataset_download_files(dataset_name, path='temp_downloads', unzip=True)
        
        print(f"[OK] Dataset downloaded successfully!")
        return True
    except ImportError:
        print("[ERROR] Kaggle package not installed")
        print("Install with: pip install kaggle")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to download from Kaggle: {str(e)}")
        return False

def download_file_direct(url, destination):
    """Download file directly from URL"""
    try:
        print(f"Downloading from {url}...")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}%", end='', flush=True)
        
        print("\n[OK] Download complete!")
        return True
    except Exception as e:
        print(f"\n[ERROR] Download failed: {str(e)}")
        return False

def extract_archive(archive_path, extract_to):
    """Extract zip/tar archive"""
    try:
        print(f"Extracting {archive_path}...")
        
        if archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.endswith(('.tar', '.tar.gz', '.tgz')):
            import tarfile
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            print(f"[ERROR] Unsupported archive format: {archive_path}")
            return False
        
        print("[OK] Extraction complete!")
        return True
    except Exception as e:
        print(f"[ERROR] Extraction failed: {str(e)}")
        return False

def organize_aptos_dataset(download_dir, target_dir):
    """
    Organize APTOS 2019 dataset into normal/disease structure
    APTOS dataset has CSV with labels: 0=No DR, 1-4=DR
    """
    import pandas as pd
    import shutil
    
    print("Organizing APTOS dataset...")
    
    # Find CSV file
    csv_files = list(Path(download_dir).glob('*.csv'))
    if not csv_files:
        print("[ERROR] No CSV file found")
        return False
    
    csv_path = csv_files[0]
    df = pd.read_csv(csv_path)
    
    # Find images directory
    img_dirs = [d for d in Path(download_dir).iterdir() if d.is_dir()]
    img_dir = None
    for d in img_dirs:
        if any(d.glob('*.png')) or any(d.glob('*.jpg')):
            img_dir = d
            break
    
    if img_dir is None:
        img_dir = download_dir
    
    # Create target directories
    normal_dir = os.path.join(target_dir, 'normal')
    disease_dir = os.path.join(target_dir, 'disease')
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(disease_dir, exist_ok=True)
    
    # Organize images
    count_normal = 0
    count_disease = 0
    
    for idx, row in df.iterrows():
        img_name = row.iloc[0]  # First column is usually image name
        label = row.iloc[1] if len(row) > 1 else 0  # Second column is label
        
        # Find image file
        img_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            candidate = img_dir / f"{img_name}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        
        if img_path and img_path.exists():
            if label == 0:  # Normal
                dest = os.path.join(normal_dir, img_path.name)
                shutil.copy2(img_path, dest)
                count_normal += 1
            else:  # Disease (1-4)
                dest = os.path.join(disease_dir, img_path.name)
                shutil.copy2(img_path, dest)
                count_disease += 1
    
    print(f"[OK] Organized dataset:")
    print(f"  Normal images: {count_normal}")
    print(f"  Disease images: {count_disease}")
    
    return count_normal > 0 and count_disease > 0

def download_sample_from_internet():
    """
    Try to download a sample dataset or provide manual instructions
    """
    print("=" * 70)
    print("Dataset Download Options")
    print("=" * 70)
    
    print("\nOPTION 1: Kaggle Datasets (Recommended)")
    print("-" * 70)
    print("1. Install Kaggle API:")
    print("   pip install kaggle")
    print("\n2. Set up Kaggle credentials:")
    print("   - Go to: https://www.kaggle.com/settings")
    print("   - Create API token")
    print("   - Place kaggle.json in ~/.kaggle/")
    print("\n3. Download APTOS 2019 dataset:")
    print("   kaggle competitions download -c aptos2019-blindness-detection")
    print("\n4. Or use this script with:")
    print("   python utils/download_real_dataset.py --kaggle aptos2019-blindness-detection")
    
    print("\nOPTION 2: Manual Download")
    print("-" * 70)
    print("1. Visit: https://www.kaggle.com/c/aptos2019-blindness-detection/data")
    print("2. Download train_images.zip")
    print("3. Download train.csv")
    print("4. Extract to data/raw/temp/")
    print("5. Run: python utils/download_real_dataset.py --organize data/raw/temp/")
    
    print("\nOPTION 3: Use Existing Public Dataset")
    print("-" * 70)
    print("If you have a dataset already:")
    print("1. Place images in: data/raw/temp/")
    print("2. Run: python utils/download_real_dataset.py --organize data/raw/temp/")
    
    print("\n" + "=" * 70)

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download and organize retinal fundus datasets')
    parser.add_argument('--kaggle', type=str, help='Kaggle dataset name to download')
    parser.add_argument('--organize', type=str, help='Organize existing dataset from directory')
    parser.add_argument('--url', type=str, help='Direct URL to download dataset')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Retinal Fundus Dataset Downloader")
    print("=" * 70)
    
    if args.kaggle:
        # Download from Kaggle
        download_kaggle_dataset(args.kaggle)
    elif args.organize:
        # Organize existing dataset
        if 'aptos' in args.organize.lower():
            organize_aptos_dataset(args.organize, RAW_DATA_DIR)
        else:
            print(f"[INFO] Organizing dataset from {args.organize}")
            print("Please ensure images are in subdirectories or CSV format")
    elif args.url:
        # Download from URL
        temp_file = os.path.join(RAW_DATA_DIR, 'temp_download.zip')
        if download_file_direct(args.url, temp_file):
            extract_archive(temp_file, RAW_DATA_DIR)
            # Try to organize
            organize_aptos_dataset(RAW_DATA_DIR, RAW_DATA_DIR)
    else:
        # Show instructions
        download_sample_from_internet()
        
        # Check if we can create a better sample by augmenting existing data
        print("\n" + "=" * 70)
        print("Creating Enhanced Sample Dataset")
        print("=" * 70)
        
        # Check existing synthetic data
        normal_dir = os.path.join(RAW_DATA_DIR, 'normal')
        disease_dir = os.path.join(RAW_DATA_DIR, 'disease')
        
        if os.path.exists(normal_dir) and os.path.exists(disease_dir):
            normal_count = len(list(Path(normal_dir).glob('*.jpg')))
            disease_count = len(list(Path(disease_dir).glob('*.jpg')))
            
            if normal_count > 0 and disease_count > 0:
                print(f"\n[INFO] Found existing dataset: {normal_count} normal, {disease_count} disease")
                print("[INFO] Dataset structure is ready!")
                print("\nFor better accuracy, download real dataset:")
                print("  - APTOS 2019: https://www.kaggle.com/c/aptos2019-blindness-detection")
                print("  - RFMiD: Contact dataset authors or check MDPI")
                print("\nCurrent dataset can be used for testing, but real data is needed for production accuracy.")

if __name__ == "__main__":
    main()

