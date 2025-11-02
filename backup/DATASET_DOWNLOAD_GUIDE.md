# Dataset Download Guide

## Quick Download Instructions

### Option 1: APTOS 2019 Dataset (Easiest - Kaggle)

**Steps:**

1. **Install Kaggle API**:
   ```bash
   pip install kaggle
   ```

2. **Get Kaggle Credentials**:
   - Go to: https://www.kaggle.com/settings
   - Click "Create New API Token"
   - Download `kaggle.json`
   - Place in: `C:\Users\<YourUsername>\.kaggle\kaggle.json`

3. **Download Dataset**:
   ```bash
   kaggle competitions download -c aptos2019-blindness-detection
   ```

4. **Extract and Organize**:
   ```bash
   # Extract the zip file
   # Then run:
   python utils/download_real_dataset.py --organize <extracted_folder>
   ```

### Option 2: Manual Download (No API needed)

1. **Visit Kaggle**:
   - Go to: https://www.kaggle.com/c/aptos2019-blindness-detection/data
   - Sign up/login (free)

2. **Download**:
   - Click "Download All" or download:
     - `train_images.zip` (contains ~3,662 images)
     - `train.csv` (contains labels)

3. **Extract**:
   - Extract `train_images.zip` to `data/raw/temp/`
   - Place `train.csv` in `data/raw/temp/`

4. **Organize**:
   ```bash
   python utils/download_real_dataset.py --organize data/raw/temp/
   ```

This will automatically:
- Read `train.csv` for labels
- Separate images into `normal/` (label=0) and `disease/` (label=1-4)
- Organize in the correct structure

### Option 3: Use Script Helper

Run the automated script:
```bash
python utils/download_real_dataset.py
```

Follow the on-screen instructions.

---

## Dataset Requirements

### Minimum for Testing:
- 200 images per class (400 total)
- Expected accuracy: 70-80%

### Recommended for Good Accuracy:
- 500 images per class (1,000 total)
- Expected accuracy: 80-90%

### Production Quality:
- 1,000+ images per class (2,000+ total)
- Expected accuracy: 85-95%

### Best (RFMiD):
- 3,200 images total
- Expected accuracy: 90-95%

---

## After Downloading

1. **Verify Dataset Structure**:
   ```
   data/raw/
     normal/
       image1.jpg
       image2.jpg
       ...
     disease/
       image1.jpg
       image2.jpg
       ...
   ```

2. **Check Image Count**:
   ```bash
   # Windows PowerShell
   (Get-ChildItem data\raw\normal -File).Count
   (Get-ChildItem data\raw\disease -File).Count
   ```

3. **Train Model**:
   ```bash
   python src/train.py
   ```

---

## Expected Results

With APTOS 2019 dataset (~3,662 images):
- **Training Accuracy**: 85-92%
- **Validation Accuracy**: 83-90%
- **Test Accuracy**: 82-89%
- **AUC-ROC**: 0.88-0.93

---

**Status**: Ready to download and organize dataset!

