"""
Create an enhanced dataset by augmenting existing images
This creates more training data from existing images
"""
import os
import cv2
import numpy as np
from pathlib import Path
from src.config import RAW_DATA_DIR

def augment_image(image, num_variations=10):
    """
    Create multiple augmented versions of an image
    """
    augmented = []
    
    for i in range(num_variations):
        aug = image.copy()
        
        # Random rotation
        angle = np.random.uniform(-30, 30)
        h, w = aug.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # Random flip
        if np.random.random() > 0.5:
            aug = cv2.flip(aug, 1)  # Horizontal flip
        
        if np.random.random() > 0.5:
            aug = cv2.flip(aug, 0)  # Vertical flip
        
        # Random brightness/contrast
        alpha = np.random.uniform(0.7, 1.3)  # Contrast
        beta = np.random.uniform(-20, 20)    # Brightness
        aug = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)
        
        # Random shift
        tx = np.random.uniform(-20, 20)
        ty = np.random.uniform(-20, 20)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # Random zoom
        zoom = np.random.uniform(0.9, 1.1)
        h_new, w_new = int(h * zoom), int(w * zoom)
        aug = cv2.resize(aug, (w_new, h_new))
        if zoom > 1:
            # Crop center
            start_y = (h_new - h) // 2
            start_x = (w_new - w) // 2
            aug = aug[start_y:start_y+h, start_x:start_x+w]
        else:
            # Pad
            aug = cv2.resize(aug, (w, h))
        
        augmented.append(aug)
    
    return augmented

def create_enhanced_dataset(source_normal_dir, source_disease_dir, 
                           target_normal_dir, target_disease_dir,
                           augment_factor=20):
    """
    Create enhanced dataset by augmenting existing images
    """
    print("=" * 70)
    print("Creating Enhanced Dataset via Augmentation")
    print("=" * 70)
    
    os.makedirs(target_normal_dir, exist_ok=True)
    os.makedirs(target_disease_dir, exist_ok=True)
    
    # Process normal images
    normal_images = list(Path(source_normal_dir).glob('*.jpg'))
    normal_images.extend(list(Path(source_normal_dir).glob('*.png')))
    
    print(f"\n[1/2] Processing {len(normal_images)} normal images...")
    normal_count = 0
    for img_path in normal_images:
        img = cv2.imread(str(img_path))
        if img is not None:
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Create augmented versions
            augmented = augment_image(img, num_variations=augment_factor)
            
            # Save augmented images
            base_name = img_path.stem
            for i, aug_img in enumerate(augmented):
                aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                save_path = os.path.join(target_normal_dir, f"{base_name}_aug_{i+1}.jpg")
                cv2.imwrite(save_path, aug_img_bgr)
                normal_count += 1
            
            if normal_count % 50 == 0:
                print(f"  Created {normal_count} normal images...")
    
    # Process disease images
    disease_images = list(Path(source_disease_dir).glob('*.jpg'))
    disease_images.extend(list(Path(source_disease_dir).glob('*.png')))
    
    print(f"\n[2/2] Processing {len(disease_images)} disease images...")
    disease_count = 0
    for img_path in disease_images:
        img = cv2.imread(str(img_path))
        if img is not None:
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Create augmented versions
            augmented = augment_image(img, num_variations=augment_factor)
            
            # Save augmented images
            base_name = img_path.stem
            for i, aug_img in enumerate(augmented):
                aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                save_path = os.path.join(target_disease_dir, f"{base_name}_aug_{i+1}.jpg")
                cv2.imwrite(save_path, aug_img_bgr)
                disease_count += 1
            
            if disease_count % 50 == 0:
                print(f"  Created {disease_count} disease images...")
    
    print("\n" + "=" * 70)
    print("Enhanced Dataset Created!")
    print("=" * 70)
    print(f"Normal images: {normal_count}")
    print(f"Disease images: {disease_count}")
    print(f"Total images: {normal_count + disease_count}")
    print(f"\nAugmentation factor: {augment_factor}x")
    print("\nYou can now train with this enhanced dataset:")
    print("  python src/train.py")

if __name__ == "__main__":
    source_normal = os.path.join(RAW_DATA_DIR, 'normal')
    source_disease = os.path.join(RAW_DATA_DIR, 'disease')
    
    target_normal = source_normal  # Overwrite with augmented
    target_disease = source_disease
    
    # Check if source images exist
    if not os.path.exists(source_normal) or not os.path.exists(source_disease):
        print("[ERROR] Source directories not found!")
        print("Please ensure images are in data/raw/normal/ and data/raw/disease/")
    else:
        # Create enhanced dataset with 20x augmentation
        create_enhanced_dataset(
            source_normal, source_disease,
            target_normal, target_disease,
            augment_factor=20  # 20 augmented versions per image
        )

