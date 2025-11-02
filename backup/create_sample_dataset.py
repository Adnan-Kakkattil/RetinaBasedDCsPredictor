"""
Script to create a small sample dataset for demonstration
Creates synthetic retinal fundus-like images for testing
"""
import os
import cv2
import numpy as np
from pathlib import Path

# Set seed for reproducibility
np.random.seed(42)

def create_synthetic_retinal_image(is_disease=False, size=(224, 224)):
    """
    Create a synthetic retinal fundus-like image
    
    Args:
        is_disease: If True, create image with 'disease' characteristics
        size: Image dimensions
    """
    # Base background (dark, like fundus)
    image = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    image[:, :] = [20, 10, 5]  # Dark reddish background
    
    # Add some circular patterns (simulating retina)
    center_x, center_y = size[0] // 2, size[1] // 2
    
    # Main circular area (brighter center)
    cv2.circle(image, (center_x, center_y), 80, (100, 80, 60), -1)
    cv2.circle(image, (center_x, center_y), 60, (150, 120, 90), -1)
    
    # Add some vessels (lines)
    for i in range(8):
        angle = i * np.pi / 4
        x1 = int(center_x + 30 * np.cos(angle))
        y1 = int(center_y + 30 * np.sin(angle))
        x2 = int(center_x + 90 * np.cos(angle))
        y2 = int(center_y + 90 * np.sin(angle))
        color = (80, 60, 40)
        cv2.line(image, (x1, y1), (x2, y2), color, 2)
    
    if is_disease:
        # Add 'disease' markers (darker spots, irregularities)
        for _ in range(5):
            x = np.random.randint(50, size[0] - 50)
            y = np.random.randint(50, size[1] - 50)
            radius = np.random.randint(3, 8)
            cv2.circle(image, (x, y), radius, (10, 5, 2), -1)
        
        # Add more irregular patterns
        for _ in range(3):
            x = np.random.randint(40, size[0] - 40)
            y = np.random.randint(40, size[1] - 40)
            cv2.ellipse(image, (x, y), (15, 10), 0, 0, 360, (40, 30, 20), -1)
    
    # Add some noise for realism
    noise = np.random.randint(-10, 10, image.shape, dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Apply slight blur for realism
    image = cv2.GaussianBlur(image, (3, 3), 0)
    
    return image

def create_sample_dataset(num_normal=20, num_disease=20):
    """
    Create a sample dataset with synthetic images
    
    Args:
        num_normal: Number of normal images to create
        num_disease: Number of disease images to create
    """
    from src.config import RAW_DATA_DIR
    
    normal_dir = os.path.join(RAW_DATA_DIR, 'normal')
    disease_dir = os.path.join(RAW_DATA_DIR, 'disease')
    
    # Create directories
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(disease_dir, exist_ok=True)
    
    print("=" * 60)
    print("Creating Sample Dataset")
    print("=" * 60)
    print(f"Generating {num_normal} normal images...")
    
    # Create normal images
    for i in range(num_normal):
        img = create_synthetic_retinal_image(is_disease=False)
        filename = f"normal_{i+1:03d}.jpg"
        filepath = os.path.join(normal_dir, filename)
        cv2.imwrite(filepath, img)
        if (i + 1) % 5 == 0:
            print(f"  Created {i + 1}/{num_normal} normal images")
    
    print(f"\nGenerating {num_disease} disease images...")
    
    # Create disease images
    for i in range(num_disease):
        img = create_synthetic_retinal_image(is_disease=True)
        filename = f"disease_{i+1:03d}.jpg"
        filepath = os.path.join(disease_dir, filename)
        cv2.imwrite(filepath, img)
        if (i + 1) % 5 == 0:
            print(f"  Created {i + 1}/{num_disease} disease images")
    
    print("\n" + "=" * 60)
    print("Dataset Created Successfully!")
    print("=" * 60)
    print(f"Normal images: {num_normal} in {normal_dir}")
    print(f"Disease images: {num_disease} in {disease_dir}")
    print(f"Total images: {num_normal + num_disease}")
    print("\nYou can now run: python src/train.py")

if __name__ == "__main__":
    # Create a small but sufficient dataset for demonstration
    create_sample_dataset(num_normal=30, num_disease=30)

