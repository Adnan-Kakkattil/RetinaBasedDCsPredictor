"""
Training script for the retina-based heart disease prediction model
"""
import os
import sys
import numpy as np
import tensorflow as tf
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import *
from src.model_builder import build_model, print_model_summary
from src.data_preprocessing import load_dataset_from_directory, split_data, save_preprocessed_data

def create_callbacks():
    """
    Create training callbacks
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    callbacks = [
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=REDUCE_LR_PATIENCE,
            min_lr=1e-7,
            verbose=1
        ),
        # Model checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODELS_DIR, MODEL_NAME),
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        # TensorBoard
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(LOGS_DIR, datetime.now().strftime("%Y%m%d-%H%M%S")),
            histogram_freq=1
        )
    ]
    
    return callbacks

def plot_training_history(history, save_path=None):
    """
    Plot training history
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision
    if 'precision' in history.history:
        axes[1, 0].plot(history.history['precision'], label='Train Precision')
        axes[1, 0].plot(history.history['val_precision'], label='Val Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Recall
    if 'recall' in history.history:
        axes[1, 1].plot(history.history['recall'], label='Train Recall')
        axes[1, 1].plot(history.history['val_recall'], label='Val Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    else:
        plt.show()

def main():
    """
    Main training function
    """
    # Get device info from module level (set in __main__)
    device_info = getattr(__import__(__name__), 'device_info', {'device_name': 'CPU'})
    
    print("=" * 50)
    print("Retina-Based Heart Disease Prediction - Training")
    print("=" * 50)
    print(f"Device: {device_info.get('device_name', 'CPU')}")
    print("=" * 50)
    
    # Load and preprocess data
    print("\n[1/5] Loading and preprocessing data...")
    images, labels, paths = load_dataset_from_directory(RAW_DATA_DIR)
    
    if len(images) == 0:
        print("ERROR: No images found!")
        print("Please ensure images are in the following structure:")
        print("  data/raw/normal/*.jpg")
        print("  data/raw/disease/*.jpg")
        print("\nOr provide a labels CSV file with columns: image_path, label")
        return
    
    print(f"Loaded {len(images)} images")
    print(f"Class distribution: {np.bincount(labels.astype(int))}")
    
    # Split data
    print("\n[2/5] Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        images, labels, 
        train_split=TRAIN_SPLIT, 
        val_split=VAL_SPLIT, 
        test_split=TEST_SPLIT
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Save preprocessed data
    save_preprocessed_data(X_train, X_val, X_test, y_train, y_val, y_test, PROCESSED_DATA_DIR)
    
    # Build model
    print("\n[3/5] Building model...")
    model = build_model(
        base_model_name=BASE_MODEL,
        input_shape=(*IMAGE_SIZE, IMAGE_CHANNELS),
        num_classes=1,
        dropout_rate=DROPOUT_RATE,
        learning_rate=LEARNING_RATE
    )
    print_model_summary(model)
    
    # Create callbacks
    callbacks = create_callbacks()
    
    # Train model
    print("\n[4/5] Training model...")
    print(f"Training for {EPOCHS} epochs with batch size {BATCH_SIZE}")
    
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Save training history
    history_path = os.path.join(MODELS_DIR, HISTORY_NAME)
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"\nTraining history saved to {history_path}")
    
    # Plot training history
    print("\n[5/5] Generating training plots...")
    plot_path = os.path.join(MODELS_DIR, 'training_history.png')
    plot_training_history(history, save_path=plot_path)
    
    # Final evaluation
    print("\n" + "=" * 50)
    print("Training completed!")
    print("=" * 50)
    print(f"Best model saved to: {os.path.join(MODELS_DIR, MODEL_NAME)}")
    print(f"\nFinal validation accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"Final validation loss: {min(history.history['val_loss']):.4f}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_results[1]:.4f}")
    print(f"Test loss: {test_results[0]:.4f}")

if __name__ == "__main__":
    # Setup GPU/CPU configuration
    from src.gpu_utils import setup_gpu, get_device_strategy
    
    # Setup GPU first
    device_info = setup_gpu()
    
    # Use distribution strategy if multiple GPUs available
    strategy = get_device_strategy()
    
    # Make device_info available globally for main()
    import src.train as train_module
    train_module.device_info = device_info
    
    if strategy:
        with strategy.scope():
            main()
    else:
        main()

