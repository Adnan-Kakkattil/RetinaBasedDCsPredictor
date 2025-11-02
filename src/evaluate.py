"""
Evaluation script for model performance assessment
"""
import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import *

def load_test_data():
    """
    Load test data from processed directory
    """
    X_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'))
    return X_test, y_test

def evaluate_model(model_path=None):
    """
    Evaluate trained model on test set
    """
    # Load model
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, MODEL_NAME)
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Please train the model first using: python src/train.py")
        return
    
    print(f"Loading model from {model_path}...")
    
    # Handle custom loss function (focal loss)
    try:
        from src.model_builder import focal_loss
        custom_objects = {'focal_loss_fn': focal_loss(gamma=2.0, alpha=0.25)}
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    except:
        try:
            # Try loading without custom objects
            model = tf.keras.models.load_model(model_path)
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Trying to rebuild model structure...")
            # Rebuild model
            from src.model_builder import build_model
            from src.config import BASE_MODEL, IMAGE_SIZE, IMAGE_CHANNELS, USE_FOCAL_LOSS
            model = build_model(
                base_model_name=BASE_MODEL,
                input_shape=(*IMAGE_SIZE, IMAGE_CHANNELS),
                num_classes=1,
                use_focal_loss=USE_FOCAL_LOSS
            )
            model.load_weights(model_path.replace('.h5', '_weights.h5'))
    
    # Load test data
    print("Loading test data...")
    X_test, y_test = load_test_data()
    
    print(f"Test samples: {len(X_test)}")
    print(f"Class distribution: {np.bincount(y_test.astype(int))}")
    
    # Predictions
    print("\nMaking predictions...")
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Calculate metrics
    accuracy = np.mean(y_pred == y_test)
    precision = tf.keras.metrics.Precision()(y_test, y_pred_proba).numpy()
    recall = tf.keras.metrics.Recall()(y_test, y_pred_proba).numpy()
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print("\n" + "=" * 50)
    print("MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")
    
    # Classification report
    print("\n" + "-" * 50)
    print("Classification Report:")
    print("-" * 50)
    print(classification_report(y_test, y_pred, target_names=['No Disease', 'Heart Disease']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Disease', 'Heart Disease'],
                yticklabels=['No Disease', 'Heart Disease'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    cm_path = os.path.join(MODELS_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to {cm_path}")
    plt.close()
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    
    roc_path = os.path.join(MODELS_DIR, 'roc_curve.png')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    print(f"ROC curve saved to {roc_path}")
    plt.close()
    
    # Feature importance visualization (using Grad-CAM if needed)
    print("\n" + "=" * 50)
    print("Evaluation completed!")
    print("=" * 50)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'auc': auc,
        'confusion_matrix': cm
    }

if __name__ == "__main__":
    evaluate_model()

