"""
Model building module using transfer learning
"""
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, MobileNetV2, EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

def build_model(base_model_name='ResNet50', input_shape=(224, 224, 3), 
                num_classes=1, dropout_rate=0.5, learning_rate=0.0001):
    """
    Build a CNN model using transfer learning
    
    Args:
        base_model_name: Name of the base model ('ResNet50', 'MobileNetV2', 'EfficientNetB0')
        input_shape: Shape of input images
        num_classes: Number of output classes (1 for binary classification)
        dropout_rate: Dropout rate for regularization
        learning_rate: Learning rate for optimizer
    
    Returns:
        Compiled Keras model
    """
    # Load base model (pretrained on ImageNet)
    if base_model_name == 'ResNet50':
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    elif base_model_name == 'MobileNetV2':
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    elif base_model_name == 'EfficientNetB0':
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    else:
        raise ValueError(f"Unsupported base model: {base_model_name}")
    
    # Freeze base model layers (we'll fine-tune later)
    base_model.trainable = False
    
    # Add custom classification layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    
    # Output layer
    if num_classes == 1:
        predictions = Dense(1, activation='sigmoid', name='predictions')(x)
    else:
        predictions = Dense(num_classes, activation='softmax', name='predictions')(x)
    
    # Create the full model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile model
    if num_classes == 1:
        loss = 'binary_crossentropy'
        metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    else:
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    return model

def unfreeze_base_model(model, layers_to_unfreeze=None):
    """
    Unfreeze base model layers for fine-tuning
    
    Args:
        model: Keras model
        layers_to_unfreeze: Number of layers to unfreeze from the end, or None to unfreeze all
    """
    base_model = None
    for layer in model.layers:
        if hasattr(layer, 'layers'):  # This is likely the base model
            base_model = layer
            break
    
    if base_model is None:
        print("Base model not found!")
        return
    
    if layers_to_unfreeze is None:
        # Unfreeze all layers
        base_model.trainable = True
    else:
        # Unfreeze only the last N layers
        base_model.trainable = True
        for layer in base_model.layers[:-layers_to_unfreeze]:
            layer.trainable = False
    
    # Recompile model with lower learning rate for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=0.00001),
        loss=model.loss,
        metrics=model.metrics
    )
    
    print(f"Unfroze layers for fine-tuning")

def print_model_summary(model):
    """
    Print model architecture summary
    """
    model.summary()
    
    # Count trainable parameters
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    total_params = sum([tf.size(w).numpy() for w in model.weights])
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")

if __name__ == "__main__":
    # Example: Build and inspect model
    print("Building model...")
    model = build_model(base_model_name='ResNet50')
    print_model_summary(model)

