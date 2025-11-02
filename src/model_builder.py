"""
Model building module using transfer learning
"""
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, ResNet101, MobileNetV2, EfficientNetB0, EfficientNetB3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss for addressing class imbalance
    FL = -alpha * (1 - p)^gamma * log(p)
    """
    def focal_loss_fn(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate focal loss
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.ones_like(y_true) * alpha
        alpha_t = tf.where(tf.equal(y_true, 1), alpha_t, 1 - alpha_t)
        
        cross_entropy = -tf.math.log(p_t)
        weight = alpha_t * tf.math.pow((1 - p_t), gamma)
        loss = weight * cross_entropy
        
        return tf.reduce_mean(loss)
    
    return focal_loss_fn

def build_model(base_model_name='ResNet101', input_shape=(224, 224, 3), 
                num_classes=1, dropout_rate=0.5, learning_rate=0.0001, use_focal_loss=False):
    """
    Build a CNN model using transfer learning with improved architecture
    
    Args:
        base_model_name: Name of the base model ('ResNet50', 'ResNet101', 'MobileNetV2', 'EfficientNetB0', 'EfficientNetB3')
        input_shape: Shape of input images
        num_classes: Number of output classes (1 for binary classification)
        dropout_rate: Dropout rate for regularization
        learning_rate: Learning rate for optimizer
        use_focal_loss: Whether to use focal loss for class imbalance
    
    Returns:
        Compiled Keras model
    """
    # Load base model (pretrained on ImageNet)
    if base_model_name == 'ResNet50':
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape,
            pooling=None
        )
    elif base_model_name == 'ResNet101':
        base_model = ResNet101(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape,
            pooling=None
        )
    elif base_model_name == 'MobileNetV2':
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape,
            pooling=None
        )
    elif base_model_name == 'EfficientNetB0':
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape,
            pooling=None
        )
    elif base_model_name == 'EfficientNetB3':
        base_model = EfficientNetB3(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape,
            pooling=None
        )
    else:
        raise ValueError(f"Unsupported base model: {base_model_name}. Options: ResNet50, ResNet101, MobileNetV2, EfficientNetB0, EfficientNetB3")
    
    # Freeze base model layers (we'll fine-tune later)
    base_model.trainable = False
    
    # Add custom classification layers with improved architecture
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    
    # Larger hidden layers for better feature learning
    x = Dense(512, activation='relu', name='fc1')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    x = Dense(256, activation='relu', name='fc2')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    x = Dense(128, activation='relu', name='fc3')(x)
    x = Dropout(dropout_rate * 0.5)(x)
    
    # Output layer
    if num_classes == 1:
        predictions = Dense(1, activation='sigmoid', name='predictions')(x)
    else:
        predictions = Dense(num_classes, activation='softmax', name='predictions')(x)
    
    # Create the full model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile model with improved metrics
    if num_classes == 1:
        if use_focal_loss:
            loss = focal_loss(gamma=2.0, alpha=0.25)
        else:
            loss = BinaryCrossentropy(from_logits=False)
        metrics = [
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc', curve='ROC')
        ]
    else:
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
    
    optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
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

