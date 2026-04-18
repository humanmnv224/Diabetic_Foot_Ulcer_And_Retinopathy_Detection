import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from config import BEST_MODEL_PATH, LEARNING_RATE


def build_model(input_shape=(224, 224, 3), num_classes=3) -> tf.keras.Model:
    """Create a transfer-learning model with a fast warmup stage."""
    base_model = EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    return model


def enable_fine_tuning(model: tf.keras.Model, unfreeze_top_layers: int = 30) -> tf.keras.Model:
    """Unfreeze top backbone layers for short fine-tuning and recompile at low LR."""
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and layer.name.startswith("efficientnet"):
            base_model = layer
            break
    if base_model is None:
        raise ValueError("Could not find EfficientNet backbone in model.")

    base_model.trainable = True

    for layer in base_model.layers[:-unfreeze_top_layers]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE * 0.1),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    return model


def build_callbacks() -> list:
    """Build callbacks for stable training with early stopping."""
    return [
        EarlyStopping(
            monitor="val_accuracy",
            patience=4,
            restore_best_weights=True,
            min_delta=0.001,
            mode="max",
        ),
        ModelCheckpoint(
            filepath=BEST_MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
            mode="max",
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=1,
            min_lr=1e-7,
            verbose=1,
        ),
    ]
