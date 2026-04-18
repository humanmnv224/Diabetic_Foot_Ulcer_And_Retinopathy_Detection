import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from config import (
    ARTIFACTS_DIR,
    ENABLE_FINE_TUNING,
    EPOCHS,
    FINAL_MODEL_PATH,
    HISTORY_PLOT_PATH,
    NUM_CLASSES,
    WARMUP_EPOCHS,
)
from data_utils import build_generators
from model_utils import build_callbacks, build_model, enable_fine_tuning


def plot_training_history(history, save_path: str = HISTORY_PLOT_PATH) -> None:
    """Plot and save training and validation accuracy/loss curves."""
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Train Accuracy")
    plt.plot(epochs_range, val_acc, label="Val Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Train Loss")
    plt.plot(epochs_range, val_loss, label="Val Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def train() -> tf.keras.Model:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    train_gen, val_gen, _ = build_generators()
    model = build_model(input_shape=(160, 160, 3), num_classes=NUM_CLASSES)

    callbacks = build_callbacks()

    # Reweight classes to handle imbalance (especially the mild class).
    class_ids = np.unique(train_gen.classes)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=class_ids,
        y=train_gen.classes,
    )
    # Use softened class weights to avoid unstable training dynamics.
    softened = np.sqrt(weights)
    softened = np.clip(softened, 0.8, 2.0)
    class_weight = {int(k): float(v) for k, v in zip(class_ids, softened)}
    print(f"Using class weights: {class_weight}")

    history_warmup = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=WARMUP_EPOCHS,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    merged_history = copy.deepcopy(history_warmup.history)

    if ENABLE_FINE_TUNING and EPOCHS > WARMUP_EPOCHS:
        model = enable_fine_tuning(model, unfreeze_top_layers=15)
        history_ft = model.fit(
            train_gen,
            validation_data=val_gen,
            initial_epoch=WARMUP_EPOCHS,
            epochs=EPOCHS,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1,
        )

        for k, v in history_ft.history.items():
            merged_history.setdefault(k, []).extend(v)

    model.save(FINAL_MODEL_PATH)
    class HistoryWrapper:
        def __init__(self, history_dict):
            self.history = history_dict

    plot_training_history(HistoryWrapper(merged_history))

    print(f"Saved final model to: {FINAL_MODEL_PATH}")
    return model


if __name__ == "__main__":
    train()
