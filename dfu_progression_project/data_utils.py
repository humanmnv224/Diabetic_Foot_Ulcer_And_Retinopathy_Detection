import os
import shutil
from pathlib import Path
from typing import Tuple

from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import (
    BATCH_SIZE,
    CLASS_NAMES,
    IMAGE_SIZE,
    SEED,
    TEST_DIR,
    TRAIN_DIR,
    VALIDATION_SPLIT,
)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_class_structure(train_dir: str = TRAIN_DIR, test_dir: str = TEST_DIR) -> None:
    """Validate that train/test folders contain the configured class subfolders."""
    for split_dir, split_name in [(train_dir, "train"), (test_dir, "test")]:
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"Missing {split_name} directory: {split_dir}. "
                "Create dataset/train and dataset/test before training."
            )

        existing = {d.name for d in Path(split_dir).iterdir() if d.is_dir()}
        missing = [c for c in CLASS_NAMES if c not in existing]
        if missing:
            raise ValueError(
                f"{split_name} is missing class folders: {missing}. "
                f"Expected exactly: {CLASS_NAMES}"
            )


def build_generators(
    train_dir: str = TRAIN_DIR,
    test_dir: str = TEST_DIR,
) -> Tuple:
    """Create train/validation/test generators with augmentation and EfficientNet preprocessing."""
    ensure_class_structure(train_dir, test_dir)

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        zoom_range=0.25,
        horizontal_flip=True,
        width_shift_range=0.15,
        height_shift_range=0.15,
        brightness_range=[0.9, 1.1],
        fill_mode='nearest',
        validation_split=VALIDATION_SPLIT,
    )

    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        subset="training",
        shuffle=True,
        seed=SEED,
    )

    val_generator = train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        subset="validation",
        shuffle=False,
        seed=SEED,
    )

    test_generator = test_datagen.flow_from_directory(
        directory=test_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False,
    )

    return train_generator, val_generator, test_generator


def prepare_train_test_split(
    source_root: str,
    output_root: str,
    test_size: float = 0.2,
    seed: int = SEED,
) -> None:
    """
    Build dataset/train and dataset/test from an already labeled source folder.

    Expected source structure:
        source_root/
            mild/
            moderate/
            severe/
    """
    source_root_path = Path(source_root)
    output_root_path = Path(output_root)
    train_path = output_root_path / "train"
    test_path = output_root_path / "test"

    for class_name in CLASS_NAMES:
        class_src = source_root_path / class_name
        if not class_src.exists() or not class_src.is_dir():
            raise FileNotFoundError(
                f"Missing class folder in source_root: {class_src}. "
                "Create mild/moderate/severe folders first."
            )

        files = [
            p for p in class_src.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS
        ]
        if len(files) < 2:
            raise ValueError(
                f"Class '{class_name}' needs at least 2 images to split. Found: {len(files)}"
            )

        train_files, test_files = train_test_split(files, test_size=test_size, random_state=seed)

        target_train_class = train_path / class_name
        target_test_class = test_path / class_name
        target_train_class.mkdir(parents=True, exist_ok=True)
        target_test_class.mkdir(parents=True, exist_ok=True)

        for src_file in train_files:
            shutil.copy2(src_file, target_train_class / src_file.name)

        for src_file in test_files:
            shutil.copy2(src_file, target_test_class / src_file.name)

    print(f"Prepared dataset at: {output_root_path}")
