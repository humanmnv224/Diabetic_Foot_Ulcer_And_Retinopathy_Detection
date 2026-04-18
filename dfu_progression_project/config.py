import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, "dataset")
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")

# Required class folders for severity classification (normal handled by lesion gate)
CLASS_NAMES = ["mild", "moderate", "severe"]
NUM_CLASSES = 3

# Model and training configuration
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 12
WARMUP_EPOCHS = 12
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2
SEED = 42
ENABLE_FINE_TUNING = False

# Prediction thresholds for severity model confidence
MIN_DFU_CLASS_CONFIDENCE = 0.45
MIN_CLASS_MARGIN = 0.05
MAX_BBOX_AREA_RATIO = 0.35

# Hybrid lesion-detector thresholds (rule-based gate)
MIN_LESION_AREA_RATIO = 0.0015
MAX_LESION_AREA_RATIO = 0.15
MIN_LESION_SCORE = 0.18

# Artifact paths
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
BEST_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_model.keras")
FINAL_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "final_model.keras")
HISTORY_PLOT_PATH = os.path.join(ARTIFACTS_DIR, "training_curves.png")
CONFUSION_MATRIX_PLOT_PATH = os.path.join(ARTIFACTS_DIR, "confusion_matrix.png")

# Grad-CAM settings
LAST_CONV_LAYER_NAME = "Conv_1"
HEATMAP_THRESHOLD = 0.6
