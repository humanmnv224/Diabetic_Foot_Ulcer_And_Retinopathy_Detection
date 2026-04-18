import argparse
import os

import tensorflow as tf

from config import BEST_MODEL_PATH
from gradcam_utils import predict_with_explainability, show_prediction_visuals


def run_prediction(image_path: str, model_path: str = BEST_MODEL_PATH) -> None:
    """Run single-image prediction with Grad-CAM and progression label."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = tf.keras.models.load_model(model_path)
    result = predict_with_explainability(model, image_path)

    print(f"Predicted Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Progression: {result['progression']}")
    print(f"Bounding Box (x, y, w, h): {result['bbox']}")

    show_prediction_visuals(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="DFU progression prediction with Grad-CAM")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--model", default=BEST_MODEL_PATH, help="Path to trained .h5 model")
    args = parser.parse_args()

    run_prediction(args.image, args.model)


if __name__ == "__main__":
    main()
