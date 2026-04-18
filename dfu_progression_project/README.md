# DFU Progression Detection (TensorFlow/Keras)

Complete deep learning project for diabetic foot ulcer progression detection from images using MobileNetV2 transfer learning, evaluation metrics, Grad-CAM explainability, bounding-box localization, and optional Streamlit UI.

## 1. Project Structure

```text
dfu_progression_project/
  artifacts/
  dataset/
    train/
      mild/
      moderate/
      severe/
    test/
      mild/
      moderate/
      severe/
  config.py
  data_utils.py
  model_utils.py
  gradcam_utils.py
  bootstrap_dataset_from_workspace.py
  review_dataset.py
  train.py
  evaluate.py
  predict.py
  streamlit_app.py
  requirements.txt
```

## 2. Install

```powershell
cd c:\Users\valva\OneDrive\Desktop\dfu\dfu_progression_project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 3. Dataset Setup

This project expects exactly 3 classes:
- `mild`
- `moderate`
- `severe`

### Fastest path (auto-bootstrap from your current DFU workspace)

This project includes a one-command bootstrap that builds progression classes from ulcer images only:

- Sources used:
  - mild: `DFU/Transfer-Learning images/internetSet` + `DFU/Transfer-Learning images/samples`
  - moderate: `DFU/Transfer-Learning images/Wound Images`
  - severe: `DFU/Patches/Abnormal(Ulcer)` + `DFU/Transfer-Learning images/Wound Images2`
- Labeling strategy:
  - deterministic source-folder mapping
  - optional class cap with `--max_per_class` for balancing

`Normal(Healthy skin)` is not used in progression classes.

Run:

```powershell
python bootstrap_dataset_from_workspace.py
```

Balanced severe class example:

```powershell
python bootstrap_dataset_from_workspace.py --max_per_class 300
```

It will create:

```text
dataset/train/{mild,moderate,severe}
dataset/test/{mild,moderate,severe}
```

Note: this bootstrap is intended for fast project execution. For clinical-grade progression labels, expert relabeling is required.

### Manual Cleanup (recommended)

Review and correct labels quickly with keyboard controls:

```powershell
python review_dataset.py --split train
python review_dataset.py --split test
```

Controls:
- `1` -> move to mild
- `2` -> move to moderate
- `3` -> move to severe
- `s` -> skip
- `q` -> quit

### Option A: You already have train/test folders

Place images in:

```text
dataset/
  train/mild
  train/moderate
  train/severe
  test/mild
  test/moderate
  test/severe
```

## 4. Train (Model 1 - Classification)

```powershell
python train.py
```

Training uses:
- `ImageDataGenerator`
- Rescale: `1./255`
- Augmentation: rotation, zoom, horizontal flip
- MobileNetV2 (ImageNet pretrained)
- Frozen base layers
- Head:
  - `GlobalAveragePooling2D`
  - `Dense(128, relu)`
  - `Dropout(0.5)`
  - `Dense(3, softmax)`
- Optimizer: Adam
- Loss: categorical_crossentropy
- Epochs: 15
- Callbacks:
  - EarlyStopping
  - ModelCheckpoint
  - ReduceLROnPlateau

Saved models:
- `artifacts/best_model.h5`
- `artifacts/final_model.h5`

## 5. Evaluate

```powershell
python evaluate.py
```

Outputs:
- Test accuracy and loss
- Classification report
- Confusion matrix plot

## 6. Predict + Grad-CAM + Bounding Box

```powershell
python predict.py --image "c:\path\to\test_image.jpg"
```

Outputs:
- Predicted class (`mild`, `moderate`, `severe`)
- Confidence score
- Progression mapping:
  - mild -> Improving
  - moderate -> Stable
  - severe -> Worsening
- Visuals:
  - Original image
  - Heatmap overlay
  - Bounding box image

## 7. Run Full Pipeline

```powershell
python bootstrap_dataset_from_workspace.py
python review_dataset.py --split train --suspects_first --only_suspects
python review_dataset.py --split test --suspects_first --only_suspects
python train.py
python evaluate.py
```

## 8. Optional Streamlit App

```powershell
streamlit run streamlit_app.py
```

Upload an image and see prediction, progression, Grad-CAM overlay, and bounding box.

## 9. Notes for Your Current DFU Workspace

Your currently visible folders do not include `mild/moderate/severe` class folders yet. To train a progression classifier, labels for these 3 classes are required.

If your current data is binary (`normal` vs `ulcer`) or unlabeled mixed images, you must first relabel/reorganize into progression classes.
