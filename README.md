# DFU + Retinopathy Multimodal Screening Project

A unified medical-imaging project that combines:

1. **DFU severity grading** from foot images (Normal, Mild, Moderate, Severe)
2. **Diabetic retinopathy grading** from retinal fundus images (No DR to Proliferative DR)
3. **Combined diabetes risk screening** that uses both model outputs in a simple rule-based triage layer

This repository is built as an engineering and research demo for image-based triage workflows.

## Important Medical Note

This software is for **research/educational demonstration** and **not for diagnosis**. Any medical decision must be made by qualified clinicians.

---

## Table of Contents

- [Project Goal](#project-goal)
- [How the Full System Works](#how-the-full-system-works)
- [Model 1: DFU Model (Foot Ulcer)](#model-1-dfu-model-foot-ulcer)
- [Model 2: Retinopathy Model (Fundus)](#model-2-retinopathy-model-fundus)
- [Combined Diabetes Decision Logic](#combined-diabetes-decision-logic)
- [Repository Layout](#repository-layout)
- [Quick Start (Windows PowerShell)](#quick-start-windows-powershell)
- [Quick Start (Linuxmacos)](#quick-start-linuxmacos)
- [Train DFU Model From Scratch](#train-dfu-model-from-scratch)
- [Run the Streamlit App](#run-the-streamlit-app)
- [Expected Inputs and Outputs](#expected-inputs-and-outputs)
- [Troubleshooting](#troubleshooting)
- [Roadmap Ideas](#roadmap-ideas)

---

## Project Goal

Diabetes complications can appear in different organs. This project combines two visual markers:

- **Foot tissue condition** (DFU severity)
- **Retinal vascular damage** (DR severity)

Instead of relying on one modality, the app performs:

1. Single-model DFU assessment
2. Single-model retinopathy assessment
3. Combined screen that merges both outcomes into a practical triage message

---

## How the Full System Works

### End-to-end flow

1. User uploads image(s) in Streamlit.
2. App loads latest DFU checkpoint from `dfu_progression_project/artifacts/checkpoints`.
3. App loads latest DR checkpoint from `diabetic-retinopathy-detection-main/artifacts`.
4. Each model outputs class probabilities.
5. App converts predictions to human-readable severity labels.
6. Combined mode applies confidence thresholds and rule logic to produce a final diabetes-risk message.

### Why this architecture

- Keeps model training and inference reproducible.
- Decouples domain-specific models (foot vs retina).
- Enables future upgrade of either model independently.

---

## Model 1: DFU Model (Foot Ulcer)

### Task

4-class classification from foot images:

- `0 -> Normal`
- `1 -> Mild`
- `2 -> Moderate`
- `3 -> Severe`

### Training stack

- **Framework**: PyTorch Lightning
- **Backbone** (default): `efficientnet_b0`
- **Alternatives available**: EfficientNet B1, DenseNet121, DenseNet169, ResNet50
- **Loss**: `CrossEntropyLoss` (optionally class-weighted)
- **Optimizer**: `AdamW`
- **LR scheduler**: `ReduceLROnPlateau` (optional)
- **Metrics**:
  - Validation loss
  - Multiclass accuracy
  - Quadratic Cohen's kappa

### Data pipeline

- Input source folders are bootstrapped from `DFU_Dataset`.
- Output split folders:
  - `dataset/train/{normal,mild,moderate,severe}`
  - `dataset/test/{normal,mild,moderate,severe}`
- CSV files for Lightning datamodule:
  - `dataset/dfu_train.csv`
  - `dataset/dfu_val.csv`

### Augmentation and preprocessing

- Resize to configured image size (default 160)
- Random affine transforms
- Color jitter
- Horizontal/vertical flips
- Rotation
- ImageNet normalization

### Inference behavior in app

- DFU image resized to `160x160`
- Softmax class probabilities computed
- Confidence threshold for strong normal decision: `0.60`

---

## Model 2: Retinopathy Model (Fundus)

### Task

5-class diabetic retinopathy severity grading:

- `0 -> No DR`
- `1 -> Mild`
- `2 -> Moderate`
- `3 -> Severe`
- `4 -> Proliferative DR`

### Training model class

- `DRModel` from `diabetic-retinopathy-detection-main/src/model.py`
- Lightning module with multiclass loss/metrics
- Commonly used checkpoint in this project: `dr-model.ckpt`

### Inference behavior in app

- Fundus image resized to `224x224`
- Softmax class probabilities computed
- Confidence threshold for strong No-DR decision: `0.55`

### Why separate repo folder is used

The integrated Streamlit app imports the retinopathy model from:

- `diabetic-retinopathy-detection-main`

This is the active source for DR inference in the current integration.

---

## Combined Diabetes Decision Logic

In **Detect Diabetes** mode, both models are evaluated together.

Definitions:

- DFU is considered normal if class is `Normal` with confidence `>= 0.60`
- Retina is considered normal if class is `No DR` with confidence `>= 0.55`
- Abnormal flags require non-normal class with confidence above the corresponding threshold

Decision rules:

1. **Both normal** -> `No diabetes signal`
2. **Both abnormal** -> `Diabetes likely`
3. **Mixed or low confidence** -> `Maybe diabetes`

This rule is intentionally simple and interpretable for triage messaging.

---

## Repository Layout

```text
.
├─ DFU_Dataset/
│  └─ DFU/
│     ├─ Original Images/
│     ├─ Patches/
│     │  ├─ Abnormal(Ulcer)/
│     │  └─ Normal(Healthy skin)/
│     └─ Transfer-Learning images/
│        ├─ internetSet/
│        ├─ samples/
│        ├─ Wound Images/
│        └─ Wound Images2/
├─ dfu_progression_project/
│  ├─ conf/config.yaml
│  ├─ dataset/
│  ├─ src/
│  ├─ artifacts/
│  ├─ train_lightning.py
│  ├─ bootstrap_dataset_from_workspace.py
│  ├─ generate_dataset_csv.py
│  └─ streamlit_app.py
└─ diabetic retinopathy detection/
   ├─ src/
   └─ artifacts/dr-model.ckpt
```

---

## Quick Start (Windows PowerShell)

### Prerequisites
- Python 3.8+
- Git

### Setup and Run

1. Clone the repository:
   ```powershell
   git clone https://github.com/humanmnv224/Diabetic_Foot_Ulcer_And_Retinopathy_Detection.git
   cd Diabetic_Foot_Ulcer_And_Retinopathy_Detection
   ```

2. Create and activate virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install all dependencies for the Streamlit app:
   ```powershell
   pip install -r requirements.txt
   ```

   > If you want to train the DFU model locally using TensorFlow, also install:
   > `pip install -r dfu_progression_project/requirements.txt`

4. Run the Streamlit app:
   ```powershell
   python -m streamlit run dfu_progression_project/streamlit_app.py
   ```

The app will automatically open in your browser at `http://localhost:8501`. Upload foot or retinal images to get diabetes risk predictions using pre-trained models.

---

## Quick Start (Linux/macOS)

### Prerequisites
- Python 3.8+
- Git

### Setup and Run

1. Clone the repository:
   ```bash
   git clone https://github.com/humanmnv224/Diabetic_Foot_Ulcer_And_Retinopathy_Detection.git
   cd Diabetic_Foot_Ulcer_And_Retinopathy_Detection
   ```

2. Create and activate virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install all dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```bash
   python -m streamlit run dfu_progression_project/streamlit_app.py
   ```

The app will automatically open in your browser at `http://localhost:8501`. Upload foot or retinal images to get diabetes risk predictions using pre-trained models.

---

## Train DFU Model From Scratch

If you want to retrain with your own hyperparameters:

```powershell
Set-Location .\dfu_progression_project
python .\train_lightning.py model_name=efficientnet_b0 max_epochs=30 batch_size=16 learning_rate=1e-4
```

Useful Hydra overrides:

- `model_name` -> choose from `efficientnet_b0`, `efficientnet_b1`, `densenet121`, `densenet169`, `resnet50`
- `max_epochs`
- `batch_size`
- `learning_rate`
- `use_class_weighting`
- `use_weighted_sampler`

Checkpoints are saved under:

- `dfu_progression_project/artifacts/checkpoints/<run-id>/`

Metrics summary JSON is saved under:

- `dfu_progression_project/artifacts/metrics/`

---

## Run the Streamlit App

After installing dependencies and activating the virtual environment:

```powershell
python -m streamlit run dfu_progression_project/streamlit_app.py
```

The app will launch automatically in your default browser at `http://localhost:8501`.

### App Modes

The UI provides three prediction modes:

1. **Detect DFU** - Upload a foot image to predict diabetic foot ulcer severity (Normal, Mild, Moderate, Severe)
2. **Detect Retinopathy** - Upload a fundus (retinal) image to predict diabetic retinopathy severity (No DR → Proliferative DR)
3. **Detect Diabetes** - Upload both foot and fundus images for combined diabetes risk screening

---

## Expected Inputs and Outputs

### Input image guidance

- Supported formats: `jpg`, `jpeg`, `png`, `bmp`, `webp`
- DFU mode: close, clear foot ulcer/foot region image
- Retina mode: retinal fundus image

### App outputs

- Predicted class and confidence
- Per-class probabilities
- Combined triage recommendation (in combined mode)

---

## Troubleshooting

### 1) Missing dependencies

If you encounter `ModuleNotFoundError`, ensure all packages are installed:

```powershell
pip install -r requirements.txt
```

Common missing packages:
- `torchvision` - Image utilities for PyTorch
- `lightning` / `pytorch-lightning` - Model training framework
- `streamlit` - Web app framework

### 2) App cannot find model checkpoints

Ensure model checkpoint files exist:

- **DFU model**: `dfu_progression_project/artifacts/checkpoints/` (at least one `.ckpt` file)
- **Retinopathy model**: `diabetic-retinopathy-detection-main/artifacts/dr-model.ckpt`

If missing, re-download or retrain the models.

### 3) Streamlit port already in use

Run on a different port:

```powershell
python -m streamlit run dfu_progression_project/streamlit_app.py --server.port 8502
```

### 4) Python/Virtual Environment issues

On Windows, if PowerShell execution policy blocks scripts:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

On Linux/macOS, ensure Python 3.8+ is available:

```bash
python3 --version
```

### 5) GPU/CUDA issues

If you see CUDA-related errors but don't have an NVIDIA GPU, the models will still run on CPU (slower). To ensure CPU-only mode:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## Roadmap Ideas

- Export DFU and DR models to ONNX/TorchScript for deployment
- Add calibration (temperature scaling) for confidence reliability
- Add explainability overlays in app for both modalities
- Add REST API (FastAPI) for production integration
- Add CI to run smoke tests and lint checks on every push

---

## License and Data

- Respect licensing and privacy requirements for all datasets and clinical images.
- The included dataset readme files indicate Roboflow export metadata and license terms.
