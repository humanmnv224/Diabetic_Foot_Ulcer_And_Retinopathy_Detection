import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

DFU_CLASS_NAMES = ["Normal", "Mild", "Moderate", "Severe"]
RETINAL_CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

DFU_REPO_ROOT = Path(__file__).resolve().parent
RETINOPATHY_REPO_ROOT = Path(__file__).resolve().parent.parent / "diabetic-retinopathy-detection-main"

DEFAULT_DFU_WEIGHTS_PATH = (
    DFU_REPO_ROOT
    / "artifacts"
    / "checkpoints"
)
DEFAULT_RETINAL_WEIGHTS_PATH = (
    RETINOPATHY_REPO_ROOT
    / "artifacts"
    / "dr-model.ckpt"
)

DFU_NORMAL_THRESHOLD = 0.60
RETINAL_NO_DR_THRESHOLD = 0.55


def get_latest_dfu_checkpoint() -> Path:
    ckpt_files = sorted(
        DEFAULT_DFU_WEIGHTS_PATH.rglob("*.ckpt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if ckpt_files:
        return ckpt_files[0]
    return DEFAULT_DFU_WEIGHTS_PATH / "dfu-model.ckpt"


def get_latest_retinal_checkpoint() -> Path:
    ckpt_files = sorted(
        (RETINOPATHY_REPO_ROOT / "artifacts").rglob("*.ckpt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if ckpt_files:
        return ckpt_files[0]
    return DEFAULT_RETINAL_WEIGHTS_PATH


def _load_module_from_path(module_name: str, module_path: Path):
    package_name = module_name.rsplit(".", 1)[0] if "." in module_name else None

    if package_name:
        package = importlib.util.module_from_spec(
            importlib.machinery.ModuleSpec(package_name, loader=None, is_package=True)
        )
        package.__path__ = [str(module_path.parent)]
        sys.modules[package_name] = package

    if module_name in sys.modules:
        del sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {module_path}")

    module = importlib.util.module_from_spec(spec)
    if package_name:
        module.__package__ = package_name
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@st.cache_resource
def _get_dfu_model_class():
    module_path = DFU_REPO_ROOT / "src" / "model.py"
    module = _load_module_from_path("dfu_src.model", module_path)
    return module.DFUModel


@st.cache_resource
def load_dfu_model_from_path(checkpoint_path: str):
    DFUModel = _get_dfu_model_class()
    model = DFUModel.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()
    return model


def load_dfu_model_from_bytes(checkpoint_bytes: bytes):
    DFUModel = _get_dfu_model_class()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ckpt") as tmp:
        tmp.write(checkpoint_bytes)
        temp_ckpt = tmp.name

    try:
        model = DFUModel.load_from_checkpoint(temp_ckpt, map_location="cpu")
    finally:
        if os.path.exists(temp_ckpt):
            os.remove(temp_ckpt)

    model.eval()
    return model


def _get_retinopathy_model_class():
    module_path = RETINOPATHY_REPO_ROOT / "src" / "model.py"
    module = _load_module_from_path("retina_src.model", module_path)
    return module.DRModel


@st.cache_resource
def load_retinal_model_from_path(checkpoint_path: str):
    DRModel = _get_retinopathy_model_class()
    model = DRModel.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()
    return model


def load_retinal_model_from_bytes(checkpoint_bytes: bytes):
    DRModel = _get_retinopathy_model_class()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ckpt") as tmp:
        tmp.write(checkpoint_bytes)
        temp_ckpt = tmp.name

    try:
        model = DRModel.load_from_checkpoint(temp_ckpt, map_location="cpu")
    finally:
        if os.path.exists(temp_ckpt):
            os.remove(temp_ckpt)

    model.eval()
    return model


def predict_dfu(model, pil_image: Image.Image):
    preprocess = transforms.Compose(
        [
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    x = preprocess(pil_image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        outputs = model(x)
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]

        if outputs.ndim == 2:
            logits = outputs[0]
        else:
            logits = outputs

        probs = torch.softmax(logits, dim=0).cpu().numpy()

    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    return pred_idx, confidence, probs


def predict_retinal(model, pil_image: Image.Image):
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    x = preprocess(pil_image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        outputs = model(x)
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]

        if outputs.ndim == 2:
            logits = outputs[0]
        else:
            logits = outputs

        probs = torch.softmax(logits, dim=0).cpu().numpy()

    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    return pred_idx, confidence, probs


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background: radial-gradient(circle at top left, #1f2937 0%, #0f172a 45%, #050816 100%);
                color: #f8fafc;
            }
            .block-container {
                padding-top: 1.5rem;
                padding-bottom: 2rem;
            }
            .hero {
                padding: 1.4rem 1.5rem;
                border-radius: 22px;
                background: linear-gradient(135deg, rgba(14,165,233,0.20), rgba(34,197,94,0.16));
                border: 1px solid rgba(148,163,184,0.22);
                box-shadow: 0 18px 45px rgba(0,0,0,0.24);
                margin-bottom: 1rem;
            }
            .hero h1 {
                margin: 0;
                font-size: 2.2rem;
                line-height: 1.1;
            }
            .hero p {
                margin: 0.4rem 0 0 0;
                color: #cbd5e1;
                font-size: 1rem;
            }
            .metric-card {
                padding: 0.9rem 1rem;
                border-radius: 18px;
                background: rgba(15, 23, 42, 0.72);
                border: 1px solid rgba(148,163,184,0.18);
                box-shadow: 0 10px 30px rgba(0,0,0,0.18);
                min-height: 110px;
            }
            .metric-card h4 {
                margin: 0 0 0.35rem 0;
                font-size: 0.98rem;
            }
            .metric-card p {
                margin: 0;
                color: #cbd5e1;
                font-size: 0.88rem;
            }
            .section-title {
                margin-top: 0.6rem;
                font-size: 1.2rem;
                font-weight: 700;
            }
            .small-muted {
                color: #94a3b8;
                font-size: 0.88rem;
            }
            .result-box {
                padding: 1rem 1.1rem;
                border-radius: 18px;
                background: rgba(2,6,23,0.72);
                border: 1px solid rgba(148,163,184,0.18);
            }
            .badge-row {
                display: flex;
                gap: 0.5rem;
                flex-wrap: wrap;
                margin: 0.35rem 0 0.8rem 0;
            }
            .pill {
                display: inline-block;
                padding: 0.28rem 0.7rem;
                border-radius: 999px;
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0.02em;
                background: rgba(30,41,59,0.95);
                border: 1px solid rgba(148,163,184,0.20);
                color: #e2e8f0;
            }
            .pill.ok { background: rgba(16,185,129,0.16); color: #a7f3d0; }
            .pill.warn { background: rgba(245,158,11,0.16); color: #fde68a; }
            .pill.bad { background: rgba(239,68,68,0.16); color: #fecaca; }
            .section-panel {
                margin-top: 0.8rem;
                padding: 1rem 1.1rem;
                border-radius: 20px;
                background: rgba(15,23,42,0.66);
                border: 1px solid rgba(148,163,184,0.18);
                box-shadow: 0 12px 30px rgba(0,0,0,0.15);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_about_us() -> None:
    st.markdown('<div class="section-title">About Us</div>', unsafe_allow_html=True)
    st.write(
        "This app is a compact screening demo built with PyTorch Lightning models for diabetic foot ulcer grading and diabetic retinopathy grading. It is designed for fast, clean image-based triage and is not a substitute for a clinician."
    )
    st.write(
        "The DFU classifier includes Normal, Mild, Moderate, and Severe classes. The retinopathy classifier covers No DR through Proliferative DR. The combined diabetes screen only flags higher risk when both checks cross their confidence thresholds."
    )


def render_summary_cards() -> None:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="metric-card"><h4>DFU Model</h4><p>4 classes: Normal, Mild, Moderate, Severe</p></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="metric-card"><h4>Retina Model</h4><p>5 classes: No DR to Proliferative DR</p></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="metric-card"><h4>Combined Mode</h4><p>Runs both models and flags high-risk when both are positive above threshold</p></div>',
            unsafe_allow_html=True,
        )


def render_top_actions() -> None:
    left, right = st.columns([7, 1.5])
    with left:
        st.markdown('<div class="small-muted">Clean screening flows for foot, retina, and combined diabetes assessment.</div>', unsafe_allow_html=True)
    with right:
        if st.button("About Us", key="top_about", use_container_width=True):
            st.session_state.page = "about"
            st.rerun()


def render_home_page() -> None:
    st.markdown('<div class="section-title">Choose a screening mode</div>', unsafe_allow_html=True)
    st.write("Use the buttons below to open a dedicated workflow. About Us stays separate so the main screen stays clean.")

    cols = st.columns(3)

    buttons = [
        ("Detect DFU", "dfu", "Foot ulcer grading with the DFU model."),
        ("Detect Retinopathy", "retina", "Retinal severity grading with the DR model."),
        ("Detect Diabetes", "diabetes", "Combined screening using both models."),
    ]

    for col, (label, target, text) in zip(cols, buttons):
        with col:
            st.markdown(
                f'<div class="metric-card"><h4>{label}</h4><p>{text}</p></div>',
                unsafe_allow_html=True,
            )
            if st.button(label, key=f"home_{target}", use_container_width=True):
                st.session_state.page = target


def latest_dfu_model():
    return load_dfu_model_from_path(str(get_latest_dfu_checkpoint()))


def latest_retinal_model():
    return load_retinal_model_from_path(str(get_latest_retinal_checkpoint()))


def render_probabilities(labels, probs, positive_label_index=None):
    for idx, label in enumerate(labels):
        score = float(probs[idx])
        prefix = "●" if positive_label_index is not None and idx == positive_label_index else "○"
        st.write(f"{prefix} {label}: {score:.4f}")
        st.progress(min(max(score, 0.0), 1.0))


def render_status_banner(title: str, verdict: str, kind: str) -> None:
    cls = "ok" if kind == "ok" else "warn" if kind == "warn" else "bad"
    st.markdown(
        f'<div class="section-panel"><div class="badge-row"><span class="pill {cls}">{title}</span></div><div style="font-size:1.1rem;font-weight:700;">{verdict}</div></div>',
        unsafe_allow_html=True,
    )


def render_dfu_ui() -> None:
    st.markdown('<div class="section-title">Diabetic Foot Ulcer Detection</div>', unsafe_allow_html=True)
    st.write("Upload a foot image to classify it as Normal, Mild, Moderate, or Severe.")

    uploaded = st.file_uploader(
        "Upload a DFU image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="dfu_uploader"
    )

    if uploaded is None:
        st.info("Upload an image to run DFU prediction.")
        return

    try:
        model = latest_dfu_model()
        pil_img = Image.open(uploaded).convert("RGB")
        pred_idx, confidence, probs = predict_dfu(model, pil_img)
    except Exception as exc:
        st.exception(exc)
        return

    left, right = st.columns([1.05, 0.95])
    with left:
        st.image(pil_img, caption="Uploaded DFU Image", use_container_width=True)

    with right:
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.subheader("Prediction")
        render_status_banner("DFU Result", f"Severity: {DFU_CLASS_NAMES[pred_idx]}", "ok" if pred_idx == 0 else "warn")
        st.caption(f"Label: {pred_idx}  |  Confidence: {confidence:.4f}")
        if pred_idx == 0 and confidence >= DFU_NORMAL_THRESHOLD:
            st.info("Confident normal-foot pattern.")
        elif pred_idx == 0:
            st.warning("Normal predicted, but confidence is below the normal threshold.")
        else:
            st.warning("This foot is showing some diabetic-foot risk signal.")
        st.write("Class probabilities")
        render_probabilities(DFU_CLASS_NAMES, probs, positive_label_index=0)
        st.markdown('</div>', unsafe_allow_html=True)


def render_retinopathy_ui() -> None:
    st.markdown('<div class="section-title">Diabetic Retinopathy Detection</div>', unsafe_allow_html=True)
    st.write("Upload a retinal fundus image to classify DR severity.")

    uploaded = st.file_uploader(
        "Upload retinal fundus image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="retinal_uploader"
    )

    if uploaded is None:
        st.info("Upload an image to run retinopathy prediction.")
        return

    try:
        model = latest_retinal_model()
        pil_img = Image.open(uploaded).convert("RGB")
        pred_idx, confidence, probs = predict_retinal(model, pil_img)
    except Exception as exc:
        st.exception(exc)
        return

    left, right = st.columns([1, 1])
    with left:
        st.image(pil_img, caption="Uploaded Fundus Image", use_container_width=True)

    with right:
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.subheader("Prediction")
        render_status_banner("Retina Result", f"Severity: {RETINAL_CLASS_NAMES[pred_idx]}", "ok" if pred_idx == 0 else "warn")
        st.caption(f"Label: {pred_idx}  |  Confidence: {confidence:.4f}")
        st.write("Class probabilities")
        render_probabilities(RETINAL_CLASS_NAMES, probs, positive_label_index=0)
        st.markdown('</div>', unsafe_allow_html=True)


def render_both_ui() -> None:
    st.markdown('<div class="section-title">Detect Diabetes</div>', unsafe_allow_html=True)
    st.write("Upload one foot image and one retinal image. The screen combines both predictions into a simple diabetes risk rule.")

    c1, c2 = st.columns(2)
    with c1:
        foot_file = st.file_uploader(
            "Upload foot image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="both_foot"
        )
    with c2:
        retina_file = st.file_uploader(
            "Upload retinal image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="both_retina"
        )

    if foot_file is None or retina_file is None:
        st.info("Upload both images to run the combined assessment.")
        return

    try:
        dfu_model = latest_dfu_model()
        retinal_model = latest_retinal_model()

        foot_img = Image.open(foot_file).convert("RGB")
        retina_img = Image.open(retina_file).convert("RGB")

        dfu_idx, dfu_conf, dfu_probs = predict_dfu(dfu_model, foot_img)
        retina_idx, retina_conf, retina_probs = predict_retinal(retinal_model, retina_img)
    except Exception as exc:
        st.exception(exc)
        return

    dfu_label_normal = dfu_idx == 0
    retinal_label_normal = retina_idx == 0
    dfu_confident_abnormal = dfu_idx != 0 and dfu_conf >= DFU_NORMAL_THRESHOLD
    retinal_confident_abnormal = retina_idx != 0 and retina_conf >= RETINAL_NO_DR_THRESHOLD

    if dfu_label_normal and retinal_label_normal:
        verdict = "No diabetes signal: both models predict normal results."
        render_status_banner("Diabetes Screen", verdict, "ok")
    elif dfu_confident_abnormal and retinal_confident_abnormal:
        verdict = "Diabetes likely: both models are abnormal above threshold"
        render_status_banner("Diabetes Screen", verdict, "bad")
    else:
        verdict = "Maybe diabetes"
        render_status_banner("Diabetes Screen", verdict, "warn")

    a, b, c = st.columns(3)
    with a:
        if dfu_label_normal and retinal_label_normal:
            recommendation = "Routine monitoring is reasonable"
        elif dfu_confident_abnormal and retinal_confident_abnormal:
            recommendation = "Needs urgent clinical review"
        else:
            recommendation = "Consider follow-up tests and clinician review"
        st.markdown('<div class="metric-card"><h4>What is needed</h4><p>' + recommendation + '</p></div>', unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.image(foot_img, caption="Foot image", use_container_width=True)
        st.write("DFU probabilities:")
        render_probabilities(DFU_CLASS_NAMES, dfu_probs, positive_label_index=0)
    with right:
        st.image(retina_img, caption="Retina image", use_container_width=True)
        st.write("Retina probabilities:")
        render_probabilities(RETINAL_CLASS_NAMES, retina_probs, positive_label_index=0)


def main() -> None:
    st.set_page_config(page_title="DFU + Retinopathy Detector", layout="wide")
    inject_styles()

    if "page" not in st.session_state:
        st.session_state.page = "home"

    st.markdown(
        """
        <div class="hero">
            <h1>Medical Screening Studio</h1>
            <p>Dedicated flows for DFU, retinopathy, and combined diabetes screening with a clean, modern interface.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_top_actions()

    if st.session_state.page == "home":
        render_summary_cards()
        st.markdown(
            "<div class='small-muted'>Thresholds: DFU positive if non-normal prediction is at least 0.60 confidence. Retina positive if non-No DR prediction is at least 0.55 confidence. Combined mode is positive only when both are positive.</div>",
            unsafe_allow_html=True,
        )
        render_home_page()
        return

    if st.button("Back to Home", key="back_home"):
        st.session_state.page = "home"
        st.rerun()

    if st.session_state.page == "dfu":
        render_dfu_ui()
    elif st.session_state.page == "retina":
        render_retinopathy_ui()
    elif st.session_state.page == "diabetes":
        render_both_ui()
    elif st.session_state.page == "about":
        render_about_us()
    else:
        st.session_state.page = "home"
        st.rerun()


if __name__ == "__main__":
    main()
