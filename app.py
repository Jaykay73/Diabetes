
# app.py
# Professional Streamlit app for Diabetic Retinopathy classification

import io
import os
import cv2
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ==============================
# Page configuration
# ==============================
st.set_page_config(
    page_title="Diabetic Retinopathy AI Classifier",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# Constants
# ==============================
IMAGE_SIZE = 384
NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR"
]

CLASS_DESCRIPTIONS = {
    0: "No visible signs of diabetic retinopathy.",
    1: "Early-stage diabetic retinopathy, usually with mild microaneurysms.",
    2: "Moderate disease signs are present. Medical review is recommended.",
    3: "Severe diabetic retinopathy. This is a high-risk stage.",
    4: "Advanced/proliferative diabetic retinopathy. Urgent specialist review is strongly advised."
}

RISK_LEVELS = {
    0: ("Low Risk", "#16a34a"),
    1: ("Early Risk", "#ca8a04"),
    2: ("Medium Risk", "#ea580c"),
    3: ("High Risk", "#dc2626"),
    4: ("Critical Risk", "#7f1d1d")
}

MODEL_PATH_DEFAULT = "best_model_final (1).pth"


# ==============================
# Styling
# ==============================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #f4f7f9;
    }
    .hero {
        padding: 3rem 2.5rem;
        border-radius: 24px;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        color: white;
        margin-bottom: 2.5rem;
        box-shadow: 0 20px 40px rgba(15, 23, 42, 0.2);
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -50%; left: -50%; width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 60%);
        transform: rotate(30deg);
    }
    .hero h1 {
        margin: 0;
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        position: relative;
        z-index: 1;
    }
    .hero p {
        margin-top: 1rem;
        color: #94a3b8;
        font-size: 1.15rem;
        max-width: 700px;
        line-height: 1.6;
        position: relative;
        z-index: 1;
    }
    .metric-card {
        background: white;
        padding: 1.8rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
        border: 1px solid #e2e8f0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        height: 100%;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(15, 23, 42, 0.1);
    }
    .metric-card h3 {
        color: #0f172a;
        margin-top: 0;
        font-size: 1.3rem;
        font-weight: 700;
    }
    .metric-card p {
        color: #64748b;
        font-size: 1rem;
        line-height: 1.5;
        margin-bottom: 0;
    }
    .prediction-card {
        background: white;
        padding: 2.5rem 2rem;
        border-radius: 24px;
        box-shadow: 0 15px 35px rgba(15, 23, 42, 0.08);
        border: 1px solid #e2e8f0;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .prediction-card::after {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 6px;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
    }
    .risk-pill {
        display: inline-block;
        color: white;
        padding: 0.5rem 1.2rem;
        border-radius: 999px;
        font-size: 0.95rem;
        font-weight: 700;
        margin-bottom: 1.2rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .prediction-card h2 {
        color: #0f172a;
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .prediction-card h3 {
        color: #334155;
        font-size: 1.4rem;
        font-weight: 600;
        margin-top: 1.5rem;
    }
    .small-muted {
        color: #64748b;
        font-size: 1.05rem;
        line-height: 1.6;
        margin-bottom: 0;
    }
    .warning-box {
        background: linear-gradient(to right, #fff7ed, #ffedd5);
        border-left: 5px solid #f97316;
        color: #9a3412;
        padding: 1.5rem;
        border-radius: 0 16px 16px 0;
        margin-top: 2rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.03);
    }
    
    /* Streamlit elements overrides */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    .stFileUploader > div > div {
        background-color: white;
        border-radius: 20px;
        border: 2px dashed #cbd5e1;
        transition: all 0.3s ease;
        padding: 2rem;
    }
    .stFileUploader > div > div:hover {
        border-color: #3b82f6;
        background-color: #f8fafc;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ==============================
# Model architecture
# Must match training notebook exactly
# ==============================
class DRClassifier(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.4, pretrained=False):
        super(DRClassifier, self).__init__()

        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg"
        )

        feature_dim = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(p=dropout_rate),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


# ==============================
# Image preprocessing
# Same idea as training: crop black border + Ben Graham enhancement
# ==============================
def crop_black_border(image, tolerance=7):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray > tolerance
    coords = np.argwhere(mask)

    if len(coords) == 0:
        return image

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return image[y0:y1, x0:x1]


def ben_graham_preprocessing_from_pil(pil_image, image_size=IMAGE_SIZE):
    image_rgb = np.array(pil_image.convert("RGB"))
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    image_bgr = crop_black_border(image_bgr, tolerance=7)

    h, w = image_bgr.shape[:2]
    interp = cv2.INTER_AREA if max(h, w) > image_size else cv2.INTER_LANCZOS4
    image_bgr = cv2.resize(image_bgr, (image_size, image_size), interpolation=interp)

    sigma = 10
    blurred = cv2.GaussianBlur(image_bgr, (0, 0), sigma)
    image_bgr = cv2.addWeighted(image_bgr, 4, blurred, -4, 128)

    mask = np.zeros_like(image_bgr)
    center = (image_size // 2, image_size // 2)
    radius = int(image_size * 0.48)
    cv2.circle(mask, center, radius, (1, 1, 1), -1)
    image_bgr = image_bgr * mask + 128 * (1 - mask)
    image_bgr = image_bgr.clip(0, 255).astype(np.uint8)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb


def get_val_transform(image_size=IMAGE_SIZE):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ==============================
# Grad-CAM
# ==============================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None

        self.forward_hook = target_layer.register_forward_hook(self._save_features)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_features(self, module, input, output):
        self.feature_maps = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()

        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        input_tensor = input_tensor.to(DEVICE)
        input_tensor.requires_grad_(True)

        logits = self.model(input_tensor)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        gradients = self.gradients[0]
        feature_maps = self.feature_maps[0]

        weights = gradients.mean(dim=(1, 2))
        cam = torch.zeros(feature_maps.shape[1:], device=DEVICE)

        for k, w in enumerate(weights):
            cam += w * feature_maps[k]

        cam = F.relu(cam)

        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.detach().cpu().numpy()

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


def create_gradcam_overlay(original_rgb, cam_array, alpha=0.45):
    h, w = original_rgb.shape[:2]
    cam_resized = cv2.resize(cam_array, (w, h))

    heatmap_uint8 = np.uint8(255 * cam_resized)
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(
        heatmap_rgb.astype(np.float32), alpha,
        original_rgb.astype(np.float32), 1 - alpha,
        0
    ).astype(np.uint8)

    return heatmap_rgb, overlay


# ==============================
# Load model
# ==============================
@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = DRClassifier(num_classes=NUM_CLASSES, dropout_rate=0.4, pretrained=False)

    checkpoint = torch.load(model_path, map_location=DEVICE)

    # Handles both raw state_dict and checkpoint dict formats
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    model.load_state_dict(checkpoint, strict=True)
    model.to(DEVICE)
    model.eval()
    return model


def predict(model, image_tensor):
    model.eval()

    with torch.no_grad():
        logits = model(image_tensor.unsqueeze(0).to(DEVICE))
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

    return pred_class, confidence, probs


def probability_chart(probs):
    df = pd.DataFrame({
        "Class": [f"{i} - {name}" for i, name in enumerate(CLASS_NAMES)],
        "Probability": probs
    })
    return df


# ==============================
# Sidebar
# ==============================
# with st.sidebar:
#     st.image("https://img.icons8.com/color/96/ophthalmology.png", width=72)
#     st.title("DR Classifier")
st.markdown("### Model settings")
model_path = st.text_input(
"Model path",
value=MODEL_PATH_DEFAULT,
 help="Put your .pth file in the same folder as app.py, or enter the full path."
    )
show_gradcam = st.checkbox("Show Grad-CAM explainability", value=True)
show_preprocessed = st.checkbox("Show preprocessed image", value=True)

#     st.markdown("---")
#     st.markdown(
#         """
#         **Classes**
#         - 0: No DR
#         - 1: Mild
#         - 2: Moderate
#         - 3: Severe
#         - 4: Proliferative
#         """
#     )

#     st.markdown("---")
#     st.caption(f"Running on: {DEVICE}")


# ==============================
# Main interface
# ==============================
st.markdown(
    """
    <div class="hero">
        <h1>Diabetic Retinopathy AI Screening</h1>
        <p>
            Experience state-of-the-art Deep Learning for medical imaging. Upload a retinal fundus image, and our model will predict the Diabetic Retinopathy grade, provide confidence distributions, and generate a visual explainability map.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload a retinal fundus image",
    type=["jpg", "jpeg", "png"],
    help="Use a clear retinal fundus image similar to the APTOS dataset images."
)

try:
    model = load_model(model_path)
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Could not load model: {e}")
    st.info("Make sure your .pth file is in the same folder as app.py and the filename is correct.")

if uploaded_file is not None and model_loaded:
    pil_image = Image.open(uploaded_file)

    processed_rgb = ben_graham_preprocessing_from_pil(pil_image)
    transform = get_val_transform()
    image_tensor = transform(image=processed_rgb)["image"]

    pred_class, confidence, probs = predict(model, image_tensor)
    risk_label, risk_color = RISK_LEVELS[pred_class]

    left, right = st.columns([1.05, 1])

    with left:
        st.subheader("Input Image")

        col_a, col_b = st.columns(2)

        with col_a:
            st.image(pil_image, caption="Original uploaded image", use_container_width=True)

        with col_b:
            if show_preprocessed:
                st.image(processed_rgb, caption="Model preprocessing view", use_container_width=True)
            else:
                st.info("Enable preprocessed image from the sidebar to view it.")

    with right:
        st.subheader("Prediction Result")

        st.markdown(
            f"""
            <div class="prediction-card">
                <span class="risk-pill" style="background:{risk_color};">{risk_label}</span>
                <h2 style="margin-bottom:0.2rem;">Grade {pred_class}: {CLASS_NAMES[pred_class]}</h2>
                <p class="small-muted">{CLASS_DESCRIPTIONS[pred_class]}</p>
                <h3>Confidence: {confidence * 100:.2f}%</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### Probability Distribution")
        prob_df = probability_chart(probs)
        st.bar_chart(prob_df.set_index("Class"))

        with st.expander("View exact probabilities"):
            st.dataframe(
                prob_df.assign(Probability=lambda x: (x["Probability"] * 100).round(2)),
                use_container_width=True
            )

    if show_gradcam:
        st.markdown("---")
        st.subheader("Explainability: Grad-CAM Attention Map")

        try:
            gradcam = GradCAM(model, model.backbone.conv_head)
            cam = gradcam.generate(image_tensor, class_idx=pred_class)
            gradcam.remove_hooks()

            heatmap_rgb, overlay = create_gradcam_overlay(processed_rgb, cam)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.image(processed_rgb, caption="Preprocessed image", use_container_width=True)
            with c2:
                st.image(heatmap_rgb, caption="Grad-CAM heatmap", use_container_width=True)
            with c3:
                st.image(overlay, caption="Attention overlay", use_container_width=True)

            st.caption(
                "Red/yellow regions indicate areas that contributed more strongly to the predicted class. "
                "This is an explainability aid, not a clinical diagnosis."
            )
        except Exception as e:
            st.warning(f"Grad-CAM could not be generated: {e}")

    st.markdown(
        """
        <div class="warning-box">
            <div style="font-size: 1.5rem;">⚠️</div>
            <div>
                <strong>Medical disclaimer:</strong> This application is for AI demonstration and decision-support only.
                It should not be used as a final medical diagnosis. A qualified ophthalmologist or clinician should review
                all serious or uncertain cases.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

elif uploaded_file is None:
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="metric-card">
                <h3>🧠 Model Architecture</h3>
                <p>EfficientNet-B0 transfer learning with custom classification head.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
            <div class="metric-card">
                <h3>🔬 Image Preprocessing</h3>
                <p>Ben Graham fundus enhancement and ImageNet normalization.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            """
            <div class="metric-card">
                <h3>🎯 Diagnostic Output</h3>
                <p>5-class DR grade prediction with confidence distribution and Grad-CAM.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.info("Upload a fundus image to test the model.")
