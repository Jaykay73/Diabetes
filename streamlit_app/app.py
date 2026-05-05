"""Streamlit demo for diabetic retinopathy grading and explainability."""

from __future__ import annotations

import base64
import os
from io import BytesIO

import requests
import streamlit as st
from PIL import Image

from dr_grading.data.preprocessing import PreprocessOptions, apply_clahe_rgb, preprocess_fundus_image


API_URL = os.getenv("DR_API_URL", "http://localhost:8000")


def pil_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_pil(payload: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(payload))).convert("RGB")


def severity_text(grade: int) -> str:
    return {
        0: "No diabetic retinopathy detected by the model.",
        1: "Mild DR pattern. Manual review is recommended for subtle lesions.",
        2: "Moderate DR pattern. Clinical follow-up should be prioritized.",
        3: "Severe DR pattern. Prompt specialist review is recommended.",
        4: "Proliferative DR pattern. Urgent clinical review is recommended.",
    }.get(grade, "Unknown grade.")


st.set_page_config(page_title="DR Grading", layout="wide")
st.title("Diabetic Retinopathy Grading")

uploaded = st.sidebar.file_uploader("Upload fundus image", type=["png", "jpg", "jpeg"])
method = st.sidebar.selectbox("XAI method", ["gradcam", "gradcam++", "eigencam", "scorecam", "layercam", "ablationcam", "lime", "shap"])
opacity = st.sidebar.slider("Heatmap opacity", min_value=0, max_value=100, value=35)
show_landmarks = st.sidebar.checkbox("Show landmark overlay", value=True)

tabs = st.tabs(["Prediction", "Explain Prediction"])

if uploaded is None:
    st.info("Upload a retinal fundus image to run prediction and explanation.")
    st.stop()

image = Image.open(uploaded).convert("RGB")
image_np = __import__("numpy").asarray(image)
ben_graham = preprocess_fundus_image(image_np, PreprocessOptions(image_size=512), method="ben_graham")
clahe = apply_clahe_rgb(ben_graham)

with tabs[0]:
    col1, col2, col3 = st.columns(3)
    col1.image(image, caption="Original", use_container_width=True)
    col2.image(ben_graham, caption="Ben Graham", use_container_width=True)
    col3.image(clahe, caption="CLAHE alternative", use_container_width=True)

    try:
        response = requests.post(f"{API_URL}/predict", json={"image": pil_to_base64(image)}, timeout=60)
        response.raise_for_status()
        prediction = response.json()
        grade = int(prediction["predicted_grade"])
        st.metric("Predicted grade", f"{grade} - {prediction['severity_label']}")
        st.progress(float(prediction["confidence"]))
        st.write(severity_text(grade))
    except requests.RequestException as exc:
        st.warning(f"Prediction API unavailable: {exc}")

with tabs[1]:
    try:
        response = requests.post(
            f"{API_URL}/explain",
            json={"image": pil_to_base64(image), "method": method},
            timeout=180,
        )
        response.raise_for_status()
        explanation = response.json()
        heatmap = base64_to_pil(explanation["heatmap_overlay"])
        landmark = base64_to_pil(explanation["landmark_overlay"])

        col1, col2 = st.columns(2)
        col1.image(heatmap, caption=f"{method} heatmap overlay", use_container_width=True)
        if show_landmarks:
            col2.image(landmark, caption="Clinical landmark overlay", use_container_width=True)

        if explanation.get("multiclass_grid"):
            st.image(base64_to_pil(explanation["multiclass_grid"]), caption="Multi-class CAM grid", use_container_width=True)

        score = float(explanation["xai_reliability_score"])
        st.metric("XAI Reliability Score", f"{score:.2f}")
        st.json(
            {
                "clinical_flags": explanation["clinical_flags"],
                "attention_breakdown": explanation["attention_breakdown"],
                "opacity_requested": opacity,
            }
        )
    except requests.RequestException as exc:
        st.warning(f"Explanation API unavailable: {exc}")
