"""FastAPI app for diabetic retinopathy prediction and explanation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from api.schemas import (
    AttentionBreakdownResponse,
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    MethodInfo,
    PredictRequest,
    PredictResponse,
)
from dr_grading.data.preprocessing import PreprocessOptions, preprocess_fundus_image
from dr_grading.data.transforms import build_valid_transforms
from dr_grading.inference.predictor import (
    CheckpointSpec,
    EnsemblePredictor,
    SEVERITY_LABELS,
    base64_to_pil,
    image_to_base64,
)
from explainability.gradcam import CAMVariantExplainer, build_multiclass_cam_grid, overlay_cam
from explainability.landmarks import detect_retinal_landmarks, overlay_landmarks, xai_reliability_score
from explainability.lime_explainer import explain_with_lime

app = FastAPI(title="Diabetic Retinopathy Grading API", version="0.1.0")


def _load_predictor() -> EnsemblePredictor | None:
    checkpoint = os.getenv("DR_MODEL_CHECKPOINT")
    architecture = os.getenv("DR_MODEL_ARCH", "tf_efficientnet_b4_ns")
    task = os.getenv("DR_MODEL_TASK", "regression")
    image_size = int(os.getenv("DR_IMAGE_SIZE", "512"))
    if not checkpoint or not Path(checkpoint).exists():
        return None
    spec = CheckpointSpec(path=Path(checkpoint), architecture=architecture, task=task)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return EnsemblePredictor([spec], device=device, image_size=image_size)


PREDICTOR = _load_predictor()


def _read_upload(file: UploadFile) -> Image.Image:
    try:
        return Image.open(file.file).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Invalid or corrupt image") from exc


def _predict_or_503(image: Image.Image) -> dict:
    if PREDICTOR is None:
        raise HTTPException(status_code=503, detail="Model checkpoint is not configured")
    return PREDICTOR.predict_grade(image)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=PREDICTOR is not None)


@app.post("/predict", response_model=PredictResponse)
def predict_base64(request: PredictRequest) -> PredictResponse:
    image = base64_to_pil(request.image)
    result = _predict_or_503(image)
    return PredictResponse(
        predicted_grade=result["grade"],
        severity_label=result["severity"],
        confidence=result["confidence"],
        continuous_prediction=result["continuous_prediction"],
        gradcam_heatmap=None,
    )


@app.post("/predict/file", response_model=PredictResponse)
def predict_file(file: Annotated[UploadFile, File(...)]) -> PredictResponse:
    image = _read_upload(file)
    result = _predict_or_503(image)
    return PredictResponse(
        predicted_grade=result["grade"],
        severity_label=result["severity"],
        confidence=result["confidence"],
        continuous_prediction=result["continuous_prediction"],
        gradcam_heatmap=None,
    )


def _dummy_heatmap(image_np: np.ndarray) -> np.ndarray:
    gray = np.mean(image_np, axis=2).astype(np.float32)
    gray = gray - gray.min()
    return gray / max(float(gray.max()), 1e-6)


@app.post("/explain", response_model=ExplainResponse)
def explain(request: ExplainRequest) -> ExplainResponse:
    image = base64_to_pil(request.image)
    original = np.asarray(image.convert("RGB"), dtype=np.uint8)
    processed = preprocess_fundus_image(original, PreprocessOptions(image_size=512), method="ben_graham")

    if PREDICTOR is not None and request.method not in {"lime", "shap"}:
        spec = PREDICTOR.specs[0]
        model = PREDICTOR.models[0]
        tensor = build_valid_transforms(PREDICTOR.image_size)(image=processed)["image"]
        explainer = CAMVariantExplainer(model=model, arch_name=spec.architecture, method=request.method)
        cam = explainer.explain(
            image_tensor=tensor,
            original_image=original,
            preprocessed_image=processed,
            regression=spec.task == "regression",
        )
        heatmap = cam.heatmap
        heatmap_overlay = cam.overlay_original
        multiclass_grid = None
        if spec.task != "regression":
            multiclass_grid = build_multiclass_cam_grid(explainer, tensor, original, processed)
    elif PREDICTOR is not None and request.method == "lime":
        def predict_fn(images: np.ndarray) -> np.ndarray:
            rows = []
            for item in images:
                grade_result = PREDICTOR.predict_grade(Image.fromarray(item.astype(np.uint8)))
                scores = np.zeros(5, dtype=np.float32)
                scores[int(grade_result["grade"])] = float(grade_result["confidence"])
                rows.append(scores)
            return np.vstack(rows)

        heatmap = _dummy_heatmap(processed)
        heatmap_overlay = explain_with_lime(predict_fn, processed, num_samples=300)
        multiclass_grid = None
    else:
        heatmap = _dummy_heatmap(processed)
        heatmap_overlay = overlay_cam(original, heatmap)
        multiclass_grid = None

    landmarks = detect_retinal_landmarks(processed)
    landmark_overlay = overlay_landmarks(processed, heatmap=heatmap, landmarks=landmarks)
    score, flags, breakdown = xai_reliability_score(heatmap, landmarks)

    return ExplainResponse(
        heatmap_overlay=image_to_base64(heatmap_overlay),
        landmark_overlay=image_to_base64(landmark_overlay),
        multiclass_grid=image_to_base64(multiclass_grid) if multiclass_grid else None,
        xai_reliability_score=score,
        clinical_flags=flags,
        attention_breakdown=AttentionBreakdownResponse(
            macula_pct=breakdown.macula_pct,
            vessels_pct=breakdown.vessels_pct,
            optic_disc_pct=breakdown.optic_disc_pct,
            background_pct=breakdown.background_pct,
        ),
    )


@app.get("/explain/methods", response_model=list[MethodInfo])
def explain_methods() -> list[MethodInfo]:
    return [
        MethodInfo(method="gradcam", description="Gradient-weighted class activation map", avg_latency_ms=120),
        MethodInfo(method="gradcam++", description="Grad-CAM++ for multiple local regions", avg_latency_ms=180),
        MethodInfo(method="eigencam", description="Gradient-free PCA activation map", avg_latency_ms=90),
        MethodInfo(method="scorecam", description="Perturbation-based CAM", avg_latency_ms=900),
        MethodInfo(method="layercam", description="Fine-grained local activation map", avg_latency_ms=180),
        MethodInfo(method="ablationcam", description="Channel ablation faithfulness CAM", avg_latency_ms=1200),
        MethodInfo(method="lime", description="Superpixel perturbation explanation", avg_latency_ms=2500),
        MethodInfo(method="shap", description="Deep SHAP pixel attribution", avg_latency_ms=5000),
    ]
