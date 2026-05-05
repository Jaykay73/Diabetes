"""Rule-based retinal landmark detection and clinical XAI alignment scoring."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
from skimage.filters import frangi


@dataclass(frozen=True)
class AttentionBreakdown:
    macula_pct: float
    vessels_pct: float
    optic_disc_pct: float
    background_pct: float


def _empty_mask(shape: tuple[int, int]) -> np.ndarray:
    return np.zeros(shape, dtype=np.uint8)


def detect_retinal_landmarks(image_np: np.ndarray) -> dict[str, object]:
    """Detect optic disc, approximate macula, and vessel network masks.

    This is intentionally lightweight and deterministic. It is useful for review
    overlays, but it is not a substitute for lesion-level clinical annotations.
    """

    if image_np.ndim != 3 or image_np.shape[2] != 3:
        raise ValueError(f"Expected RGB image [H,W,3], got {image_np.shape}")

    rgb = image_np.astype(np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape

    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(height, width) / 60)
    threshold = int(np.percentile(blurred[blurred > 0], 97)) if np.any(blurred > 0) else 255
    _, bright = cv2.threshold(blurred, max(0, threshold - 1), 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    optic_disc_box: tuple[int, int, int, int] | None = None
    optic_disc_mask = _empty_mask((height, width))
    if contours:
        candidates = sorted(contours, key=cv2.contourArea, reverse=True)
        for contour in candidates:
            area = cv2.contourArea(contour)
            if area < 0.001 * height * width:
                continue
            x, y, box_w, box_h = cv2.boundingRect(contour)
            ratio = box_w / max(box_h, 1)
            if 0.5 <= ratio <= 1.8:
                optic_disc_box = (int(x), int(y), int(box_w), int(box_h))
                cv2.drawContours(optic_disc_mask, [contour], -1, 255, thickness=-1)
                break

    macula_mask = _empty_mask((height, width))
    cv2.ellipse(
        macula_mask,
        (width // 2, height // 2),
        (max(8, width // 10), max(8, height // 14)),
        0,
        0,
        360,
        255,
        -1,
    )

    vesselness = frangi(1.0 - gray.astype(np.float32) / 255.0)
    if np.isfinite(vesselness).any() and float(vesselness.max()) > 0:
        vessel_mask = (vesselness > np.percentile(vesselness, 92)).astype(np.uint8) * 255
    else:
        vessel_mask = _empty_mask((height, width))

    return {
        "optic_disc_box": optic_disc_box,
        "optic_disc_mask": optic_disc_mask,
        "macula_mask": macula_mask,
        "vessel_mask": vessel_mask,
    }


def overlay_landmarks(
    image_np: np.ndarray,
    heatmap: np.ndarray | None = None,
    landmarks: dict[str, object] | None = None,
    alpha: float = 0.35,
) -> Image.Image:
    """Draw landmark contours and optional heatmap over an RGB image."""

    canvas = image_np.astype(np.uint8).copy()
    if heatmap is not None:
        heatmap = cv2.resize(heatmap, (canvas.shape[1], canvas.shape[0]))
        colored = cv2.applyColorMap(np.uint8(np.clip(heatmap, 0, 1) * 255), cv2.COLORMAP_JET)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        canvas = cv2.addWeighted(canvas, 1 - alpha, colored, alpha, 0)

    landmarks = landmarks or detect_retinal_landmarks(image_np)
    macula_mask = landmarks["macula_mask"].astype(np.uint8)
    vessel_mask = landmarks["vessel_mask"].astype(np.uint8)
    optic_disc_mask = landmarks["optic_disc_mask"].astype(np.uint8)

    for mask, color, label in [
        (macula_mask, (0, 255, 0), "Macula"),
        (vessel_mask, (255, 255, 0), "Vessel network"),
        (optic_disc_mask, (255, 0, 255), "Optic Disc"),
    ]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, color, 1)
        if contours:
            x, y, _, _ = cv2.boundingRect(max(contours, key=cv2.contourArea))
            cv2.putText(canvas, label, (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    return Image.fromarray(canvas)


def compute_attention_breakdown(heatmap: np.ndarray, landmarks: dict[str, object]) -> AttentionBreakdown:
    """Compute top-20%-activation mass over clinical regions vs background."""

    heatmap = np.asarray(heatmap, dtype=np.float32)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    top = heatmap >= np.percentile(heatmap, 80)
    total = float(top.sum())
    if total <= 0:
        return AttentionBreakdown(0.0, 0.0, 0.0, 1.0)

    macula = cv2.resize(landmarks["macula_mask"], (heatmap.shape[1], heatmap.shape[0])) > 0
    vessels = cv2.resize(landmarks["vessel_mask"], (heatmap.shape[1], heatmap.shape[0])) > 0
    optic = cv2.resize(landmarks["optic_disc_mask"], (heatmap.shape[1], heatmap.shape[0])) > 0

    macula_pct = float((top & macula).sum() / total)
    vessels_pct = float((top & vessels).sum() / total)
    optic_pct = float((top & optic).sum() / total)
    clinical = macula | vessels | optic
    background_pct = float((top & ~clinical).sum() / total)
    return AttentionBreakdown(macula_pct, vessels_pct, optic_pct, background_pct)


def xai_reliability_score(heatmap: np.ndarray, landmarks: dict[str, object]) -> tuple[float, list[str], AttentionBreakdown]:
    """Return reliability score, clinical flags, and attention breakdown."""

    breakdown = compute_attention_breakdown(heatmap, landmarks)
    score = float(np.clip(breakdown.macula_pct + breakdown.vessels_pct, 0.0, 1.0))
    flags: list[str] = []
    if breakdown.macula_pct >= 0.20:
        flags.append("Macular involvement detected")
    if breakdown.vessels_pct >= 0.20:
        flags.append("Vascular changes detected")
    if score < 0.40:
        flags.append("Model may be using non-clinical features")
    if breakdown.background_pct >= 0.75:
        flags.append("Low model confidence - review manually")
    return score, flags, breakdown
