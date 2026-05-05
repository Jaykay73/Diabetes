"""LIME image explanations."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from skimage.segmentation import slic


def explain_with_lime(model_predict_fn, image_np: np.ndarray, num_samples: int = 1000, top_k: int = 8) -> Image.Image:
    """Explain an image with LIME and color supporting/opposing superpixels."""

    try:
        from lime.lime_image import LimeImageExplainer
    except ImportError as exc:
        raise ImportError("Install lime to use explain_with_lime.") from exc

    explainer = LimeImageExplainer()
    explanation = explainer.explain_instance(
        image_np.astype(np.uint8),
        classifier_fn=model_predict_fn,
        segmentation_fn=lambda image: slic(image, n_segments=120, compactness=10, sigma=1),
        top_labels=5,
        hide_color=0,
        num_samples=num_samples,
    )
    label = explanation.top_labels[0]
    segments = explanation.segments
    weights = dict(explanation.local_exp[label])
    strongest = sorted(weights.items(), key=lambda item: abs(item[1]), reverse=True)[:top_k]

    overlay = image_np.astype(np.uint8).copy()
    color_layer = overlay.copy()
    for segment_id, weight in strongest:
        mask = segments == segment_id
        color = np.array([0, 220, 0], dtype=np.uint8) if weight >= 0 else np.array([230, 0, 0], dtype=np.uint8)
        color_layer[mask] = color

    blended = cv2.addWeighted(overlay, 0.65, color_layer, 0.35, 0)
    return Image.fromarray(blended)
