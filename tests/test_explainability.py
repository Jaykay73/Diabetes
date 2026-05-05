import base64
from io import BytesIO

import numpy as np
from PIL import Image

from explainability.landmarks import detect_retinal_landmarks, xai_reliability_score


def _base64_image() -> str:
    image = Image.new("RGB", (64, 64), color=(20, 20, 20))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def test_landmark_detection_optic_disc_synthetic() -> None:
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:128, :128]
    mask = (yy - 40) ** 2 + (xx - 90) ** 2 <= 15**2
    image[mask] = 255

    landmarks = detect_retinal_landmarks(image)

    assert landmarks["optic_disc_box"] is not None


def test_xai_reliability_score_range() -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[20:40, 20:40] = 180
    heatmap = np.zeros((64, 64), dtype=np.float32)
    heatmap[24:36, 24:36] = 1.0
    landmarks = detect_retinal_landmarks(image)

    score, flags, breakdown = xai_reliability_score(heatmap, landmarks)

    assert 0.0 <= score <= 1.0
    assert isinstance(flags, list)
    assert 0.0 <= breakdown.background_pct <= 1.0


def test_explain_endpoint_returns_base64() -> None:
    pytest = __import__("pytest")
    pytest.importorskip("fastapi")
    pytest.importorskip("torch")
    from fastapi.testclient import TestClient

    from api.main import app

    client = TestClient(app)
    response = client.post("/explain", json={"image": _base64_image(), "method": "gradcam"})

    assert response.status_code == 200
    payload = response.json()
    base64.b64decode(payload["heatmap_overlay"])
    base64.b64decode(payload["landmark_overlay"])
    assert 0.0 <= payload["xai_reliability_score"] <= 1.0
