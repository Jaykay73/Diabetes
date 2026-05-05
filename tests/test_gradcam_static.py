import numpy as np

from explainability.gradcam import normalize_heatmap


def test_normalize_heatmap_range() -> None:
    heatmap = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    normalized = normalize_heatmap(heatmap)

    assert normalized.dtype == np.float32
    assert float(normalized.min()) == 0.0
    assert float(normalized.max()) == 1.0


def test_normalize_heatmap_constant_returns_zeros() -> None:
    heatmap = np.ones((4, 4), dtype=np.float32)

    normalized = normalize_heatmap(heatmap)

    assert np.allclose(normalized, 0.0)
