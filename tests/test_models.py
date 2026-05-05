import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from dr_grading.models.architectures import (  # noqa: E402
    BaselineEfficientNetClassifier,
    OrdinalBackboneModel,
    RegressionBackboneModel,
    coral_loss,
)


def test_baseline_model_forward_shape() -> None:
    model = BaselineEfficientNetClassifier(
        arch="resnet18",
        pretrained=False,
        num_classes=5,
        dropout=0.1,
    )
    output = model(torch.randn(2, 3, 64, 64))

    assert output.shape == (2, 5)


def test_regression_model_forward_shape() -> None:
    model = RegressionBackboneModel(arch="resnet18", pretrained=False, dropout=0.1)
    output = model(torch.randn(2, 3, 64, 64))

    assert output.shape == (2, 1)


def test_ordinal_model_and_coral_loss() -> None:
    model = OrdinalBackboneModel(
        arch="resnet18",
        pretrained=False,
        num_classes=5,
        dropout=0.1,
    )
    logits = model(torch.randn(2, 3, 64, 64))
    levels = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.float32)
    loss = coral_loss(logits, levels)

    assert logits.shape == (2, 4)
    assert loss.item() >= 0
