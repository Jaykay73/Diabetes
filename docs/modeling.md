# Phase 4: Model Architecture

## Approach A: EfficientNet-B4 Classifier

`BaselineEfficientNetClassifier` uses a pretrained timm EfficientNet-B4 backbone with a
5-logit classification head. It is simple and useful for smoke tests, but it optimizes
cross-entropy rather than the ordered structure of DR grades.

Typical loss:

```python
loss = torch.nn.CrossEntropyLoss(weight=class_weights)(logits, labels)
```

## Approach B: Regression Trick

`RegressionBackboneModel` predicts one continuous number from 0 to 4 and trains with MSE.
At inference, predictions are clipped and rounded, or converted with optimized thresholds.

This often beats plain classification because QWK penalizes ordered distance. Predicting
`2.7` for a true grade 3 is much better aligned with QWK than forcing independent class
logits where grade 0 and grade 4 are treated as unrelated categories.

Typical loss:

```python
loss = torch.nn.MSELoss()(prediction, labels.float().view(-1, 1))
```

## Approach C: Strong Ordinal Model

`OrdinalBackboneModel` supports stronger timm backbones such as:

- `tf_efficientnet_b5_ns`
- `tf_efficientnet_b6_ns`
- `convnext_large_in22k`
- `tf_efficientnetv2_m`

It uses:

- features-only backbone
- GeM pooling
- ordinal threshold logits for `label > 0`, `label > 1`, `label > 2`, `label > 3`
- CORAL-style binary cross-entropy
- optional label smoothing

## Common Mistakes

- Reporting accuracy instead of QWK.
- Applying softmax to regression outputs.
- Rounding validation predictions before threshold search.
- Assuming all timm models expose the same final convolution layer.
- Using ConvNeXt-Large at high resolution without checking VRAM and batch size.
