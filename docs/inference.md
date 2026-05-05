# Phase 6: Inference and Ensembling

Generate test predictions with one or more checkpoints:

```bash
python scripts/predict_submission.py \
  --checkpoint models/checkpoints/tf_efficientnet_b4_ns_fold0.pth \
  --architecture tf_efficientnet_b4_ns \
  --task regression \
  --thresholds 0.55 1.48 2.45 3.42 \
  --output submission.csv
```

For fold or multi-model ensembles, repeat `--checkpoint`, `--architecture`, `--task`, and
optionally `--weight`.

## Included

- original + horizontal flip + vertical flip TTA
- fold checkpoint averaging
- multi-model weighted averaging
- continuous prediction to grade conversion
- scipy threshold optimization helper

## Common Mistakes

- Averaging rounded class labels instead of continuous predictions.
- Using thresholds optimized on test predictions.
- Forgetting to apply the same preprocessing at inference as training.
- Running TTA that changes clinical meaning. H/V flips are acceptable for fundus grading;
  aggressive color or geometric TTA can be risky.
