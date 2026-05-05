# Phase 5: Training Loop

Training entry point:

```bash
python scripts/create_folds.py
python scripts/train.py --fold 0 --use-weighted-sampler
```

The trainer includes:

- mixed precision with `torch.cuda.amp`
- gradient clipping
- warmup plus cosine LR schedule
- early stopping on validation QWK
- best checkpoint per fold
- W&B logging
- class-weighted classification loss

## QWK First

Use QWK as the validation decision metric. Accuracy can look reasonable while adjacent
grade mistakes and severe under-grading make the model clinically weak.

## Common Mistakes

- Saving the lowest validation loss instead of highest QWK.
- Searching thresholds on the test set.
- Balancing validation data.
- Forgetting to seed fold creation.
- Letting external Kaggle 2015 images leak into APTOS validation folds.
