# Phase 3: Dataset and Dataloader

Use `StratifiedKFold`, not a random split:

```bash
python scripts/create_folds.py --config configs/config.yaml
```

The output defaults to:

```text
data/interim/aptos2019_folds.csv
```

## Class Imbalance

APTOS is heavily imbalanced toward grade 0. For training, use one of:

- `WeightedRandomSampler` for balanced sampling per epoch.
- Oversampling minority classes in the dataframe.
- Class-weighted loss in Phase 5.

Do not balance the validation fold. Validation must reflect the original distribution.

## 2015 Kaggle DR Data

Use Kaggle 2015 primarily for pretraining or warm-starting. It comes from a different
distribution and has noisier labels, so mixing it into final APTOS training can help or
hurt. If you mix it in, freeze the APTOS folds first and never include external data in
APTOS validation threshold search.

## Common Mistakes

- Random split instead of stratified folds.
- Duplicate patient/image leakage across folds when using external datasets.
- Oversampling validation data.
- Optimizing thresholds on test predictions.
- Forgetting that QWK rewards ordered predictions differently from accuracy.
