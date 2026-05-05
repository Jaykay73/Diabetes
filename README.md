# Diabetic Retinopathy Grading

Production-grade PyTorch pipeline for APTOS 2019 Blindness Detection diabetic retinopathy grading.

## Phase 1 Structure

```text
.
├── api/
├── configs/
│   └── config.yaml
├── data/
│   ├── external/
│   ├── interim/
│   ├── processed/
│   └── raw/
│       └── aptos2019/
├── models/
│   └── checkpoints/
├── notebooks/
├── reports/
│   ├── figures/
│   └── quality/
├── scripts/
│   └── run_eda.py
├── src/
│   └── dr_grading/
│       ├── config.py
│       ├── data/
│       │   ├── eda.py
│       │   └── quality.py
│       └── utils/
│           └── logging.py
├── streamlit_app/
└── tests/
```

Expected raw APTOS layout:

```text
data/raw/aptos2019/
├── train.csv
├── test.csv
├── train_images/
│   └── <id_code>.png
└── test_images/
    └── <id_code>.png
```

Run Phase 1 EDA:

```bash
python scripts/run_eda.py --config configs/config.yaml
```

Outputs are written to `reports/figures/` and `reports/quality/`.
