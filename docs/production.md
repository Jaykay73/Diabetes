# Production Stages

## FastAPI

Run locally:

```bash
export DR_MODEL_CHECKPOINT=models/checkpoints/model.pth
export DR_MODEL_ARCH=tf_efficientnet_b4_ns
export DR_MODEL_TASK=regression
uvicorn api.main:app --reload
```

Endpoints:

- `GET /health`
- `POST /predict`
- `POST /predict/file`
- `POST /explain`
- `GET /explain/methods`

## Streamlit

```bash
export DR_API_URL=http://localhost:8000
streamlit run streamlit_app/app.py
```

## Docker

```bash
docker compose config
docker compose build
docker compose up
```

## Deployment Notes

- Keep preprocessing identical between training and serving.
- Save thresholds and model config next to checkpoints.
- Prefer CPU inference with EfficientNet-B0/B3/B4 or GPU inference for larger ensembles.
- Do not present model output as a clinical diagnosis without validation and regulatory review.
