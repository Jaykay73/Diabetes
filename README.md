
# Diabetic Retinopathy Streamlit Demo

## Files needed

Put these files in one folder:

```text
dr_streamlit_app/
├── app.py
├── best_model_phase1 (1).pth
└── requirements.txt
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

## Notes

- The app uses the same EfficientNet-B0 architecture as the training script.
- The model is loaded with `pretrained=False` because you are loading your trained `.pth` weights.
- The expected image size is 384×384.
- The app includes Ben Graham preprocessing and Grad-CAM explainability.
