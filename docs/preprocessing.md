# Phase 2: Image Preprocessing

APTOS images vary heavily in crop, illumination, camera type, and field of view. Strong
competition pipelines usually normalize the retinal circle before training.

## Recommended Default

Use Ben Graham preprocessing at `512x512` for iteration:

```bash
python scripts/preprocess_images.py --split train --method ben_graham --image-size 512
python scripts/preprocess_images.py --split test --method ben_graham --image-size 512
```

For final high-resolution training or ensembling, repeat with `768x768` if GPU memory allows.

## 512 vs 768

`512x512`:
- Faster training and larger batch sizes.
- Good baseline for EfficientNet-B4/B5 and ConvNeXt fine-tuning.
- Some small lesions can become harder to see.

`768x768`:
- Preserves microaneurysms, hemorrhages, and exudate detail better.
- Costs substantially more VRAM and training time.
- Works best with gradient accumulation, mixed precision, and smaller batches.

## Common Mistakes

- Applying preprocessing separately per fold with different logic. Preprocessing should be deterministic.
- Cropping too aggressively and cutting off peripheral lesions.
- Treating CLAHE as always better. It can amplify artifacts and camera noise.
- Training on raw images and validating on processed images, or mixing preprocessing methods accidentally.
- Saving JPEG intermediates. Use PNG to avoid compression artifacts.
