# Pneumonia Detection from Chest X-Rays: Custom CNN vs. MobileNetV2

This deep-learning project classifies children's chest X-rays as **NORMAL** or **PNEUMONIA**. It compares a
small CNN trained from scratch with an ImageNet-pretrained **MobileNetV2** that is fine-tuned in two
stages. The notebook also includes data exploration, threshold tuning, error analysis by pneumonia type,
and **Grad-CAM** heatmaps that show where the CNN looks.

Everything is in one notebook, [`pneumonia.ipynb`](pneumonia.ipynb), built with TensorFlow/Keras and run
on Google Colab with a T4 GPU.

## Highlights

- **Patient-level split:** train and validation are split by patient with `StratifiedGroupKFold`, using the
  patient ID in each filename, so no patient appears in both sets.
- **Class imbalance:** pneumonia makes up 74% of the training images. Class weights handle this
  (NORMAL 1.93, PNEUMONIA 0.67).
- **Two-stage transfer learning:** first the MobileNetV2 backbone is frozen. Then its top 30 layers are
  fine-tuned with a small learning rate, with BatchNorm layers kept frozen.
- **Threshold tuning:** each model is scored at the default 0.5 threshold and at a threshold tuned on the
  validation set (the highest threshold that still catches at least 95% of pneumonia cases).
- **Explainability:** an error analysis by subtype (normal, bacterial, viral) and Grad-CAM heatmaps.

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
by Kermany et al. The notebook downloads it automatically with `kagglehub`.

| Folder | Normal | Bacterial | Viral | Total |
|--------|-------:|----------:|------:|------:|
| `train/` | 1,341 | 2,530 | 1,345 | 5,216 |
| `val/`   |     8 |     8 |     0 |    16 |
| `test/`  |   234 |   242 |   148 |   624 |

The original `train/` and `val/` folders are merged, then split by patient into **4,182 training images and
1,050 validation images**. `test/` is used only for final evaluation. Image sizes vary widely (median about
1272×888 pixels), so every image is resized to 224×224.

## Models

| Model | Pretrained | Training | Parameters |
|-------|------------|----------|-----------:|
| Custom CNN | none | 4 conv blocks (32→256 filters, BatchNorm, max-pool), global avg pool, Dense 128, Dropout 0.5. Adam 1e-3, up to 25 epochs | 423K |
| MobileNetV2 | ImageNet | **Stage 1:** frozen backbone, new head (global avg pool, Dropout 0.3, sigmoid), Adam 1e-3, 10 epochs. **Stage 2:** top 30 layers unfrozen, Adam 1e-5, up to 10 epochs | ~2.3M |

**Training setup (both models):**
- Loss: binary cross-entropy.
- Augmentation: random rotation, zoom, translation and contrast, during training only.
- Callbacks: `EarlyStopping` (patience 5, restores the best weights), `ReduceLROnPlateau` (factor 0.3,
  patience 2) and `ModelCheckpoint`.

## Results (test set, 624 images)

| Model | Threshold | Accuracy | Precision | Recall (pneumonia) | Specificity (normal) | F1 | ROC-AUC |
|-------|----------:|---------:|----------:|-------------------:|---------------------:|---:|--------:|
| Custom CNN | 0.50 | 0.667 | 0.652 | **1.000** | 0.111 | 0.790 | 0.940 |
| Custom CNN | 0.95 (tuned) | 0.808 | 0.772 | 0.982 | 0.517 | 0.865 | 0.940 |
| MobileNetV2 | 0.50 | 0.837 | 0.796 | 0.992 | 0.577 | 0.884 | **0.971** |
| MobileNetV2 | 0.75 (tuned) | **0.856** | **0.819** | 0.987 | **0.637** | **0.895** | **0.971** |

### Key findings

- **Transfer learning wins on every metric.** MobileNetV2 reaches 0.971 ROC-AUC against 0.940 for the
  custom CNN. At the tuned thresholds it also has higher accuracy (85.6% vs 80.8%) and F1 (0.895 vs 0.865).
- **The threshold matters a lot.** At 0.5 the custom CNN calls almost everything pneumonia (specificity
  0.111, accuracy 66.7%). The tuned threshold raises its accuracy to 80.8%. For MobileNetV2, tuning lifts
  specificity from 0.577 to 0.637.
- **Pneumonia is rarely missed, but normal X-rays are often flagged.** At 0.5, MobileNetV2 catches 98.8% of
  bacterial and 100% of viral pneumonia cases, but labels 42.3% of normal X-rays as pneumonia.
- **MobileNetV2 trains much more steadily.** For the custom CNN's first 5 epochs it predicted pneumonia for
  every validation image (validation AUC 0.50), and early stopping ended its training at epoch 12.
  MobileNetV2 passed 93% validation accuracy after its first epoch.

## How to run

**On Google Colab (easiest):** open `pneumonia.ipynb` in Colab, choose **Runtime → Change runtime type →
T4 GPU**, then click **Run all**. The dataset downloads automatically.

**Locally:**
```bash
git clone https://github.com/vikhyath04/pneumonia-detection.git
cd pneumonia-detection
pip install tensorflow kagglehub scikit-learn pandas numpy matplotlib pillow jupyter
jupyter notebook pneumonia.ipynb
```
If `kagglehub` asks for credentials, run `kagglehub.login()` with your Kaggle account. The first run also
downloads the ImageNet weights for MobileNetV2 (about 9 MB).

On a T4 GPU each epoch takes about 35–45 seconds, so the whole notebook runs in roughly 20 minutes.
Running it saves the trained models as `pneumonia_custom_cnn.keras` and `pneumonia_mobilenetv2.keras`.
These model files are not included in this repo.

## Notebook outline

1. Imports, settings and dataset download
2. Build an image index with subtypes and patient IDs
3. Exploration: images per split and type, sample X-rays, image sizes, brightness by type
4. Patient-level train/validation split and the `tf.data` pipeline
5. Class weights, augmentation, metrics and callbacks
6. Custom CNN: build, train and plot training curves
7. MobileNetV2: frozen training, then fine-tuning
8. Test evaluation: 0.5 vs tuned thresholds, classification reports, confusion matrices, ROC and
   precision-recall curves
9. Error analysis by subtype, and the most confident mistakes
10. Grad-CAM heatmaps for the custom CNN
11. Saving the models

## Limitations

- **Narrow data:** all images are of children and come from a single source.
- **Single run:** results come from one train/validation split and one training run.
- **Threshold tuning:** thresholds are tuned on the validation split, which is relatively small.
- **Grad-CAM scope:** it is shown only for the custom CNN.
- **Not for clinical use:** this is a learning project, not a diagnostic tool.

## Reference

Kermany, D. S. et al. (2018). *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep
Learning.* Cell, 172(5).
