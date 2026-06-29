# 🫁 Pneumonia Detection from Chest X-Ray Images

A deep learning project for binary classification of chest X-ray images into **Normal** and **Pneumonia** categories. The project implements and compares two transfer learning architectures — **VGG16** and **DenseNet121** — using a two-stage fine-tuning strategy built on TensorFlow / Keras.

---

## 📁 Project Structure

```
CV_Pneumonia_Detector/
│
├── dataset/                        # Dataset root (git-ignored, see below)
│   ├── data_ori/                   # Original unprocessed dataset
│   │   ├── train/
│   │   │   ├── NORMAL/
│   │   │   └── PNEUMONIA/
│   │   ├── val/
│   │   └── test/
│   │       ├── NORMAL/
│   │       └── PNEUMONIA/
│   └── data_stratified/            # Stratified re-split dataset (used for training)
│       ├── train/
│       │   ├── NORMAL/
│       │   └── PNEUMONIA/
│       ├── val/
│       │   ├── NORMAL/
│       │   └── PNEUMONIA/
│       └── test/
│           ├── NORMAL/
│           └── PNEUMONIA/
│
├── densenet121/                    # DenseNet121 model module
│   ├── densenet121.py              # Training script (two-stage fine-tuning)
│   ├── densenet121_test.py         # Evaluation & confusion matrix generation
│   ├── densenet121_best.keras      # Saved best model weights (git-ignored)
│   └── confusion_matrix_densenet121.png
│
├── vgg16/                          # VGG16 model module
│   ├── vgg16.py                    # Training script (two-stage fine-tuning)
│   ├── vgg16_test.py               # Evaluation & confusion matrix generation
│   ├── vgg16_best.keras            # Saved best model weights (git-ignored)
│   └── confusion_matrix_vgg16.png
│
├── manipulation/                   # Data preparation utilities
│   └── kocok.py                    # Stratified shuffle & re-split script
│
├── requirements.txt                # Python dependencies
├── .gitignore                      # Ignores dataset files & Keras model weights
└── README.md                       # This file
```

---

## 🧠 Model Architectures

Both models follow an identical high-level design, differing only in the backbone CNN:

| Component | Details |
|---|---|
| **Backbone** | VGG16 / DenseNet121 (ImageNet pre-trained) |
| **Pooling** | Global Average Pooling 2D |
| **Regularisation** | Dropout (0.5) |
| **Classifier Head** | Dense(1, sigmoid) — binary output |
| **Input Size** | 224 × 224 × 3 |

### Data Augmentation (On-the-Fly)

Applied during training only. Horizontal flip is intentionally **excluded** to preserve anatomical heart position in chest X-rays.

| Augmentation | Parameters |
|---|---|
| Random Zoom | ±5 % height & width |
| Random Rotation | ±5 % (≈ ±18°) |
| Random Brightness | ±10 % |
| Random Contrast | ±10 % |
| Gaussian Noise | σ = 0.1 |

---

## 🏋️ Training Strategy

A **two-stage fine-tuning** approach is used for both models:

### Stage 1 — Feature Extraction
- The backbone is **frozen** (weights not updated).
- Only the custom classifier head is trained.
- Optimizer: Adam (lr = `1e-3`)
- Epochs: up to **30** (with early stopping)

### Stage 2 — Fine-Tuning
- The backbone is **unfrozen** for end-to-end training.
  - *DenseNet121*: BatchNormalization layers remain frozen to preserve running statistics.
  - *VGG16*: All layers are unfrozen (no BatchNorm layers present).
- Optimizer: Adam (lr = `1e-5`)
- Epochs: up to **50** (with early stopping)

### Callbacks

| Callback | Configuration |
|---|---|
| **EarlyStopping** | Monitors `val_loss`, patience = 10, restores best weights |
| **ModelCheckpoint** | Saves best model to `*_best.keras` |
| **ReduceLROnPlateau** | Halves LR after 5 epochs without improvement (min LR = `1e-6`) |
| **CSVLogger** | Logs training metrics per stage to CSV |

### Class Imbalance Handling

Dynamic **class weights** are computed from the actual training label distribution to counteract any class imbalance:

```
weight_class_i = total_samples / (2 × count_class_i)
```

---

## 📊 Evaluation

Each model's test script (`*_test.py`) performs the following:

1. Loads the saved best model (`*_best.keras`).
2. Runs `model.evaluate()` on the held-out test set to report loss, accuracy, precision, and recall.
3. Generates per-sample predictions and produces:
   - A full **classification report** (precision, recall, F1-score per class).
   - A **confusion matrix** heatmap saved as a PNG image.

---

## 🔀 Data Preparation — Stratified Split

The script `manipulation/kocok.py` handles dataset preparation:

1. **Collects** all images from the original `train/` and `val/` folders.
2. **Shuffles** them with a fixed seed (`42`) for reproducibility.
3. **Re-splits** them into new `train` (85 %) and `val` (15 %) sets using **stratified sampling** (per-class ratio is preserved).
4. Outputs the result into `dataset/data_stratified/`.

> **Note:** The original `test/` set is kept untouched and used separately for final evaluation.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- A CUDA-capable GPU is recommended (scripts default to batch size 32, suitable for ≥ 12 GB VRAM).

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the Dataset

Place your chest X-ray dataset inside `dataset/data_ori/` following this structure:

```
dataset/data_ori/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

Then run the stratified re-split:

```bash
python manipulation/kocok.py
```

### 3. Train a Model

```bash
# Train DenseNet121
cd densenet121
python densenet121.py

# Train VGG16
cd vgg16
python vgg16.py
```

### 4. Evaluate on Test Set

```bash
# Evaluate DenseNet121
cd densenet121
python densenet121_test.py

# Evaluate VGG16
cd vgg16
python vgg16_test.py
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `tensorflow` | Deep learning framework (includes Keras) |
| `numpy` | Numerical operations |
| `scikit-learn` | Classification report & confusion matrix |
| `matplotlib` | Plotting |
| `seaborn` | Confusion matrix heatmap styling |
| `opencv-python` | Image processing utilities |
| `albumentations` | Advanced image augmentation library |
| `pandas` | Data manipulation |

---

## 📝 Notes

- **Model weights** (`.keras`, `.h5`, `.hdf5`) and **dataset images** are excluded from version control via `.gitignore`. Only directory structure placeholders (`.gitkeep`) are tracked.
- The random seed is locked to `42` across all scripts to ensure **reproducibility**.
- Both models use `label_mode='binary'` and a sigmoid output, framing the task as binary classification (Normal vs. Pneumonia).
