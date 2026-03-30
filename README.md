# Chest X-Ray Pneumonia Classification using CNN

## Overview

This project builds a custom Convolutional Neural Network (CNN) from scratch to classify chest X-ray images of paediatric patients (ages 1–5) as either **NORMAL** or **PNEUMONIA**. The CNN is designed, implemented, and trained entirely without any pretrained weights or external backbones. The dataset contains 5,863 JPEG images organised into train, val, and test splits.

---

## Project Structure

```
chest_xray_cnn/
├── data/
│   └── chest_xray/
│       ├── train/
│       │   ├── NORMAL/
│       │   └── PNEUMONIA/
│       ├── val/
│       │   ├── NORMAL/
│       │   └── PNEUMONIA/
│       └── test/
│           ├── NORMAL/
│           └── PNEUMONIA/
├── models/
│   └── best_model.pth
├── outputs/
│   ├── training_history.png
│   ├── confusion_matrix.png
│   ├── results.json
│   └── gradcam_*.png
├── train.py
├── predict.py
├── gradcam.py
├── requirements.txt
└── README.md
```

---

## CNN Architecture

The model `PneumoniaCNN` is a custom CNN built from scratch using PyTorch. It consists of four convolutional blocks followed by a global average pooling layer and a three-layer fully connected classifier.

```
Input Image (3 x 128 x 128)
        |
[Conv Block 1]
  Conv2d(3 -> 32, 3x3) + BatchNorm + ReLU
  Conv2d(32 -> 32, 3x3) + BatchNorm + ReLU
  MaxPool2d(2x2) + Dropout2d(0.1)
  Output: 32 x 64 x 64
        |
[Conv Block 2]
  Conv2d(32 -> 64, 3x3) + BatchNorm + ReLU
  Conv2d(64 -> 64, 3x3) + BatchNorm + ReLU
  MaxPool2d(2x2) + Dropout2d(0.2)
  Output: 64 x 32 x 32
        |
[Conv Block 3]
  Conv2d(64 -> 128, 3x3) + BatchNorm + ReLU
  Conv2d(128 -> 128, 3x3) + BatchNorm + ReLU
  MaxPool2d(2x2) + Dropout2d(0.3)
  Output: 128 x 16 x 16
        |
[Conv Block 4]
  Conv2d(128 -> 256, 3x3) + BatchNorm + ReLU
  Conv2d(256 -> 256, 3x3) + BatchNorm + ReLU
  MaxPool2d(2x2) + Dropout2d(0.3)
  Output: 256 x 8 x 8
        |
[Global Average Pool]
  AdaptiveAvgPool2d(4x4)
  Output: 256 x 4 x 4 = 4096 features
        |
[Fully Connected Classifier]
  Linear(4096 -> 512) + BatchNorm + ReLU + Dropout(0.5)
  Linear(512 -> 128)  + BatchNorm + ReLU + Dropout(0.4)
  Linear(128 -> 2)
        |
Output: NORMAL or PNEUMONIA
```

### Why each component is used

**Conv2d layers** are the core of the CNN. They apply learnable filters across the input image to detect local patterns such as edges, textures, and shapes. The filter count doubles at each block (32→64→128→256) so the network learns increasingly abstract features early blocks detect simple edges and intensity gradients, later blocks detect complex opacity patterns characteristic of pneumonia.

**Two Conv layers per block** allows deeper feature extraction before spatial resolution is reduced by pooling. This is inspired by the VGG design principle and gives the network more capacity to learn complex patterns at each spatial scale.

**BatchNorm2d after every Conv** normalises the activations within each mini-batch. This stabilises training, allows higher learning rates, and acts as a mild regulariser, especially important when training from scratch on a relatively small medical dataset.

**MaxPool2d** halves the spatial dimensions at each block, reducing computation and making the learned features more translation-invariant.

**Progressive Dropout2d** (0.1 → 0.2 → 0.3) applies spatial dropout on feature maps. Unlike standard dropout, Dropout2d drops entire feature channels, forcing the network to learn redundant representations and reducing overfitting.

**AdaptiveAvgPool2d** replaces a raw flatten operation. It averages each feature map spatially before the classifier, significantly reducing the parameter count and improving generalisation.

**Kaiming He initialisation** is applied to all Conv and Linear layers. This initialisation strategy is mathematically derived for ReLU networks and prevents vanishing or exploding gradients at the start of training from scratch.

---

## Methodology

### 1. Data Preprocessing and Augmentation

All images are resized to 128×128 pixels and normalised using mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.

Training augmentations are applied only to the train split to increase effective dataset size and improve generalisation:

- Random horizontal flip — mimics natural variation in patient positioning
- Random rotation up to ±15° — accounts for slight tilt variation in X-ray capture
- Colour jitter (brightness and contrast ±0.2) — simulates exposure variation between different X-ray machines
- Random affine translation (±5%) — minor positional shifts

These augmentations are conservative and clinically plausible, avoiding distortions that would not occur in real X-rays such as vertical flips or extreme zoom.

### 2. Class Imbalance Handling

The dataset is imbalanced: the training set contains 3,875 PNEUMONIA samples versus only 1,341 NORMAL samples. Two mechanisms address this:

**Weighted Cross-Entropy Loss** — class weights are computed as the inverse of class frequency from the training set and passed to `nn.CrossEntropyLoss`. This penalises misclassification of the minority (NORMAL) class more heavily during training.

**Data augmentation** — applying transforms to the training set effectively increases the diversity of minority class examples seen during each epoch.

### 3. Optimiser and Learning Rate Schedule

**Adam optimiser** with learning rate `1e-3` and weight decay `1e-4`. Adam is well-suited for training CNNs from scratch as it adapts per-parameter learning rates and converges faster than SGD on small-to-medium datasets.

**Cosine Annealing LR Scheduler** decays the learning rate smoothly from `1e-3` to `1e-6` over 40 epochs. This avoids abrupt drops that can destabilise training and allows the model to settle into a sharp, well-generalising minimum in the later epochs.

**Gradient clipping** at `max_norm=1.0` prevents exploding gradients, which can occur during the early epochs of training a deep network from random initialisation.

### 4. Mixed Precision Training (AMP)

`torch.amp.autocast` and `GradScaler` enable automatic mixed precision training. On the RTX 3060 with Tensor Cores this reduces VRAM usage, enables larger batch sizes, and speeds up training by 30–50% compared to full FP32, while maintaining numerical stability via dynamic loss scaling.

### 5. Model Selection

The best model checkpoint is saved to `models/best_model.pth` whenever validation accuracy improves. At the end of training this best checkpoint is reloaded for final evaluation on the held-out test set, ensuring reported results reflect peak performance rather than the final epoch.

### 6. Evaluation

- **Accuracy** on the 624-image held-out test set
- **Classification Report** — precision, recall, and F1-score per class via scikit-learn
- **Confusion Matrix** — visualised with seaborn to show false positives and false negatives
- **Grad-CAM** — gradient-weighted class activation maps highlight which regions of the X-ray the CNN uses to make its prediction

---

## GPU Setup (RTX 3060 12 GB)

Check your CUDA version:

```bash
nvidia-smi
```

Install PyTorch with CUDA 12.4 support (compatible with CUDA 12.x drivers):

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Install remaining dependencies:

```bash
pip install scikit-learn matplotlib seaborn numpy Pillow
```

Verify GPU is detected:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected output: `True  NVIDIA GeForce RTX 3060`

---

## Dataset Setup

1. Download the dataset from [Kaggle: Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
2. Extract and place it under `data/chest_xray/` matching the project structure above

---

## Usage

### Train the Model

```bash
python train.py
```

Training will:
- Print per-epoch loss and accuracy for train and val splits
- Save the best model checkpoint to `models/best_model.pth`
- Save loss and accuracy plots to `outputs/training_history.png`
- Save a confusion matrix to `outputs/confusion_matrix.png`
- Print the full classification report on the test set
- Save a results summary to `outputs/results.json`

### Run Inference

Single image:
```bash
python predict.py --image path/to/xray.jpg
```

Entire folder:
```bash
python predict.py --directory data/chest_xray/test/PNEUMONIA/
```

### Grad-CAM Visualisation

```bash
python gradcam.py --image path/to/xray.jpg
```

Generates a 3-panel figure — original image, heatmap, and overlay — saved to `outputs/gradcam_<filename>.png`. The heatmap highlights which regions of the lung the CNN focused on when making its prediction.

---

## Findings

### Results on Test Set (624 images)

| Metric | Value |
|---|---|
| Test Accuracy | 90.38% |
| Pneumonia Precision | 88.2% |
| Pneumonia Recall | 97.7% |
| Normal Precision | 95.3% |
| Normal Recall | 78.2% |
| Best Epoch | 20 of 40 |

### Confusion Matrix

| | Predicted NORMAL | Predicted PNEUMONIA |
|---|---|---|
| True NORMAL | 183 | 51 |
| True PNEUMONIA | 9 | 381 |

### Key Observations

**Pneumonia recall of 97.7%** means the CNN missed only 9 pneumonia cases out of 390. This is the most clinically important metric — a false negative (missed pneumonia) is more dangerous than a false positive (unnecessary follow-up). The weighted loss function directly drives this behaviour.

**Normal recall of 78.2%** means 51 healthy patients were flagged as pneumonia. This is the trade-off from prioritising pneumonia sensitivity. In a real screening pipeline false positives lead to further review rather than missed treatment, making this an acceptable trade-off.

**Class imbalance impact** — the training set contains nearly 3× more PNEUMONIA samples than NORMAL. Without weighted loss the model would over-predict PNEUMONIA and achieve inflated accuracy while missing NORMAL cases. Weighted Cross-Entropy directly corrects this bias.

**Validation curve volatility** — the validation set contains only 16 images, so one misclassified image changes validation accuracy by 6.25%. The test set result of 90.38% on 624 images is the reliable performance indicator.

**Grad-CAM interpretability** — saliency maps show the CNN focuses on the lung parenchyma, particularly the middle and lower lung fields, when predicting PNEUMONIA. This is consistent with the consolidation and infiltrate patterns a radiologist would examine. NORMAL predictions show more diffuse, less localised activation.

### Recommendations for Clinical Deployment

This model is a screening aid only and must not replace radiologist review. Deployment considerations:
- Calibrate the classification threshold based on the desired sensitivity and specificity trade-off for the target clinical setting
- Monitor for distribution shift if deploying outside the paediatric age range or with different X-ray equipment
- Log and audit all predictions for ongoing performance tracking

---

## Configuration

All hyperparameters are defined at the top of `train.py`:

| Parameter | Value | Notes |
|---|---|---|
| `IMAGE_SIZE` | 128 | Input resolution for the CNN |
| `BATCH_SIZE` | 32 | Fits comfortably in 12 GB VRAM with AMP |
| `NUM_EPOCHS` | 40 | More epochs needed when training from scratch |
| `LEARNING_RATE` | 1e-3 | Initial LR, decayed via cosine annealing |
| `NUM_CLASSES` | 2 | NORMAL / PNEUMONIA |

---

## Dependencies

| Package | Purpose |
|---|---|
| torch / torchvision | CNN implementation, training, data loading, transforms |
| scikit-learn | Classification report, confusion matrix |
| matplotlib / seaborn | Training plots and visualisations |
| numpy | Numerical operations |
| Pillow | Image loading and preprocessing |
