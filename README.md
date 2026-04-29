# Hand OA Severity Classification

Automatic classification of hand osteoarthritis severity from DIP finger joint X-ray images using deep learning. Developed as a final project for CS790 A1 — Computer Vision in AI, Boston University, Spring 2026.

---

## Project overview

This project trains and evaluates deep learning models to classify Kellgren-Lawrence (KL) grades (0–4) from distal interphalangeal (DIP) finger joint X-ray images. The dataset was prepared through a collaboration between Boston University and Tufts Medical Center, using hand X-ray images from the Osteoarthritis Initiative (OAI).

**Clinical motivation:** Manual KL grading is time-consuming and subject to inter-rater variability. An automated system provides standardized, reproducible severity assessments to support clinical screening.

---

## Dataset

- **Source:** Osteoarthritis Initiative (OAI) — digitized by BU and Tufts Medical Center
- **Joints:** DIP2, DIP3, DIP4, DIP5 (4 joints per hand)
- **Images:** 41,060 pre-cropped ROIs at 180×180 pixels
- **Total samples:** 13,200 (after joint selection)
- **Split:** 70% train / 15% val / 15% test (patient-level — no data leakage)
- **Class imbalance:** 49.7× (Grade 0: 6,044 vs Grade 4: 108 training samples)

| KL Grade | Label | Train | Val | Test |
|----------|-------|-------|-----|------|
| Grade 0 | No OA | 6,044 | 1,217 | 1,233 |
| Grade 1 | Doubtful OA | 1,230 | 284 | 260 |
| Grade 2 | Mild OA | 1,759 | 379 | 408 |
| Grade 3 | Moderate OA | 139 | 32 | 44 |
| Grade 4 | Severe OA | 108 | 28 | 35 |

---

## Model architecture

**EfficientNetB3OA** — a two-component model:


Input (3, 180, 180)
    ↓
EfficientNet-B3 backbone (pretrained ImageNet)
    10,696,232 parameters
    ↓
AdaptiveAvgPool2d → (1, 1536)
    ↓
Custom classifier head (MLP)
    Dropout(0.4) → Linear(1536→256) → ReLU → Dropout(0.3) → Linear(256→5)
    394,757 parameters
    ↓
KL Grade prediction (0–4)


**Total parameters:** 11,090,989

**Two-stage training:**
- Stage 1 (epochs 1–5): backbone frozen, head only — LR = 1e-4
- Stage 2 (epoch 6+): full model fine-tuning — LR = 1e-5



## Experiments

Three experiments were conducted using the same EfficientNet-B3 architecture:

| Experiment | Loss Function | Data | Best Epoch | Val F1 |
|------------|---------------|------|------------|--------|
| `efficientnetb3_crossentropy_300ep` | Weighted CrossEntropy | Original | 60 | 0.5208 |
| `efficientnetb3_focalloss_300ep` | Focal Loss (γ=2.0) | Original | 55 | 0.5387 |
| `efficientnetb3_balanced1000_crossentropy_300ep` | CrossEntropy | Balanced (1000/class) | 39 | 0.5281 |

### Test results comparison

| Metric | CrossEntropy | Focal Loss | Balanced |
|--------|-------------|------------|----------|
| Test Accuracy | **67.1%** | 62.7% | 58.9% |
| Macro F1 | 0.5127 | **0.5223** | 0.4240 |
| QWK | **0.7379** | 0.7056 | 0.6432 |
| MAE | **0.3838** | 0.4293 | 0.5056 |
| Macro AUC | **0.8620** | 0.8620 | 0.8187 |
| Binary Accuracy | 97.3% | **97.7%** | 96.8% |
| Sensitivity | 69.6% | 68.4% | **78.5%** |
| Specificity | 98.5% | **98.9%** | 97.5% |
| PPV | 65.5% | **72.0%** | 56.9% |
| NPV | 98.7% | 98.7% | **99.1%** |

---

## Repository structure

OA-severity-classification/
├── train.py                         # Training pipeline
├── evaluate.py                      # Evaluation and metrics
├── losses.py                        # FocalLoss implementation
├── expirements.md                   # Experiment log
│
├── models/
│   └── efficientnet.py              # EfficientNetB3OA model class
│
├── data_pipeline/
│   ├── preprocessing.py             # OADataset, get_dataloaders
│   ├── splitting.py                 # Patient-level data splitting
│   └── balanced_sampler.py         # Balanced oversampling
│
├── data/splits/
│   ├── train.csv                    # Training split
│   ├── val.csv                      # Validation split
│   ├── test.csv                     # Test split
│   ├── master_dataset.csv           # Full dataset metadata
│   └── balanced/
│       ├── train_balanced.csv       # Balanced training split (1000/class)
│       ├── val_balanced.csv         # Balanced validation split
│       └── test_balanced.csv        # Test split (unchanged)
│
└── reports/
    ├── training_history_*.json      # Per-epoch metrics
    ├── test_results_*.json          # Test set results
    ├── evaluation_results_*.json    # Full evaluation metrics
    ├── training_curves_*.png        # Loss, accuracy, F1 curves
    ├── confusion_matrix_*.png       # Confusion matrices
    ├── per_grade_metrics_*.png      # Per-grade precision/recall/F1
    ├── roc_curves_*.png             # ROC curves and AUC
    ├── gradcam_heatmaps_*.png       # Grad-CAM interpretability
    └── binary_group_confusion_matrix_*.png

---


## Setup and usage

### Requirements

bash
pip install torch torchvision timm scikit-learn pandas matplotlib pillow grad-cam


### Running on Google Colab

python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone repo and install
import os
!git clone https://github.com/Shushu2023/OA-severity-classification.git
os.chdir('/content/OA-severity-classification')
!pip install timm -q

# Set environment variables
BASE_DIR = '/.../OA-severity-classification'
os.environ['OA_BASE_DIR']        = BASE_DIR
os.environ['OA_SPLITS_DIR']      = os.path.join(BASE_DIR, 'data', 'splits')
os.environ['OA_CHECKPOINTS_DIR'] = os.path.join(BASE_DIR, 'checkpoints')
os.environ['OA_REPORTS_DIR']     = os.path.join(BASE_DIR, 'reports')

# Run training
!python train.py


### Switching experiments

Change only one line in `train.py` and `evaluate.py`:

python
# Weighted CrossEntropy (original data)
EXPERIMENT_NAME = 'efficientnetb3_crossentropy_300ep'

# Focal Loss (original data)
EXPERIMENT_NAME = 'efficientnetb3_focalloss_300ep'

# Balanced sampling (1000 per class)
EXPERIMENT_NAME = 'efficientnetb3_balanced1000_crossentropy_300ep'


The correct data splits and loss function are selected automatically based on the experiment name.

### Running evaluation

bash
python evaluate.py


Generates all plots and saves full evaluation results to `reports/`.

---

## Training configuration

| Hyperparameter | Value |
|----------------|-------|
| Input shape | (32, 3, 180, 180) |
| Epochs configured | 300 |
| Early stopping patience | 15 |
| Batch size | 32 |
| Optimizer | AdamW |
| Weight decay | 1e-4 |
| Stage 1 LR | 1e-4 |
| Stage 2 LR | 1e-5 |
| LR scheduler | CosineAnnealingLR |
| Min LR | 1e-6 |
| Head dropout 1 | p=0.4 |
| Head dropout 2 | p=0.3 |
| Focal loss gamma | 2.0 |
| Balanced samples/class | 1,000 |

---

## Hardware

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA Tesla T4 (15.6 GB VRAM) |
| Platform | Google Colab |
| Framework | PyTorch 2.10.0+cu128 |
| Training time | ~60–120 minutes per experiment |

---

## Key metrics explained

- **QWK (Quadratic Weighted Kappa):** Measures ordinal agreement — accounts for how far off each prediction is. Range −1 to 1. Values above 0.61 indicate substantial agreement.
- **MAE (Mean Absolute Error):** Average KL grade distance between prediction and truth. Lower is better.
- **Sensitivity:** Percentage of true OA cases correctly detected. Critical for clinical safety.
- **NPV (Negative Predictive Value):** When model predicts Non-OA, how often it is correct. High NPV means safe to clear patients.
- **Macro AUC:** Average AUC across all 5 grades using One-vs-Rest approach.

---

## References

- Tan, M. and Le, Q.V. (2019). EfficientNet: Rethinking Model Scaling for CNNs. ICML. arXiv:1905.11946
- Lin, T.Y. et al. (2017). Focal Loss for Dense Object Detection. ICCV. arXiv:1708.02002
- Loshchilov, I. and Hutter, F. (2019). Decoupled Weight Decay Regularization. ICLR. arXiv:1711.05101
- Raghu, M. et al. (2019). Transfusion: Understanding Transfer Learning for Medical Imaging. NeurIPS. arXiv:1902.07208
- Osteoarthritis Initiative (OAI). National Institutes of Health. https://nda.nih.gov/oai
- World Health Organization (2023). Osteoarthritis Fact Sheet. https://www.who.int/news-room/fact-sheets/detail/osteoarthritis

---

## Team

- Shaima Nimeri
- Ahuva Friedman
