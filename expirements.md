# Experiment Log — OA Severity Classification

## Experiment 1 — EfficientNet-B3 CrossEntropy 30 epochs
- Date        : April 2026
- Architecture: EfficientNet-B3
- Loss        : Weighted CrossEntropy
- Epochs      : 30 configured / 28 best
- Patience    : 7
- LR Stage 1  : 1e-4
- LR Stage 2  : 1e-5

### Results
| Metric          | Value  |
|-----------------|--------|
| Test Accuracy   | 51.8%  |
| Macro F1        | 0.3734 |
| QWK             | 0.4877 |
| MAE             | ~0.85  |
| Macro AUC       | 0.780  |
| Binary Accuracy | N/A    |

### Files
- checkpoints/best_model_efficientnetb3_crossentropy_30ep.pth
- reports/evaluation_results_efficientnetb3_crossentropy_30ep.json

---

## Experiment 2 — EfficientNet-B3 CrossEntropy 300 epochs
- Date        : April 2026
- Architecture: EfficientNet-B3
- Loss        : Weighted CrossEntropy
- Epochs      : 300 configured / 60 best (early stopping)
- Patience    : 15
- LR Stage 1  : 1e-4
- LR Stage 2  : 1e-5

### Results
| Metric          | Value  |
|-----------------|--------|
| Test Accuracy   | 67.1%  |
| Macro F1        | 0.5127 |
| QWK             | 0.7379 |
| MAE             | 0.3838 |
| Macro AUC       | 0.862  |
| Binary Accuracy | 97.3%  |
| Sensitivity     | 69.6%  |
| NPV             | 98.7%  |

### Files
- checkpoints/best_model_efficientnetb3_crossentropy_300ep.pth
- reports/evaluation_results_efficientnetb3_crossentropy_300ep.json

---

## Experiment 3 — EfficientNet-B3 Focal Loss 300 epochs
- Loss        : FocalLoss gamma=2.0 + class weights
- Best epoch  : 55
- Early stop  : epoch 70

| Metric          | Value  |
|-----------------|--------|
| Test Accuracy   | 62.7%  |
| Macro F1        | 0.5223 |
| QWK             | 0.7056 |
| MAE             | 0.4293 |
| Binary Accuracy | 97.7%  |
| Sensitivity     | 68.4%  |
| PPV             | 72.0%  |
| NPV             | 98.7%  |
## Experiment 4 — EfficientNet-B3 Balanced 1000 CrossEntropy 300ep
- Loss        : CrossEntropyLoss (no weights — balanced data)
- Balanced    : 1,000 samples per grade (oversample minority)
- Best epoch  : 39  (Val F1: 0.5281)
- Early stop  : epoch 54
- Train time  : 59.0 minutes

| Metric          | Value  |
|-----------------|--------|
| Test Accuracy   | 58.9%  |
| Macro F1        | 0.4240 |
| QWK             | 0.6432 |
| MAE             | 0.5056 |
| Macro AUC       | 0.8187 |
| Binary Accuracy | 96.8%  |
| Sensitivity     | 78.5%  ← best of all experiments |
| PPV             | 56.9%  |
| NPV             | 99.1%  ← best of all experiments |
| OA detected     | 62/79  ← best of all experiments |

### Files
- checkpoints/best_model_efficientnetb3_focalloss_300ep.pth
- reports/evaluation_results_efficientnetb3_focalloss_300ep.json