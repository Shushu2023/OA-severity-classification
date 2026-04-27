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
- Date        : Pending
- Architecture: EfficientNet-B3
- Loss        : Focal Loss (gamma=2.0) + class weights
- Epochs      : 300 configured / ?? best
- Patience    : 15

### Results
| Metric          | Value |
|-----------------|-------|
| Test Accuracy   | ??    |
| Macro F1        | ??    |
| QWK             | ??    |

### Files
- checkpoints/best_model_efficientnetb3_focalloss_300ep.pth
- reports/evaluation_results_efficientnetb3_focalloss_300ep.json