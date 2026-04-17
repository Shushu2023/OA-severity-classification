import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
from sklearn.metrics import (f1_score, confusion_matrix,
                             classification_report,
                             cohen_kappa_score,
                             mean_absolute_error)
from PIL import Image
import torchvision.transforms as T

# ── Import project modules ────────────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_pipeline.preprocessing import get_dataloaders, N_CLASSES, BATCH_SIZE
from models.efficientnet import EfficientNetB3OA

# ── Constants ─────────────────────────────────────────────────────────────────
GRADE_NAMES  = ['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4']
GRADE_COLORS = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']


def get_paths():
    """Auto-detect paths — supports laptop, Colab, and SCC."""
    if os.environ.get('OA_BASE_DIR'):
        BASE_DIR        = os.environ['OA_BASE_DIR']
        SPLITS_DIR      = os.environ.get('OA_SPLITS_DIR',
                          os.path.join(BASE_DIR, 'data', 'splits'))
        CHECKPOINTS_DIR = os.environ.get('OA_CHECKPOINTS_DIR',
                          os.path.join(BASE_DIR, 'checkpoints'))
        REPORTS_DIR     = os.environ.get('OA_REPORTS_DIR',
                          os.path.join(BASE_DIR, 'reports'))
    else:
        BASE_DIR        = os.path.abspath(os.path.dirname(__file__))
        SPLITS_DIR      = os.path.join(BASE_DIR, 'data', 'splits')
        CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'checkpoints')
        REPORTS_DIR     = os.path.join(BASE_DIR, 'reports')

    print(f"BASE_DIR        : {BASE_DIR}")
    print(f"CHECKPOINTS_DIR : {CHECKPOINTS_DIR}")
    print(f"REPORTS_DIR     : {REPORTS_DIR}")

    return BASE_DIR, SPLITS_DIR, CHECKPOINTS_DIR, REPORTS_DIR


def load_model(checkpoint_path, device, n_classes=N_CLASSES):
    """Load trained model from checkpoint."""
    model = EfficientNetB3OA(n_classes=n_classes, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"Model loaded from epoch {checkpoint['epoch']} "
          f"(Val F1: {checkpoint['val_f1']:.4f})")
    return model


def plot_training_curves(history_path, reports_dir):
    """
    Plot training and validation loss, accuracy and F1
    across all epochs from training_history.json
    """
    with open(history_path, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── Loss ──────────────────────────────────────────────────────────────
    axes[0].plot(epochs, history['train_loss'],
                 'b-o', markersize=4, label='Train')
    axes[0].plot(epochs, history['val_loss'],
                 'r-o', markersize=4, label='Val')
    axes[0].axvline(x=5, color='gray', linestyle='--',
                    alpha=0.7, label='Stage 2 start')
    axes[0].set_title('Loss', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ── Accuracy ───────────────────────────────────────────────────────────
    axes[1].plot(epochs, history['train_acc'],
                 'b-o', markersize=4, label='Train')
    axes[1].plot(epochs, history['val_acc'],
                 'r-o', markersize=4, label='Val')
    axes[1].axvline(x=5, color='gray', linestyle='--',
                    alpha=0.7, label='Stage 2 start')
    axes[1].set_title('Accuracy', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # ── Macro F1 ───────────────────────────────────────────────────────────
    axes[2].plot(epochs, history['train_f1'],
                 'b-o', markersize=4, label='Train')
    axes[2].plot(epochs, history['val_f1'],
                 'r-o', markersize=4, label='Val')
    axes[2].axvline(x=5, color='gray', linestyle='--',
                    alpha=0.7, label='Stage 2 start')
    axes[2].set_title('Macro F1 Score', fontsize=13, fontweight='bold')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1 Score')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Training Curves — EfficientNet-B3 OA Classification',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    save_path = os.path.join(reports_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Training curves saved to {save_path}")


def plot_confusion_matrix(test_labels, test_preds, reports_dir):
    """Plot normalized and raw confusion matrix."""
    cm      = confusion_matrix(test_labels, test_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ['Confusion Matrix (Raw Counts)', 'Confusion Matrix (Normalized)'],
        ['d', '.2f']
    ):
        im = ax.imshow(data, interpolation='nearest', cmap='Blues')
        plt.colorbar(im, ax=ax)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Grade', fontsize=11)
        ax.set_ylabel('True Grade', fontsize=11)
        ax.set_xticks(range(N_CLASSES))
        ax.set_yticks(range(N_CLASSES))
        ax.set_xticklabels(GRADE_NAMES, rotation=45, ha='right')
        ax.set_yticklabels(GRADE_NAMES)

        thresh = data.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(data[i, j], fmt),
                        ha='center', va='center', fontsize=10,
                        color='white' if data[i, j] > thresh else 'black')

    plt.tight_layout()
    save_path = os.path.join(reports_dir, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Confusion matrix saved to {save_path}")


def plot_per_grade_metrics(test_labels, test_preds, reports_dir):
    """Plot precision, recall and F1 per grade."""
    report = classification_report(
        test_labels, test_preds,
        target_names=GRADE_NAMES,
        output_dict=True,
        zero_division=0
    )

    metrics    = ['precision', 'recall', 'f1-score']
    x          = np.arange(N_CLASSES)
    width      = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#2196F3', '#4CAF50', '#FF9800']
    for i, metric in enumerate(metrics):
        values = [report[grade][metric] for grade in GRADE_NAMES]
        bars   = ax.bar(x + i * width, values, width,
                        label=metric.capitalize(),
                        color=colors[i], edgecolor='white', alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

    ax.set_xlabel('KL Grade', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Precision, Recall and F1 per KL Grade',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(GRADE_NAMES)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = os.path.join(reports_dir, 'per_grade_metrics.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Per grade metrics saved to {save_path}")


def plot_roc_curves(test_labels, test_probs, reports_dir):
    """
    Plot ROC curves for each KL grade using One-vs-Rest approach.
    Also plots macro average AUC.
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    import numpy as np

    # Binarize labels for one-vs-rest
    test_labels_bin = label_binarize(test_labels, classes=[0, 1, 2, 3, 4])
    test_probs_arr  = np.array(test_probs)

    colors     = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']
    grade_names = ['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4']

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ── Left plot: all 5 grades individually ─────────────────────────────
    fpr_all = {}
    tpr_all = {}
    auc_all = {}

    for i in range(N_CLASSES):
        fpr, tpr, _ = roc_curve(test_labels_bin[:, i], test_probs_arr[:, i])
        roc_auc     = auc(fpr, tpr)
        fpr_all[i]  = fpr
        tpr_all[i]  = tpr
        auc_all[i]  = roc_auc

        axes[0].plot(fpr, tpr, color=colors[i], linewidth=2,
                     label=f'{grade_names[i]} (AUC = {roc_auc:.3f})')

    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0].set_title('ROC Curves per KL Grade (One-vs-Rest)',
                       fontsize=13, fontweight='bold')
    axes[0].legend(loc='lower right', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])

    # ── Right plot: macro average ─────────────────────────────────────────
    # Compute macro average ROC
    all_fpr = np.unique(np.concatenate([fpr_all[i] for i in range(N_CLASSES)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(N_CLASSES):
        mean_tpr += np.interp(all_fpr, fpr_all[i], tpr_all[i])
    mean_tpr  /= N_CLASSES
    macro_auc  = auc(all_fpr, mean_tpr)

    axes[1].plot(all_fpr, mean_tpr, color='#2E75B6',
                 linewidth=3, label=f'Macro average (AUC = {macro_auc:.3f})')

    for i in range(N_CLASSES):
        axes[1].plot(fpr_all[i], tpr_all[i], color=colors[i],
                     linewidth=1.5, alpha=0.5,
                     label=f'{grade_names[i]} (AUC = {auc_all[i]:.3f})')

    axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')
    axes[1].set_xlabel('False Positive Rate', fontsize=12)
    axes[1].set_ylabel('True Positive Rate', fontsize=12)
    axes[1].set_title('Macro Average ROC Curve',
                       fontsize=13, fontweight='bold')
    axes[1].legend(loc='lower right', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])

    plt.suptitle('ROC Curves — EfficientNet-B3 OA KL Grade Classification',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    save_path = os.path.join(reports_dir, 'roc_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"ROC curves saved to {save_path}")

    # Print AUC summary
    print(f"\n── AUC Summary ─────────────────────────────────────")
    for i in range(N_CLASSES):
        print(f"  {grade_names[i]} : AUC = {auc_all[i]:.4f}")
    print(f"  Macro avg  : AUC = {macro_auc:.4f}")

    return auc_all, macro_auc

def compute_quadratic_weighted_kappa(test_labels, test_preds):
    """
    Compute Quadratic Weighted Kappa (QWK).
    Standard metric for ordinal classification tasks like KL grading.
    Higher is better. Range: -1 to 1.
    """
    qwk = cohen_kappa_score(test_labels, test_preds, weights='quadratic')
    print(f"Quadratic Weighted Kappa : {qwk:.4f}")
    return qwk

#binary classification grouping
#combining the 5 kl grades into two clinical groups
#Non-OA group: Grade 0, Grade 1, Grad 2-->1,901 test samples
#OA group     : Grade 3, Grad 4   --> 79 test samples
def evaluate_binary_groups(test_labels, test_preds, reports_dir):
    """
    Evaluate model using clinical binary grouping:
    Non-OA group : Grade 0, 1, 2
    OA group     : Grade 3, 4
    """
    import numpy as np
    from sklearn.metrics import (confusion_matrix, classification_report,
                                  roc_curve, auc)

    # Convert 5-class labels to binary
    # 0 = Non-OA (grades 0,1,2)  1 = OA (grades 3,4)
    binary_true  = [0 if label <= 2 else 1 for label in test_labels]
    binary_preds = [0 if pred  <= 2 else 1 for pred  in test_preds]

    # Calculate binary metrics
    binary_labels_arr = np.array(binary_true)
    binary_preds_arr  = np.array(binary_preds)

    correct   = np.sum(binary_labels_arr == binary_preds_arr)
    total     = len(binary_labels_arr)
    accuracy  = correct / total

    # Confusion matrix
    cm = confusion_matrix(binary_true, binary_preds)

    # Per group counts
    non_oa_total = sum(1 for l in binary_true if l == 0)
    oa_total     = sum(1 for l in binary_true if l == 1)
    non_oa_correct = cm[0][0] if cm.shape[0] > 1 else cm[0][0]
    oa_correct     = cm[1][1] if cm.shape[0] > 1 else 0

    print(f"\n── Binary Group Evaluation ─────────────────────────")
    print(f"  Non-OA group (Grade 0,1,2) : {non_oa_total} samples")
    print(f"  OA group     (Grade 3,4)   : {oa_total} samples")
    print(f"\n  Binary Accuracy            : {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"\n  Non-OA correctly identified: {non_oa_correct}/{non_oa_total} "
          f"({non_oa_correct/non_oa_total*100:.1f}%)")
    print(f"  OA correctly identified    : {oa_correct}/{oa_total} "
          f"({oa_correct/oa_total*100:.1f}%)")

    print(f"\n── Binary Classification Report ────────────────────")
    print(classification_report(
        binary_true, binary_preds,
        target_names=['Non-OA (Grade 0,1,2)', 'OA (Grade 3,4)'],
        zero_division=0
    ))

    print(f"── Binary Confusion Matrix ─────────────────────────")
    print(f"{'':25} {'Pred Non-OA':>12} {'Pred OA':>10}")
    print(f"{'True Non-OA':25} {cm[0][0]:>12} {cm[0][1]:>10}")
    print(f"{'True OA':25} {cm[1][0]:>12} {cm[1][1]:>10}")

    # Plot binary confusion matrix
    fig, ax = plt.subplots(figsize=(7, 6))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_title('Binary Group Confusion Matrix\n(Non-OA vs OA)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted Group', fontsize=11)
    ax.set_ylabel('True Group', fontsize=11)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Non-OA\n(Grade 0,1,2)', 'OA\n(Grade 3,4)'])
    ax.set_yticklabels(['Non-OA\n(Grade 0,1,2)', 'OA\n(Grade 3,4)'])

    for i in range(2):
        for j in range(2):
            ax.text(j, i,
                    f'{cm[i,j]}\n({cm_norm[i,j]:.2f})',
                    ha='center', va='center', fontsize=12,
                    color='white' if cm_norm[i,j] > 0.5 else 'black')

    plt.tight_layout()
    save_path = os.path.join(reports_dir, 'binary_group_confusion_matrix.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nBinary confusion matrix saved to {save_path}")

    # Clinical metrics
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv         = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv         = tn / (tn + fn) if (tn + fn) > 0 else 0

    print(f"\n── Clinical Metrics ────────────────────────────────")
    print(f"  Sensitivity (OA recall)    : {sensitivity:.4f} "
          f"({sensitivity*100:.1f}%) ← % of OA cases correctly detected")
    print(f"  Specificity (Non-OA recall): {specificity:.4f} "
          f"({specificity*100:.1f}%) ← % of Non-OA cases correctly identified")
    print(f"  PPV (OA precision)         : {ppv:.4f} "
          f"({ppv*100:.1f}%) ← when model says OA how often correct")
    print(f"  NPV (Non-OA precision)     : {npv:.4f} "
          f"({npv*100:.1f}%) ← when model says Non-OA how often correct")

    return {
        'binary_accuracy' : accuracy,
        'sensitivity'     : sensitivity,
        'specificity'     : specificity,
        'ppv'             : ppv,
        'npv'             : npv,
        'non_oa_correct'  : int(non_oa_correct),
        'non_oa_total'    : int(non_oa_total),
        'oa_correct'      : int(oa_correct),
        'oa_total'        : int(oa_total),
    }


def run_evaluation(model, test_loader, criterion, device):
    """Run full evaluation on test set and return predictions."""
    model.eval()

    total_loss = 0.0
    all_preds  = []
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss   = criterion(logits, labels)
            total_loss += loss.item()

            preds, probs = model.get_prediction(logits)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    f1       = f1_score(all_labels, all_preds,
                        average='macro', zero_division=0)

    return avg_loss, accuracy, f1, all_preds, all_labels, all_probs


def generate_gradcam_samples(model, base_dir, splits_dir,
                              reports_dir, device, n_samples=3):
    """
    Generate Grad-CAM heatmaps for sample images from each grade.
    Shows which regions of the X-ray the model focused on.
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        print("pytorch-grad-cam not installed.")
        print("Run: pip install grad-cam")
        print("Skipping Grad-CAM generation.")
        return

    print("\n── Generating Grad-CAM heatmaps ────────────────────")

    # Target the last conv layer of EfficientNet-B3
    target_layers = [model.backbone.blocks[-1]]

    # Load test CSV
    test_df = pd.read_csv(os.path.join(splits_dir, 'test.csv'))

    # Transform for loading images
    transform = T.Compose([
        T.Resize((180, 180)),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(mean=[0.2361, 0.2361, 0.2361],
                    std=[0.2095, 0.2095, 0.2095])
    ])

    fig, axes = plt.subplots(N_CLASSES, n_samples,
                             figsize=(n_samples * 4, N_CLASSES * 4))

    cam = GradCAM(model=model, target_layers=target_layers)

    for grade in range(N_CLASSES):
        grade_samples = test_df[
            test_df['kl_grade'] == grade
        ].sample(min(n_samples, len(test_df[test_df['kl_grade'] == grade])),
                 random_state=42)

        for col, (_, row) in enumerate(grade_samples.iterrows()):
            relative_path = row['image_path'].replace('\\', '/')
            img_path      = os.path.join(base_dir, relative_path)

            if not os.path.exists(img_path):
                axes[grade][col].axis('off')
                continue

            # Load original image for overlay
            img_pil  = Image.open(img_path).convert('RGB')
            img_np   = np.array(img_pil.resize((180, 180))) / 255.0

            # Prepare tensor for model
            img_gray = Image.open(img_path).convert('L')
            input_tensor = transform(img_gray).unsqueeze(0).to(device)

            # Generate Grad-CAM
            grayscale_cam = cam(input_tensor=input_tensor)
            cam_image     = show_cam_on_image(
                img_np.astype(np.float32),
                grayscale_cam[0],
                use_rgb=True
            )

            axes[grade][col].imshow(cam_image)
            axes[grade][col].axis('off')
            if col == 0:
                axes[grade][col].set_ylabel(
                    f'{GRADE_NAMES[grade]}',
                    fontsize=11, fontweight='bold'
                )

    plt.suptitle('Grad-CAM Heatmaps — Regions Used for KL Grade Prediction',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    save_path = os.path.join(reports_dir, 'gradcam_heatmaps.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Grad-CAM heatmaps saved to {save_path}")


# ── Main evaluation pipeline ──────────────────────────────────────────────────
if __name__ == '__main__':

    print("=" * 65)
    print("  OA SEVERITY CLASSIFICATION — EVALUATION")
    print("=" * 65)

    # ── Setup ─────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")

    BASE_DIR, SPLITS_DIR, CHECKPOINTS_DIR, REPORTS_DIR = get_paths()

    # ── Load data ─────────────────────────────────────────────────────────
    print("\n── Loading test data ───────────────────────────────")
    num_workers = 0 if sys.platform == 'win32' else 2
    _, _, test_loader, class_weights = get_dataloaders(
        SPLITS_DIR, BASE_DIR,
        batch_size=BATCH_SIZE,
        num_workers=num_workers
    )

    # ── Load model ────────────────────────────────────────────────────────
    print("\n── Loading model ───────────────────────────────────")
    checkpoint_path = os.path.join(CHECKPOINTS_DIR, 'best_model.pth')
    model           = load_model(checkpoint_path, device)

    # ── Loss function ─────────────────────────────────────────────────────
    class_weights = class_weights.to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    # ── Run evaluation ────────────────────────────────────────────────────
    print("\n── Running evaluation on test set ──────────────────")
    test_loss, test_acc, test_f1, test_preds, test_labels, test_probs = \
        run_evaluation(model, test_loader, criterion, device)

    print(f"\nTest Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_acc:.4f}  ({test_acc*100:.1f}%)")
    print(f"Test Macro F1 : {test_f1:.4f}")

    # ── Quadratic Weighted Kappa ──────────────────────────────────────────
    print("\n── Quadratic Weighted Kappa ────────────────────────")
    qwk = compute_quadratic_weighted_kappa(test_labels, test_preds)

   # ── Mean Absolute Error ───────────────────────────────────────────────
    print("\n── Mean Absolute Error ─────────────────────────────")
    mae = mean_absolute_error(test_labels, test_preds)
    print(f"Mean Absolute Error : {mae:.4f}")

    # ── Binary Group Evaluation ───────────────────────────────────────────
    print("\n── Binary Group Evaluation (Non-OA vs OA) ──────────")
    binary_results = evaluate_binary_groups(
        test_labels, test_preds, REPORTS_DIR
    )
    
    # ── Classification report ─────────────────────────────────────────────
    print("\n── Classification Report ───────────────────────────")
    print(classification_report(
        test_labels, test_preds,
        target_names=GRADE_NAMES,
        zero_division=0
    ))

    # ── Plot training curves ──────────────────────────────────────────────
    print("\n── Plotting training curves ────────────────────────")
    history_path = os.path.join(REPORTS_DIR, 'training_history.json')
    if os.path.exists(history_path):
        plot_training_curves(history_path, REPORTS_DIR)
    else:
        print(f"training_history.json not found at {history_path}")

    # ── Plot confusion matrix ─────────────────────────────────────────────
    print("\n── Plotting confusion matrix ───────────────────────")
    plot_confusion_matrix(test_labels, test_preds, REPORTS_DIR)

    # ── Plot per grade metrics ────────────────────────────────────────────
    print("\n── Plotting per grade metrics ──────────────────────")
    plot_per_grade_metrics(test_labels, test_preds, REPORTS_DIR)

    # ── Plot ROC curves ───────────────────────────────────────────────────────
    print("\n── Plotting ROC curves ─────────────────────────────")
    auc_scores, macro_auc = plot_roc_curves(
        test_labels, test_probs, REPORTS_DIR
    )
    

    # ── Grad-CAM ──────────────────────────────────────────────────────────
    print("\n── Generating Grad-CAM heatmaps ────────────────────")
    generate_gradcam_samples(
        model, BASE_DIR, SPLITS_DIR, REPORTS_DIR, device
    )

    # ── Save full results ─────────────────────────────────────────────────
    results = {
        'test_loss'              : test_loss,
        'test_accuracy'          : test_acc,
        'test_macro_f1'          : test_f1,
        'quadratic_weighted_kappa': qwk,
        'mean_absolute_error'     : mae,
        'binary_group_results'    : binary_results,
        'auc_per_grade'           : {f'grade_{i}': float(auc_scores[i]) for i in range(N_CLASSES)},
        'macro_auc'               : float(macro_auc),
    }
    results_path = os.path.join(REPORTS_DIR, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {results_path}")

    print(f"\n{'='*65}")
    print(f"  EVALUATION COMPLETE")
    print(f"  Test Accuracy          : {test_acc*100:.1f}%")
    print(f"  Test Macro F1          : {test_f1:.4f}")
    print(f"  Quadratic Weighted Kappa: {qwk:.4f}")
    print(f"  Mean Absolute Error     : {mae:.4f}")
    print(f"  Binary Accuracy         : {binary_results['binary_accuracy']*100:.1f}%")
    print(f"{'='*65}")