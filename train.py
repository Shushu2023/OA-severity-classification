import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import time
import json

# ── Import project modules ────────────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_pipeline.preprocessing import get_dataloaders, N_CLASSES, BATCH_SIZE
from models.efficientnet import EfficientNetB3OA
from losses import FocalLoss 

# ── Training constants ────────────────────────────────────────────────────────
NUM_EPOCHS        = 300
STAGE1_EPOCHS     = 5        # freeze backbone — train head only
LEARNING_RATE     = 1e-4     # initial learning rate
WEIGHT_DECAY      = 1e-4     # L2 regularization
EARLY_STOP_PATIENCE = 15      # stop if val F1 does not improve for 15 epochs
MIN_LR            = 1e-6     # minimum learning rate for scheduler

# EXPERIMENT_NAME     = 'efficientnetb3_crossentropy_300ep' # used  for weighted crossentropy loss expirement
FOCAL_GAMMA     = 2.0
EXPERIMENT_NAME = 'efficientnetb3_focalloss_300ep' 

def get_device():
    """Detect and return best available device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Device : GPU — {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print(f"Device : CPU — no GPU found")
        print(f"Warning: training on CPU will be very slow.")
        print(f"Recommend moving to Google Colab for full training.")
    return device

def get_paths():
    """Auto-detect paths — supports local laptop, Colab, and SCC."""

    # Check for environment variable overrides (set by Colab cell)
    if os.environ.get('OA_BASE_DIR'):
        BASE_DIR        = os.environ['OA_BASE_DIR']
        SPLITS_DIR      = os.environ.get('OA_SPLITS_DIR',
                          os.path.join(BASE_DIR, 'data', 'splits'))
        CHECKPOINTS_DIR = os.environ.get('OA_CHECKPOINTS_DIR',
                          os.path.join(BASE_DIR, 'checkpoints'))
        REPORTS_DIR     = os.environ.get('OA_REPORTS_DIR',
                          os.path.join(BASE_DIR, 'reports'))
    else:
        # Default — local laptop or SCC
        BASE_DIR        = os.path.abspath(os.path.dirname(__file__))
        SPLITS_DIR      = os.path.join(BASE_DIR, 'data', 'splits')
        CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'checkpoints')
        REPORTS_DIR     = os.path.join(BASE_DIR, 'reports')

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print(f"BASE_DIR        : {BASE_DIR}")
    print(f"SPLITS_DIR      : {SPLITS_DIR}")
    print(f"CHECKPOINTS_DIR : {CHECKPOINTS_DIR}")
    print(f"REPORTS_DIR     : {REPORTS_DIR}")

    return BASE_DIR, SPLITS_DIR, CHECKPOINTS_DIR, REPORTS_DIR



def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """
    Run one full training epoch.
    Passes all batches through the model, calculates loss,
    and updates weights via backpropagation.

    Returns:
        avg_loss : average loss across all batches
        accuracy : overall accuracy for this epoch
        f1       : macro F1 score for this epoch
    """
    model.train()  # set model to training mode — enables dropout

    total_loss  = 0.0
    all_preds   = []
    all_labels  = []
    n_batches   = len(loader)

    for batch_idx, (images, labels) in enumerate(loader):

        # Move data to GPU if available
        images = images.to(device)
        labels = labels.to(device)

        # ── Forward pass ──────────────────────────────────────────────────
        optimizer.zero_grad()          # clear gradients from last batch
        logits = model(images)         # get predictions
        loss   = criterion(logits, labels)  # calculate loss

        # ── Backward pass ─────────────────────────────────────────────────
        loss.backward()                # calculate gradients
        optimizer.step()               # update weights

        # ── Track metrics ─────────────────────────────────────────────────
        total_loss += loss.item()

        preds, _ = model.get_prediction(logits)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Print progress every 50 batches
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx+1:>3}/{n_batches} | "
                  f"Loss: {loss.item():.4f}")

    # Calculate epoch metrics
    avg_loss = total_loss / n_batches
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    f1       = f1_score(all_labels, all_preds,
                        average='macro', zero_division=0)

    return avg_loss, accuracy, f1


def evaluate_one_epoch(model, loader, criterion, device):
    """
    Evaluate model on validation or test set.
    No weight updates — inference only.

    Returns:
        avg_loss : average loss
        accuracy : overall accuracy
        f1       : macro F1 score
        all_preds  : list of all predictions
        all_labels : list of all true labels
    """
    model.eval()  # set model to eval mode — disables dropout

    total_loss = 0.0
    all_preds  = []
    all_labels = []

    with torch.no_grad():  # disable gradient calculation — saves memory
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss   = criterion(logits, labels)

            total_loss += loss.item()

            preds, _ = model.get_prediction(logits)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    f1       = f1_score(all_labels, all_preds,
                        average='macro', zero_division=0)

    return avg_loss, accuracy, f1, all_preds, all_labels


def save_checkpoint(model, optimizer, epoch, val_f1, 
                    checkpoints_dir, filename):
    """Save model checkpoint to disk."""
    checkpoint = {
        'epoch'               : epoch,
        'model_state_dict'    : model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_f1'              : val_f1,
    }
    path = os.path.join(checkpoints_dir, filename)
    torch.save(checkpoint, path)
    return path


def print_epoch_summary(epoch, num_epochs, stage,
                         train_loss, train_acc, train_f1,
                         val_loss,   val_acc,   val_f1,
                         lr, epoch_time):
    """Print a formatted summary after each epoch."""
    print(f"\n{'='*65}")
    print(f"Epoch {epoch:>2}/{num_epochs}  |  Stage: {stage}  |  "
          f"Time: {epoch_time:.1f}s  |  LR: {lr:.2e}")
    print(f"{'─'*65}")
    print(f"{'':12} {'Loss':>10} {'Accuracy':>10} {'Macro F1':>10}")
    print(f"{'Train':12} {train_loss:>10.4f} {train_acc:>10.4f} "
          f"{train_f1:>10.4f}")
    print(f"{'Val':12} {val_loss:>10.4f} {val_acc:>10.4f} "
          f"{val_f1:>10.4f}")
    print(f"{'='*65}")

import time
def train(num_epochs=NUM_EPOCHS, test_run=False):
    """
    Full training pipeline.

    Args:
        num_epochs : total epochs to train
        test_run   : if True run only 2 epochs for quick testing
    """
    training_start = time.time() #track training time

    
    if test_run:
        num_epochs = 2
        print("TEST RUN MODE — training for 2 epochs only")

    print("=" * 65)
    print("  OA SEVERITY CLASSIFICATION — TRAINING")
    print(f"  Model    : EfficientNet-B3")
    print(f"  Classes  : {N_CLASSES} (KL grades 0-4)")
    print(f"  Epochs   : {num_epochs}")
    print(f"  Batch    : {BATCH_SIZE}")
    print("=" * 65)

    # ── Setup ─────────────────────────────────────────────────────────────
    device = get_device()
    BASE_DIR, SPLITS_DIR, CHECKPOINTS_DIR, REPORTS_DIR = get_paths()

    # ── Data ──────────────────────────────────────────────────────────────
    print("\n── Loading data ────────────────────────────────────")
    num_workers = 0 if sys.platform == 'win32' else 4
    train_loader, val_loader, test_loader, class_weights = \
        get_dataloaders(SPLITS_DIR, BASE_DIR,
                        batch_size=BATCH_SIZE,
                        num_workers=num_workers)

    # ── Model ─────────────────────────────────────────────────────────────
    print("\n── Creating model ──────────────────────────────────")
    model = EfficientNetB3OA(n_classes=N_CLASSES, pretrained=True)
    model = model.to(device)

    # ── Loss function with class weights to measure how wrong the model is during training. ──────────────────────────────────
    #class_weights = class_weights.to(device)
    #criterion     = nn.CrossEntropyLoss(weight=class_weights)

    #_____________ Loss function FocalLoss to measure how wrong the model is during training.___________________________
    class_weights = class_weights.to(device)
    criterion     = FocalLoss(
        weight=class_weights, 
        gamma=FOCAL_GAMMA,
        reduction='mean' # used because it makes the loss value independent of batch size.
    )

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # ── Learning rate scheduler ───────────────────────────────────────────
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=MIN_LR
    )

    # ── Training history ──────────────────────────────────────────────────
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss'  : [], 'val_acc'  : [], 'val_f1'  : [],
        'lr'        : [],
        'epoch_time' : [] 
    }

    # ── Early stopping ────────────────────────────────────────────────────
    #initialzaiont befro the training loop begins
    best_val_f1    = 0.0
    patience_count = 0
    best_epoch     = 0

    # ── Stage 1 — freeze backbone ─────────────────────────────────────────
    print(f"\n── Stage 1: Freeze backbone (epochs 1-{STAGE1_EPOCHS}) ──────")
    model.freeze_backbone()

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(1, num_epochs + 1):

        epoch_start = time.time()

        # Switch to Stage 2 after STAGE1_EPOCHS
        if epoch == STAGE1_EPOCHS + 1:
            print(f"\n── Stage 2: Unfreeze backbone (epoch {epoch}+) ──────")
            model.unfreeze_backbone()

            # Reinitialize optimizer with all parameters
            optimizer = optim.AdamW(
                model.parameters(),
                lr=LEARNING_RATE / 10,  # lower LR for fine-tuning
                weight_decay=WEIGHT_DECAY
            )
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=num_epochs - STAGE1_EPOCHS,
                eta_min=MIN_LR
            )

        # Determine current stage label
        stage = "1 — Head only" if epoch <= STAGE1_EPOCHS \
                else "2 — Full model"

        print(f"\nEpoch {epoch}/{num_epochs} — {stage}")

        # ── Train ──────────────────────────────────────────────────────
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # ── Validate ───────────────────────────────────────────────────
        val_loss, val_acc, val_f1, val_preds, val_labels = \
            evaluate_one_epoch(model, val_loader, criterion, device)

        # ── Update scheduler ───────────────────────────────────────────
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # ── Log history ────────────────────────────────────────────────
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['lr'].append(current_lr)

        # ── Print summary ──────────────────────────────────────────────
        epoch_time = time.time() - epoch_start
        print_epoch_summary(
            epoch, num_epochs, stage,
            train_loss, train_acc, train_f1,
            val_loss,   val_acc,   val_f1,
            current_lr, epoch_time
        )
        history['epoch_time'].append(epoch_time)

       # ── Save best model ────────────────────────────────────────────
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch
            #save best model with experiment name
            path = save_checkpoint(
                model, optimizer, epoch, val_f1,
                CHECKPOINTS_DIR,
                f'best_model_{EXPERIMENT_NAME}.pth'
            )
            print(f"  ✓ New best model saved  "
                  f"(Val F1: {val_f1:.4f}) → {path}")
            patience_count = 0
        else:
            patience_count += 1
            print(f"  No improvement — patience {patience_count}"
                  f"/{EARLY_STOP_PATIENCE}")

        # ── Save latest checkpoint every epoch ─────────────────────────
        #Save latest checkpoint with experiment name
        save_checkpoint(
            model, optimizer, epoch, val_f1,
            CHECKPOINTS_DIR,
            f'latest_checkpoint_{EXPERIMENT_NAME}.pth'
        )

        # ── Early stopping ─────────────────────────────────────────────
        #check if patience is exhausted
        if patience_count >= EARLY_STOP_PATIENCE and \
                not test_run:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            print(f"No improvement for {EARLY_STOP_PATIENCE} epochs")
            break
     # ── Training time summary ─────────────────────────────────────────────
    total_time      = time.time() - training_start
    avg_stage1_time = np.mean(history['epoch_time'][:STAGE1_EPOCHS])
    avg_stage2_time = np.mean(history['epoch_time'][STAGE1_EPOCHS:])

    print(f"\n── Training Time Summary ───────────────────────────")
    print(f"  Total training time    : {total_time/60:.1f} minutes")
    print(f"  Avg Stage 1 epoch time : {avg_stage1_time:.1f} seconds")
    print(f"  Avg Stage 2 epoch time : {avg_stage2_time:.1f} seconds")
    if torch.cuda.is_available():
        print(f"  GPU                    : {torch.cuda.get_device_name(0)}")

    # ── Save training history ─────────────────────────────────────────────
    history_path = os.path.join(
        REPORTS_DIR,
        f'training_history_{EXPERIMENT_NAME}.json'
    )
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")

    # ── Final evaluation on test set ──────────────────────────────────────
    print("\n" + "=" * 65)
    print("  FINAL EVALUATION ON TEST SET")
    print("=" * 65)

    # Load best model for test evaluation
    best_checkpoint = torch.load(
        os.path.join(CHECKPOINTS_DIR, 'best_model.pth'),
        map_location=device
    )
    model.load_state_dict(best_checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {best_checkpoint['epoch']} "
          f"(Val F1: {best_checkpoint['val_f1']:.4f})")

    test_loss, test_acc, test_f1, test_preds, test_labels = \
        evaluate_one_epoch(model, test_loader, criterion, device)

    print(f"\nTest Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_acc:.4f}")
    print(f"Test Macro F1 : {test_f1:.4f}")

    # ── Classification report ─────────────────────────────────────────────
    print(f"\n── Classification Report ───────────────────────────")
    grade_names = ['Grade 0', 'Grade 1', 'Grade 2',
                   'Grade 3', 'Grade 4']
    print(classification_report(
        test_labels, test_preds,
        target_names=grade_names,
        zero_division=0
    ))

    # ── Confusion matrix ──────────────────────────────────────────────────
    print(f"── Confusion Matrix ────────────────────────────────")
    cm = confusion_matrix(test_labels, test_preds)
    print(f"{'':10}", end='')
    for name in grade_names:
        print(f"{name:>10}", end='')
    print()
    for i, row in enumerate(cm):
        print(f"{grade_names[i]:10}", end='')
        for val in row:
            print(f"{val:>10}", end='')
        print()

    # ── Save results ──────────────────────────────────────────────────────
    results = {
        'best_epoch'    : best_epoch,
        'best_val_f1'   : best_val_f1,
        'test_loss'     : test_loss,
        'test_accuracy' : test_acc,
        'test_macro_f1' : test_f1,
        'total_training_time_minutes' : round(total_time / 60, 1),
        'avg_stage1_epoch_seconds'    : round(avg_stage1_time, 1),
        'avg_stage2_epoch_seconds'    : round(avg_stage2_time, 1),
        'gpu'                         : torch.cuda.get_device_name(0)
                                        if torch.cuda.is_available()
                                        else 'CPU',
        'experiment_name'             : EXPERIMENT_NAME,
    }
    results_path = os.path.join(
        REPORTS_DIR,
        f'test_results_{EXPERIMENT_NAME}.json'
    )
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nTest results saved to {results_path}")

    print(f"\n{'='*65}")
    print(f"  TRAINING COMPLETE")
    print(f"  Best epoch    : {best_epoch}")
    print(f"  Best Val F1   : {best_val_f1:.4f}")
    print(f"  Test Macro F1 : {test_f1:.4f}")
    print(f"{'='*65}")

    return model, history, results


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':

    # Check for test_run flag
    test_run = '--test' in sys.argv

    # Run training
    model, history, results = train(test_run=test_run)