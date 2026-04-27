import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal loss for multi-class imbalanced classification.

    Combines class weighting with dynamic focusing on hard
    misclassified examples — down-weights easy correctly
    classified samples so training focuses on hard cases.

    Reference:
        Lin et al. (2017) — Focal Loss for Dense Object Detection
        arXiv:1708.02002

    Args:
        weight    : class weights tensor shape (n_classes,)
                    same as nn.CrossEntropyLoss weight parameter
        gamma     : focusing parameter (default 2.0)
                    0.0 = standard weighted cross-entropy
                    1.0 = mild focusing
                    2.0 = standard focal loss (recommended)
        reduction : mean or sum (default mean)
    """
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight    = weight
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits, labels):
        # Step 1 — standard cross-entropy loss per sample
        ce_loss = F.cross_entropy(
            logits, labels,
            weight=self.weight,
            reduction='none'    # keep per-sample losses
        )

        # Step 2 — get probability of correct class
        probs        = torch.softmax(logits, dim=1)
        correct_prob = probs.gather(
            dim=1,
            index=labels.unsqueeze(1)
        ).squeeze(1)

        # Step 3 — apply focal weighting
        # (1 - p)^gamma down-weights easy correctly classified samples
        focal_weight = (1 - correct_prob) ** self.gamma

        # Step 4 — combine focal weight with cross-entropy
        focal_loss = focal_weight * ce_loss

        # Step 5 — reduce to scalar
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss