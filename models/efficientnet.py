

################################
#Define model architecture
##################################
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' # fixes the Windows OpenMP conflict

import torch #deep learning framework
import torch.nn as nn
import timm #library of pretrained computer vision models —

#class definition
class EfficientNetB3OA(nn.Module):
    """
    EfficientNet-B3 fine-tuned for hand OA severity classification.

    Architecture:
        Backbone  : EfficientNet-B3 pretrained on ImageNet
        Head      : GlobalAvgPool → Dropout(0.4) → FC(256)
                    → ReLU → Dropout(0.3) → Output(n_classes)

    Training strategy:
        Stage 1 — Freeze backbone, train classifier head only (5 epochs)
        Stage 2 — Unfreeze backbone, fine-tune full model (remaining epochs)
    """
    #__init__ method is constructor to create model
    def __init__(self, n_classes=5, dropout_rate=0.4, pretrained=True):
        """
        Args:
            n_classes    : number of output classes (5 for KL grades 0-4)
            dropout_rate : dropout probability in classifier head
            pretrained   : load ImageNet pretrained weights
        """
        super(EfficientNetB3OA, self).__init__()

        # ── Load pretrained backbone ──────────────────────────────────────
        self.backbone = timm.create_model(
            'efficientnet_b3', # model name
            pretrained=pretrained, #loads ImageNet wieghts
            num_classes=0,       # remove default classifier
            global_pool='avg'    # global average pooling
        )

        # Number of features output by backbone
        backbone_features = self.backbone.num_features
        print(f"Backbone output features : {backbone_features}")

        # ── Custom classifier head ────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(backbone_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, n_classes)
        )

        # ── Initialize classifier weights ─────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        """
        Initialize classifier head with He initialization.
        This gives a better starting point than random initialization
        for ReLU activated networks.
        """
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Forward pass through backbone and classifier.

        Args:
            x      : tensor of shape (batch, 3, 180, 180)
        Returns:
            logits : tensor of shape (batch, n_classes)
                     raw scores before softmax
                     pass to CrossEntropyLoss directly
        """
        features = self.backbone(x)        # (batch, backbone_features)
        logits   = self.classifier(features) # (batch, n_classes)
        return logits

    def freeze_backbone(self):
        """
        Freeze all backbone parameters.
        Only classifier head will be trained.
        Call at the start of training — Stage 1.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        print(f"Backbone frozen — trainable parameters: {trainable:,}")

    def unfreeze_backbone(self):
        """
        Unfreeze all backbone parameters.
        Full model will be fine-tuned.
        Call after Stage 1 — Stage 2.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        print(f"Backbone unfrozen — trainable parameters: {trainable:,}")

    def count_parameters(self):
        """Print total, trainable and frozen parameter counts."""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        frozen    = total - trainable
        print(f"Total parameters     : {total:,}")
        print(f"Trainable parameters : {trainable:,}")
        print(f"Frozen parameters    : {frozen:,}")
        return trainable

    def get_prediction(self, logits):
        """
        Convert raw logits to predicted class.

        Args:
            logits : tensor of shape (batch, n_classes)
        Returns:
            preds  : tensor of shape (batch,) — predicted grade per image
            probs  : tensor of shape (batch, n_classes) — class probabilities
        """
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return preds, probs


# ── Test the model ────────────────────────────────────────────────────────────
if __name__ == '__main__':

    print("=" * 55)
    print("  EFFICIENTNET-B3 MODEL TEST")
    print("=" * 55)

    # ── Create model ──────────────────────────────────────────────────────
    print("\n── Creating model ───────────────────────────────────")
    model = EfficientNetB3OA(n_classes=5, pretrained=True)

    # ── Test Stage 1 — frozen backbone ───────────────────────────────────
    print("\n── Stage 1 — Frozen backbone ────────────────────────")
    model.freeze_backbone()
    model.count_parameters()

    # ── Test Stage 2 — unfrozen backbone ─────────────────────────────────
    print("\n── Stage 2 — Unfrozen backbone ──────────────────────")
    model.unfreeze_backbone()
    model.count_parameters()

    # ── Test forward pass ─────────────────────────────────────────────────
    print("\n── Forward pass test ────────────────────────────────")
    dummy_input = torch.randn(4, 3, 180, 180)  # batch of 4 images
    model.eval()
    with torch.no_grad():
        logits = model(dummy_input)

    print(f"Input shape  : {dummy_input.shape}")
    print(f"Output shape : {logits.shape}")

    # ── Test prediction ───────────────────────────────────────────────────
    print("\n── Prediction test ──────────────────────────────────")
    preds, probs = model.get_prediction(logits)
    print(f"Predicted grades : {preds.tolist()}")
    print(f"\nClass probabilities for first image:")
    for i, p in enumerate(probs[0].tolist()):
        bar = '█' * int(p * 40)
        print(f"  Grade {i} : {p:.4f}  {bar}")
    print(f"  Sum      : {probs[0].sum():.4f}  (should be 1.0000)")

    # ── Test loss calculation ─────────────────────────────────────────────
    print("\n── Loss calculation test ────────────────────────────")
    class_weights = torch.FloatTensor([0.3071, 1.5089, 1.0551, 13.3525, 17.1852])
    criterion     = nn.CrossEntropyLoss(weight=class_weights)
    dummy_labels  = torch.LongTensor([0, 2, 1, 3])  # fake labels
    loss          = criterion(logits, dummy_labels)
    print(f"Loss value : {loss.item():.4f}")
    print(f"Loss dtype : {loss.dtype}")

    # ── Device check ──────────────────────────────────────────────────────
    print("\n── Device check ─────────────────────────────────────")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device available : {device}")
    if device.type == 'cuda':
        print(f"GPU name         : {torch.cuda.get_device_name(0)}")
    else:
        print(f"No GPU found — running on CPU")
        print(f"This is fine for testing — use Colab for full training")

    print("\n✓ EfficientNet-B3 model working correctly")