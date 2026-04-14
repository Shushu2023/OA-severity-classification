import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# ── Dataset constants ─────────────────────────────────────────────────────────
IMAGE_SIZE   = 180
DATASET_MEAN = 0.2361 # computed from hand training images for normalization
DATASET_STD  = 0.2095 #computed from hand training images for normalization
N_CLASSES    = 5
BATCH_SIZE   = 32


# ── Transforms ────────────────────────────────────────────────────────────────
def get_train_transforms():
    """No augmentation — resize and normalize only."""
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)), #ensure every image is of the same size
        T.Grayscale(num_output_channels=3), #convert gray to color
        T.ToTensor(), #  convert 0-255 to 0-1 float and rearrange image  dimession becomes  3x180x180 tensor
        T.Normalize(#subtract dataset mean, divide by dataset std
            mean=[DATASET_MEAN, DATASET_MEAN, DATASET_MEAN],
            std=[DATASET_STD,   DATASET_STD,  DATASET_STD]
        )
    ])


def get_val_transforms():
    """Same as train — no augmentation."""
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(
            mean=[DATASET_MEAN, DATASET_MEAN, DATASET_MEAN],
            std=[DATASET_STD,   DATASET_STD,  DATASET_STD]
        )
    ])


# ── Dataset class ─────────────────────────────────────────────────────────────
#define a custom dataset class that inherits from PyTorch's built in dataset base class
#to be able to work with PyTorch's DataLoader
#must implement 3 PyTorch methods: __init__,__len__, __getitem__
#reads CSV -stores paths and labels in memory
class OADataset(Dataset):
    """
    PyTorch Dataset for OA severity classification.

    image_path column stores a relative path from the project root.
    Full path is reconstructed at runtime as:
        base_dir / image_path
    This makes the dataset portable across laptop, Colab, and SCC.
    """
    #runs once when dataset is created to create datset object
    def __init__(self, csv_path, base_dir, transform=None):
        """
        Args:
            csv_path : path to train.csv, val.csv, or test.csv
            base_dir : project root directory — used to resolve
                       relative image paths in the CSV
            transform: torchvision transforms to apply
        """
        self.df        = pd.read_csv(csv_path) #CSV only contains file paths and labels, not the actual images.
        self.base_dir  = base_dir
        self.transform = transform

        # Verify required columns exist in the CSV before training starts
        required_cols = ['patient_id', 'joint', 'kl_grade', 'image_path']
        for col in required_cols:
            assert col in self.df.columns, \
                f"Column '{col}' not found in {csv_path}"

        print(f"Dataset loaded     : {os.path.basename(csv_path)}")
        print(f"Total samples      : {len(self.df)}")
        print(f"Base directory     : {base_dir}")
        print(f"Grade distribution : "
              f"{self.df['kl_grade'].value_counts().sort_index().to_dict()}")

    def __len__(self): # tells DataLoader how many samples exist to know when a full epoch is complete
        return len(self.df)

    def __getitem__(self, idx): #Images are loaded one by one on demand called thousands of times per epoch
        """
        Loads one sample.
        Reconstructs full image path from base_dir + relative path.
        """
        row           = self.df.iloc[idx]
        relative_path = row['image_path']
        label         = int(row['kl_grade'])

        # Reconstruct full path — works on any machine
        relative_path_fixed = relative_path.replace('\\', '/') #backslashes \ (Windows format) but Linux (Colab) needs forward slashes /.
        full_path = os.path.join(self.base_dir, relative_path_fixed)

        # Load image as grayscale  L gray mode because PIL can sometimes open PNG files in diffent modes like RGBA
        image = Image.open(full_path).convert('L')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label
   
   #calculate training weights used by CrossEntropyLoss to conpensate for class imbalance
   #weight = total / (n_classes x class_count)#Grades with fewer samples get higher weights
   # #method is only called on the training dataset-never on validation or test 
    def get_class_weights(self):
        """
        Calculate class weights from this dataset.
        Call only on training set — never on val or test.
        """
        counts    = self.df['kl_grade'].value_counts().sort_index()
        total     = len(self.df)
        n_classes = N_CLASSES

        weights = torch.FloatTensor([
            total / (n_classes * counts.get(i, 1))
            for i in range(n_classes)
        ])

        print("\nClass weights:")
        for i, w in enumerate(weights):
            print(f"  Grade {i} : {counts.get(i, 0):>5} samples "
                  f"→ weight {w:.4f}")

        return weights


# ── DataLoader factory ────────────────────────────────────────────────────────
#creates all three Dataloaders OADateset for each split to wrap each dataset in a DataLoader
def get_dataloaders(splits_dir, base_dir, batch_size=BATCH_SIZE, num_workers=0):
    """
    Creates train, val, and test DataLoaders.

    Args:
        splits_dir  : folder containing train/val/test CSV files
        base_dir    : project root — used to resolve relative image paths
        batch_size  : samples per batch (default 32)
        num_workers : parallel workers (use 0 on Windows)

    Returns:
        train_loader, val_loader, test_loader, class_weights
    """
    train_csv = os.path.join(splits_dir, 'train.csv')
    val_csv   = os.path.join(splits_dir, 'val.csv')
    test_csv  = os.path.join(splits_dir, 'test.csv')

    # Create datasets
    train_dataset = OADataset(train_csv, base_dir,
                              transform=get_train_transforms())
    val_dataset   = OADataset(val_csv,   base_dir,
                              transform=get_val_transforms())
    test_dataset  = OADataset(test_csv,  base_dir,
                              transform=get_val_transforms())

    # Class weights from training set only
    class_weights = train_dataset.get_class_weights()

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True,  num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,   batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,  batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )

    print(f"\n── DataLoaders created ──────────────────────────")
    print(f"Train batches : {len(train_loader)}")
    print(f"Val batches   : {len(val_loader)}")
    print(f"Test batches  : {len(test_loader)}")

    return train_loader, val_loader, test_loader, class_weights


# ── Test the pipeline ─────────────────────────────────────────────────────────
#runs only when python data_pipeline/preprocessing.py is executed directly 
# to verify entire pipeline works end to tend befor commiting to full training run
if __name__ == '__main__':

    # Project root is one level up from data_pipeline/
    BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    SPLITS_DIR = os.path.join(BASE_DIR, 'data', 'splits')

    print("=" * 55)
    print("  PREPROCESSING PIPELINE TEST")
    print("=" * 55)

    train_loader, val_loader, test_loader, class_weights = \
        get_dataloaders(SPLITS_DIR, BASE_DIR)

    # Test one batch
    print("\n── Testing one batch ────────────────────────────")
    images, labels = next(iter(train_loader))
    print(f"Batch image shape : {images.shape}")
    print(f"Batch label shape : {labels.shape}")
    print(f"Image dtype       : {images.dtype}")
    print(f"Label dtype       : {labels.dtype}")
    print(f"Pixel min         : {images.min():.4f}")
    print(f"Pixel max         : {images.max():.4f}")
    print(f"Sample labels     : {labels[:8].tolist()}")
    print("\n✓ Preprocessing pipeline working correctly")