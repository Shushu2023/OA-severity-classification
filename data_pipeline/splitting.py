import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ── Constants ─────────────────────────────────────────────────────────────────
RANDOM_SEED  = 42
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
TEST_RATIO   = 0.15
N_CLASSES    = 5
FINAL_JOINTS = ['dip2', 'dip3', 'dip4', 'dip5']

KL_COLS_MAP = {
    'dip2': 'v00DIP2_KL',
    'dip3': 'v00DIP3_KL',
    'dip4': 'v00DIP4_KL',
    'dip5': 'v00DIP5_KL',
}

# Relative image path from project root
RELATIVE_IMAGE_DIR = os.path.join('data', 'raw', 'images', 'Finger Joints')


def get_paths():
    """Auto-detect project root and build all required paths."""
    BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    DATA_DIR   = os.path.join(BASE_DIR, 'data', 'raw')
    IMAGE_DIR  = os.path.join(DATA_DIR, 'images', 'Finger Joints')
    LABELS_PATH = os.path.join(DATA_DIR, 'hand.xlsx')
    SPLITS_DIR = os.path.join(BASE_DIR, 'data', 'splits')
    REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
    return BASE_DIR, DATA_DIR, IMAGE_DIR, LABELS_PATH, SPLITS_DIR, REPORTS_DIR


def build_master_dataset(df, image_dir, base_dir):
    """
    Build master dataset from raw Excel file.
    Includes only valid patient-joint pairs where:
        - Image file exists on disk
        - KL grade label is present and non-null
        - Joint is one of the 4 DIP joints
        - KL grade is original 5-class (0-4, no merging)

    Returns:
        valid_df : DataFrame with columns
                   [patient_id, joint, kl_grade, image_path]
    """
    valid_samples = []

    # Get patients with images
    patients_with_images = set()
    for f in os.listdir(image_dir):
        if f.endswith('.png'):
            patients_with_images.add(f.split('_')[0])

    print(f"Patients with images : {len(patients_with_images)}")

    for _, row in df.iterrows():
        patient_id = str(int(row['id']))

        if patient_id not in patients_with_images:
            continue

        for joint in FINAL_JOINTS:
            kl_col   = KL_COLS_MAP[joint]
            kl_grade = row[kl_col]
            img_file = f'{patient_id}_{joint}.png'
            img_path = os.path.join(image_dir, img_file)

            # Only include if both image and label exist
            if os.path.exists(img_path) and not pd.isna(kl_grade):
                # Store relative path from project root
                relative_path = os.path.join(RELATIVE_IMAGE_DIR, img_file)
                valid_samples.append({
                    'patient_id' : patient_id,
                    'joint'      : joint,
                    'kl_grade'   : int(kl_grade),
                    'image_path' : relative_path,
                })

    valid_df = pd.DataFrame(valid_samples)
    print(f"Total valid samples  : {len(valid_df):,}")
    return valid_df


def patient_level_split(df, seed=RANDOM_SEED):
    """
    Split at patient level to prevent data leakage.
    All joints from same patient stay in same split.

    Returns:
        train_df, val_df, test_df
    """
    unique_patients = df['patient_id'].unique()
    print(f"Unique patients      : {len(unique_patients)}")

    # Split off test (15%)
    train_val_patients, test_patients = train_test_split(
        unique_patients, test_size=TEST_RATIO, random_state=seed
    )

    # Split remaining into train (70%) and val (15%)
    val_size = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train_patients, val_patients = train_test_split(
        train_val_patients, test_size=val_size, random_state=seed
    )

    train_df = df[df['patient_id'].isin(train_patients)].reset_index(drop=True)
    val_df   = df[df['patient_id'].isin(val_patients)].reset_index(drop=True)
    test_df  = df[df['patient_id'].isin(test_patients)].reset_index(drop=True)

    print(f"\nPatient split:")
    print(f"  Train : {len(train_patients):,} patients → {len(train_df):,} samples")
    print(f"  Val   : {len(val_patients):,} patients → {len(val_df):,} samples")
    print(f"  Test  : {len(test_patients):,} patients → {len(test_df):,} samples")

    return train_df, val_df, test_df


def verify_no_leakage(train_df, val_df, test_df):
    """Verify no patient appears in more than one split."""
    train_ids = set(train_df['patient_id'].unique())
    val_ids   = set(val_df['patient_id'].unique())
    test_ids  = set(test_df['patient_id'].unique())

    print(f"\n── Data leakage check ──────────────────────────────")
    tv = len(train_ids.intersection(val_ids))
    tt = len(train_ids.intersection(test_ids))
    vt = len(val_ids.intersection(test_ids))

    print(f"Train ∩ Val  : {tv}  "
          f"{'✓ No leakage' if tv == 0 else '✗ LEAKAGE DETECTED'}")
    print(f"Train ∩ Test : {tt}  "
          f"{'✓ No leakage' if tt == 0 else '✗ LEAKAGE DETECTED'}")
    print(f"Val ∩ Test   : {vt}  "
          f"{'✓ No leakage' if vt == 0 else '✗ LEAKAGE DETECTED'}")

    return tv == 0 and tt == 0 and vt == 0


def print_grade_distribution(train_df, val_df, test_df):
    """Print exact grade distribution across all three splits."""
    print(f"\n── Grade distribution ──────────────────────────────")
    print(f"{'Grade':<8} | {'Train':>7} | {'Val':>7} | "
          f"{'Test':>7} | {'Total':>7}")
    print("-" * 45)

    for grade in range(N_CLASSES):
        t  = len(train_df[train_df['kl_grade'] == grade])
        v  = len(val_df[val_df['kl_grade'] == grade])
        te = len(test_df[test_df['kl_grade'] == grade])
        print(f"Grade {grade}  | {t:>7} | {v:>7} | {te:>7} | {t+v+te:>7}")

    print("-" * 45)
    print(f"{'TOTAL':<8} | {len(train_df):>7} | {len(val_df):>7} | "
          f"{len(test_df):>7} | {len(train_df)+len(val_df)+len(test_df):>7}")


def calculate_class_weights(train_df):
    """Calculate and print class weights from training set."""
    counts      = train_df['kl_grade'].value_counts().sort_index()
    total       = len(train_df)
    n_classes   = N_CLASSES

    print(f"\n── Class weights ───────────────────────────────────")
    weights = []
    for i in range(n_classes):
        w = total / (n_classes * counts.get(i, 1))
        weights.append(round(w, 4))
        print(f"  Grade {i} : {counts.get(i, 0):>5} samples → weight {w:.4f}")

    print(f"\nCLASS_WEIGHTS = {weights}")
    return weights


def save_splits(valid_df, train_df, val_df, test_df, splits_dir):
    """Save all CSV files to splits directory."""
    os.makedirs(splits_dir, exist_ok=True)

    valid_df.to_csv(os.path.join(splits_dir, 'master_dataset.csv'), index=False)
    train_df.to_csv(os.path.join(splits_dir, 'train.csv'),          index=False)
    val_df.to_csv(os.path.join(splits_dir,   'val.csv'),            index=False)
    test_df.to_csv(os.path.join(splits_dir,  'test.csv'),           index=False)

    print(f"\n── Splits saved ────────────────────────────────────")
    print(f"  master_dataset.csv : {len(valid_df):,} samples")
    print(f"  train.csv          : {len(train_df):,} samples")
    print(f"  val.csv            : {len(val_df):,} samples")
    print(f"  test.csv           : {len(test_df):,} samples")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    print("=" * 55)
    print("  SPLITTING PIPELINE")
    print("  Joints  : DIP2, DIP3, DIP4, DIP5")
    print("  Classes : 5 (KL grades 0-4, no merging)")
    print("  Visit   : v00 (baseline)")
    print("  Split   : 70/15/15 patient-level")
    print("=" * 55)

    # Get paths
    BASE_DIR, DATA_DIR, IMAGE_DIR, LABELS_PATH, SPLITS_DIR, REPORTS_DIR = \
        get_paths()

    # Load raw Excel
    print(f"\nLoading {LABELS_PATH}...")
    df = pd.read_excel(LABELS_PATH)
    print(f"Raw data shape : {df.shape}")

    # Build master dataset
    print(f"\n── Building master dataset ─────────────────────────")
    valid_df = build_master_dataset(df, IMAGE_DIR, BASE_DIR)

    # Split
    print(f"\n── Splitting ───────────────────────────────────────")
    train_df, val_df, test_df = patient_level_split(valid_df)

    # Verify no leakage
    all_clear = verify_no_leakage(train_df, val_df, test_df)
    if not all_clear:
        raise ValueError("Data leakage detected — do not proceed.")

    # Print distributions
    print_grade_distribution(train_df, val_df, test_df)

    # Calculate class weights
    calculate_class_weights(train_df)

    # Save
    save_splits(valid_df, train_df, val_df, test_df, SPLITS_DIR)

    print("\n✓ Splitting pipeline complete.")