import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ── Constants ────────────────────────────────────────────────────────────────
RANDOM_SEED  = 42
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
TEST_RATIO   = 0.15


def load_master_dataset(splits_dir):
    """Load master dataset from CSV."""
    path = os.path.join(splits_dir, 'master_dataset.csv')
    assert os.path.exists(path), f"master_dataset.csv not found at {path}"
    df = pd.read_csv(path)
    print(f"Master dataset loaded : {df.shape}")
    return df


def patient_level_split(df, seed=RANDOM_SEED):
    """
    Split dataset at the PATIENT level to prevent data leakage.
    All joints from the same patient always stay in the same split.
    
    Returns train_df, val_df, test_df
    """
    # Get unique patients
    unique_patients = df['patient_id'].unique()
    print(f"\nTotal unique patients : {len(unique_patients)}")

    # Step 1 — Split off test set (15%)
    train_val_patients, test_patients = train_test_split(
        unique_patients,
        test_size=TEST_RATIO,
        random_state=seed
    )

    # Step 2 — Split remaining into train (70%) and val (15%)
    # val_size is relative to train_val remaining
    val_size = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train_patients, val_patients = train_test_split(
        train_val_patients,
        test_size=val_size,
        random_state=seed
    )

    print(f"\nPatient split:")
    print(f"  Train patients : {len(train_patients)} "
          f"({len(train_patients)/len(unique_patients)*100:.1f}%)")
    print(f"  Val patients   : {len(val_patients)} "
          f"({len(val_patients)/len(unique_patients)*100:.1f}%)")
    print(f"  Test patients  : {len(test_patients)} "
          f"({len(test_patients)/len(unique_patients)*100:.1f}%)")

    # Assign rows to splits based on patient ID
    train_df = df[df['patient_id'].isin(train_patients)].reset_index(drop=True)
    val_df   = df[df['patient_id'].isin(val_patients)].reset_index(drop=True)
    test_df  = df[df['patient_id'].isin(test_patients)].reset_index(drop=True)

    return train_df, val_df, test_df


def verify_no_leakage(train_df, val_df, test_df):
    """
    Confirm no patient appears in more than one split.
    This is the most important check for data integrity.
    """
    train_patients = set(train_df['patient_id'].unique())
    val_patients   = set(val_df['patient_id'].unique())
    test_patients  = set(test_df['patient_id'].unique())

    train_val_overlap  = train_patients.intersection(val_patients)
    train_test_overlap = train_patients.intersection(test_patients)
    val_test_overlap   = val_patients.intersection(test_patients)

    print(f"\n── Data leakage check ─────────────────────────────")
    print(f"Train ∩ Val  overlap : {len(train_val_overlap)}  "
          f"{'✓ No leakage' if len(train_val_overlap) == 0 else '✗ LEAKAGE DETECTED'}")
    print(f"Train ∩ Test overlap : {len(train_test_overlap)}  "
          f"{'✓ No leakage' if len(train_test_overlap) == 0 else '✗ LEAKAGE DETECTED'}")
    print(f"Val ∩ Test   overlap : {len(val_test_overlap)}  "
          f"{'✓ No leakage' if len(val_test_overlap) == 0 else '✗ LEAKAGE DETECTED'}")

    all_clear = (len(train_val_overlap) == 0 and
                 len(train_test_overlap) == 0 and
                 len(val_test_overlap) == 0)

    return all_clear


def verify_grade_distribution(train_df, val_df, test_df):
    """
    Check that grade distribution is roughly preserved across splits.
    Perfect stratification is not possible with patient level splits
    but distributions should be similar.
    """
    print(f"\n── Grade distribution across splits ───────────────")
    print(f"{'Grade':<8} | {'Train':>10} | {'Val':>10} | {'Test':>10}")
    print("-" * 45)

    for grade in sorted(train_df['kl_grade_merged'].unique()):
        t_pct = (train_df['kl_grade_merged'] == grade).sum() / len(train_df) * 100
        v_pct = (val_df['kl_grade_merged']   == grade).sum() / len(val_df)   * 100
        te_pct = (test_df['kl_grade_merged'] == grade).sum() / len(test_df)  * 100
        print(f"Grade {grade}  | {t_pct:>9.1f}% | {v_pct:>9.1f}% | {te_pct:>9.1f}%")


def verify_joint_distribution(train_df, val_df, test_df):
    """Check joint distribution is consistent across splits."""
    print(f"\n── Joint distribution across splits ───────────────")
    print(f"{'Joint':<8} | {'Train':>8} | {'Val':>8} | {'Test':>8}")
    print("-" * 38)

    all_joints = sorted(train_df['joint'].unique())
    for joint in all_joints:
        t  = (train_df['joint'] == joint).sum()
        v  = (val_df['joint']   == joint).sum()
        te = (test_df['joint']  == joint).sum()
        print(f"{joint.upper():<8} | {t:>8} | {v:>8} | {te:>8}")


def save_splits(train_df, val_df, test_df, splits_dir):
    """Save train, val, test splits to CSV files."""
    train_path = os.path.join(splits_dir, 'train.csv')
    val_path   = os.path.join(splits_dir, 'val.csv')
    test_path  = os.path.join(splits_dir, 'test.csv')

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path,     index=False)
    test_df.to_csv(test_path,   index=False)

    print(f"\n── Splits saved ───────────────────────────────────")
    print(f"Train : {train_path}  ({len(train_df):,} samples)")
    print(f"Val   : {val_path}    ({len(val_df):,} samples)")
    print(f"Test  : {test_path}   ({len(test_df):,} samples)")
    print(f"Total : {len(train_df) + len(val_df) + len(test_df):,} samples")


def plot_split_distributions(train_df, val_df, test_df, reports_dir):
    """Visualize grade distribution across all three splits."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    splits     = [('Train', train_df), ('Val', val_df), ('Test', test_df)]
    colors     = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
    grade_labels = ['0\nNone', '1\nDoubtful', '2\nMild', '3+\nSevere']

    for ax, (split_name, split_df) in zip(axes, splits):
        counts = split_df['kl_grade_merged'].value_counts().sort_index()
        bars   = ax.bar(counts.index.astype(int), counts.values,
                        color=colors[:len(counts)],
                        edgecolor='white', width=0.6)

        ax.set_title(f'{split_name} ({len(split_df):,} samples)',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('KL Grade')
        ax.set_ylabel('Count')
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(grade_labels)

        for grade, count in counts.items():
            pct = count / len(split_df) * 100
            ax.text(int(grade), count + 20, f'{count}\n({pct:.1f}%)',
                    ha='center', fontsize=8, fontweight='bold')

    plt.suptitle('Grade Distribution Across Train / Val / Test Splits',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, 'split_distributions.png'),
                dpi=150, bbox_inches='tight')
    plt.show()
    print("Split distribution chart saved to reports/")


if __name__ == '__main__':

    # ── Paths ────────────────────────────────────────────────────────────────
    BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    SPLITS_DIR  = os.path.join(BASE_DIR, 'data', 'splits')
    REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

    # ── Run pipeline ─────────────────────────────────────────────────────────
    print("=" * 55)
    print("  SPLITTING PIPELINE")
    print("=" * 55)

    # Step 1 — Load
    df = load_master_dataset(SPLITS_DIR)

    # Step 2 — Split
    train_df, val_df, test_df = patient_level_split(df)

    # Step 3 — Verify no leakage
    all_clear = verify_no_leakage(train_df, val_df, test_df)
    if not all_clear:
        raise ValueError("Data leakage detected — do not proceed.")

    # Step 4 — Verify distributions
    verify_grade_distribution(train_df, val_df, test_df)
    verify_joint_distribution(train_df, val_df, test_df)

    # Step 5 — Save
    save_splits(train_df, val_df, test_df, SPLITS_DIR)

    # Step 6 — Plot
    plot_split_distributions(train_df, val_df, test_df, REPORTS_DIR)

    print("\n✓ Splitting pipeline complete.")