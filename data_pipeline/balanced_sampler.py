# data_pipeline/balanced_sampler.py
import pandas as pd
import numpy as np
import os


def create_balanced_splits(
    splits_dir,
    output_dir,
    target_per_class=1000,
    random_state=42
):
    """
    Create balanced train and val splits by oversampling
    minority classes and undersampling majority classes.

    Args:
        splits_dir     : path to original splits CSV files
        output_dir     : path to save balanced CSV files
        target_per_class: target number of samples per grade
        random_state   : random seed for reproducibility
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(random_state)

    for split in ['train', 'val']:
        csv_path = os.path.join(splits_dir, f'{split}.csv')
        df       = pd.read_csv(csv_path)

        print(f"\n── Balancing {split}.csv ───────────────────────")
        print(f"Original distribution:")
        print(df['kl_grade'].value_counts().sort_index())

        balanced_dfs = []

        for grade in range(5):
            grade_df = df[df['kl_grade'] == grade]
            n        = len(grade_df)

            if n >= target_per_class:
                # Undersample — randomly select target_per_class rows
                sampled = grade_df.sample(
                    n=target_per_class,
                    random_state=random_state
                )
            else:
                # Oversample — sample with replacement
                sampled = grade_df.sample(
                    n=target_per_class,
                    replace=True,
                    random_state=random_state
                )

            balanced_dfs.append(sampled)
            print(f"  Grade {grade}: {n:>5} → {target_per_class} samples")

        balanced_df = pd.concat(balanced_dfs).sample(
            frac=1,
            random_state=random_state
        ).reset_index(drop=True)

        out_path = os.path.join(output_dir, f'{split}_balanced.csv')
        balanced_df.to_csv(out_path, index=False)
        print(f"Saved to {out_path}")
        print(f"Total samples: {len(balanced_df)}")

    # Test split stays unchanged — never balance test set
    test_df = pd.read_csv(os.path.join(splits_dir, 'test.csv'))
    test_out = os.path.join(output_dir, 'test_balanced.csv')
    test_df.to_csv(test_out, index=False)
    print(f"\nTest split copied unchanged: {len(test_df)} samples")


if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    SPLITS_DIR = os.path.join(BASE_DIR, 'data', 'splits')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'splits', 'balanced')

    create_balanced_splits(
        splits_dir=SPLITS_DIR,
        output_dir=OUTPUT_DIR,
        target_per_class=1000
    )