"""Prepare and reformat the PhysioNet PTB-XL dataset into standard CSV files.

Reads the official raw PTB-XL dataset (WFDB format) from the dataset/ folder:
1. Maps SCP diagnostic statements into the 5 clinical superclasses (NORM, MI, STTC, CD, HYP).
2. Performs a stratified extraction across all 5 diagnostic superclasses.
3. Reads 12-lead, 100 Hz signal waveforms using wfdb.
4. Outputs standard train_meta.csv and train_signal.csv into data/raw/.
"""

import ast
import os
import numpy as np
import pandas as pd
import wfdb

DATASET_ROOT = os.path.join(
    "dataset", "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)
OUTPUT_DIR = os.path.join("data", "raw")
DATABASE_CSV = os.path.join(DATASET_ROOT, "ptbxl_database.csv")
SCP_CSV = os.path.join(DATASET_ROOT, "scp_statements.csv")

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def reformat_ptbxl(n_samples_per_class: int = 200, random_state: int = 42):
    """Reformat a stratified sample of real PTB-XL clinical ECGs into CSVs."""
    print("=" * 60)
    print("Reformatting Real PhysioNet PTB-XL Dataset")
    print("=" * 60)

    if not os.path.exists(DATABASE_CSV) or not os.path.exists(SCP_CSV):
        raise FileNotFoundError(f"PTB-XL database files not found in {DATASET_ROOT}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading PTB-XL database metadata...")
    db = pd.read_csv(DATABASE_CSV)
    agg_df = pd.read_csv(SCP_CSV, index_col=0)
    diag_map = agg_df[agg_df.diagnostic == 1]["diagnostic_class"].to_dict()

    def get_primary_class(scp_str):
        try:
            scp = ast.literal_eval(scp_str)
            for code in scp.keys():
                if code in diag_map:
                    return diag_map[code]
        except Exception:
            pass
        return None

    db["primary_class"] = db["scp_codes"].apply(get_primary_class)
    db_valid = db[db["primary_class"].isin(CLASSES)].copy()
    print(f"Total valid clinical recordings found: {len(db_valid)}")

    # Stratified sampling
    selected_records = []
    rng = np.random.default_rng(random_state)
    for cls in CLASSES:
        cls_df = db_valid[db_valid["primary_class"] == cls]
        n_take = min(n_samples_per_class, len(cls_df))
        chosen_indices = rng.choice(cls_df.index, size=n_take, replace=False)
        selected_records.append(cls_df.loc[chosen_indices])

    sampled_df = pd.concat(selected_records).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    print(f"Selected stratified cohort of {len(sampled_df)} patients:")
    for cls, count in sampled_df["primary_class"].value_counts().items():
        print(f"  {cls:6s}: {count} ECGs")

    # Build train_meta.csv
    meta_rows = []
    signal_chunks = []

    print("\nExtracting 12-lead signals via WFDB...")
    for idx, row in sampled_df.iterrows():
        ecg_id = int(row["ecg_id"])
        cls = row["primary_class"]
        filename_lr = row["filename_lr"]

        # Read actual physical signal array (1000 timesteps x 12 leads)
        record_path = os.path.join(DATASET_ROOT, filename_lr)
        sig, fields = wfdb.rdsamp(record_path)

        # Meta row
        meta_dict = {"ecg_id": ecg_id}
        for c in CLASSES:
            meta_dict[c] = 1 if c == cls else 0
        meta_rows.append(meta_dict)

        # Signal dataframe
        sig_df = pd.DataFrame(sig, columns=[f"channel-{i}" for i in range(12)])
        sig_df.insert(0, "ecg_id", ecg_id)
        signal_chunks.append(sig_df)

        if (idx + 1) % 100 == 0 or (idx + 1) == len(sampled_df):
            print(f"  Processed {idx + 1}/{len(sampled_df)} clinical ECG recordings...")

    meta_df = pd.DataFrame(meta_rows)
    meta_out = os.path.join(OUTPUT_DIR, "train_meta.csv")
    meta_df.to_csv(meta_out, index=False)
    print(f"\nWritten {meta_out}: {meta_df.shape}")

    full_signal_df = pd.concat(signal_chunks, ignore_index=True)
    signal_out = os.path.join(OUTPUT_DIR, "train_signal.csv")
    full_signal_df.to_csv(signal_out, index=False)
    print(f"Written {signal_out}: {full_signal_df.shape}")
    print("\nDataset preparation from real PTB-XL complete!")


if __name__ == "__main__":
    reformat_ptbxl(n_samples_per_class=200)
