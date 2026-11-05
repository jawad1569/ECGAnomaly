"""
Script to extract a sample ECG CSV from test data for website testing.
Creates a single patient's ECG signal in the format expected by the API.
"""

import pandas as pd
import os
import random

# --- CONFIG ---
DATA_DIR = "data/raw"
# Using TEST files
SIGNAL_FILE = os.path.join(DATA_DIR, "test_signal.csv")
META_FILE = os.path.join(DATA_DIR, "test_meta.csv")
OUTPUT_FILE = "sample_upload_TEST_PATIENT_1.csv"


def create_sample_file():
    print(f"Loading data from {SIGNAL_FILE}...")

    # 1. Read just the Metadata first to pick a random patient
    if not os.path.exists(META_FILE):
        print("Error: Meta file not found.")
        return
    meta_df = pd.read_csv(META_FILE)

    # Compute target from label columns (same as training script)
    label_cols = ["NORM", "MI", "STTC", "CD", "HYP"]
    meta_df["target"] = meta_df[label_cols].idxmax(axis=1)

    # Pick a random patient ID
    random_index = random.randint(0, len(meta_df) - 1)
    target_ecg_id = meta_df.iloc[random_index]["ecg_id"]
    target_diagnosis = meta_df.iloc[random_index]["target"]  # "NORM", "MI", etc.

    print(f"Selected Patient ID: {target_ecg_id}")
    print(
        f"True Diagnosis Class: {target_diagnosis} (Keep this secret to test your model!)"
    )

    # 2. Extract that patient's signals
    print("Extracting signals (this might take 10-20 seconds)...")

    chunk_size = 50000
    found_signals = []

    for chunk in pd.read_csv(SIGNAL_FILE, chunksize=chunk_size):
        patient_data = chunk[chunk["ecg_id"] == target_ecg_id]
        if not patient_data.empty:
            found_signals.append(patient_data)

            # Check if we have enough rows (standard ECG length is ~1000 samples)
            if sum([len(df) for df in found_signals]) >= 1000:
                break

    if not found_signals:
        print("Error: Could not find signal data for this ID.")
        return

    # Combine found rows
    final_df = pd.concat(found_signals)

    # 3. Clean it for the "User Upload" format
    # Keep all columns as they are (including ecg_id for reference)
    print(f"Extracted {len(final_df)} rows for Patient {target_ecg_id}.")
    final_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅ SUCCESS! Created '{OUTPUT_FILE}'.")
    print("   -> Use this file to drag-and-drop onto your website.")
    print(f"   -> Expected Prediction: Class {target_diagnosis}")
    print(f"   -> File size: {os.path.getsize(OUTPUT_FILE) / 1024:.2f} KB")


if __name__ == "__main__":
    create_sample_file()
