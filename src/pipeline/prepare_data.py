"""Data preparation (step 1): PTB-XL -> validated feature matrix.

Separate from training. Reads the raw PTB-XL dataset (WFDB, 100 Hz) from dataset/,
and writes data/processed/ecg_features.npz (X, y, fold) plus validation_report.json.

Validation (per ECG and per dataset):
  * signal shape is (time, 12) with at least 100 samples
  * no missing (NaN / inf) values
  * amplitude within +/-50 mV
  * every record has a label in the 5 diagnostic superclasses
Invalid ECGs are dropped and counted in the report.

Split: official PTB-XL `strat_fold` (1-9 train, 10 held-out test), stored as `fold`.
Run:  python src/pipeline/prepare_data.py     (or: make data)
"""
import ast
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from src.utils.features import extract_features, validate_signal  # noqa: E402

ROOT = os.path.join("dataset", "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")
OUT_DIR = os.path.join("data", "processed")
OUT_NPZ = os.path.join(OUT_DIR, "ecg_features.npz")
OUT_REPORT = os.path.join(OUT_DIR, "validation_report.json")
CLASSES = sorted(["NORM", "MI", "STTC", "CD", "HYP"])


def _primary_label(scp_str, diag_map):
    """Diagnostic superclass with the highest likelihood, or None."""
    best, best_like = None, -1
    for code, like in ast.literal_eval(scp_str).items():
        if code in diag_map and like > best_like:
            best, best_like = diag_map[code], like
    return best


def _process(filename):
    import wfdb

    try:
        sig, _ = wfdb.rdsamp(os.path.join(ROOT, filename))
        validate_signal(sig)
        return extract_features(sig), None
    except Exception as exc:  # invalid or unreadable record
        return None, str(exc)[:80]


def prepare(force=False):
    if os.path.exists(OUT_NPZ) and not force:
        print(f"{OUT_NPZ} already exists (use force=True to rebuild)")
        return OUT_NPZ
    db = pd.read_csv(os.path.join(ROOT, "ptbxl_database.csv"))
    scp = pd.read_csv(os.path.join(ROOT, "scp_statements.csv"), index_col=0)
    diag_map = scp[scp.diagnostic == 1]["diagnostic_class"].to_dict()

    n_total = len(db)
    db["label"] = db["scp_codes"].apply(lambda s: _primary_label(s, diag_map))
    n_unlabelled = int(db["label"].isna().sum())
    db = db[db["label"].isin(CLASSES)].reset_index(drop=True)
    print(f"{len(db)} labelled of {n_total} records ({n_unlabelled} without diagnostic label)")

    t = time.time()
    results = Parallel(n_jobs=-1, batch_size=64)(delayed(_process)(f) for f in db.filename_lr)
    ok = np.array([r[0] is not None for r in results])
    errors = [r[1] for r in results if r[1]]
    X = np.array([r[0] for r in results if r[0] is not None])
    db = db[ok].reset_index(drop=True)
    y = db["label"].map({c: i for i, c in enumerate(CLASSES)}).values
    fold = db["strat_fold"].values
    print(f"features {X.shape} in {time.time() - t:.0f}s; dropped {len(errors)} invalid ECGs")

    # dataset-level checks
    assert not np.isnan(X).any() and np.isfinite(X).all(), "NaN/inf in feature matrix"
    report = {
        "records_total": n_total,
        "records_without_label": n_unlabelled,
        "records_invalid_signal": len(errors),
        "records_used": int(len(X)),
        "n_features": int(X.shape[1]),
        "class_counts": {c: int((y == i).sum()) for i, c in enumerate(CLASSES)},
        "train_records_folds_1_9": int((fold <= 9).sum()),
        "test_records_fold_10": int((fold == 10).sum()),
        "invalid_reasons": sorted(set(errors)),
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(OUT_NPZ, X=X, y=y, fold=fold, classes=np.array(CLASSES))
    with open(OUT_REPORT, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))
    return OUT_NPZ


def load_prepared():
    """Return X, y, fold, classes from the prepared file (builds it if missing)."""
    if not os.path.exists(OUT_NPZ):
        prepare()
    z = np.load(OUT_NPZ)
    return z["X"], z["y"], z["fold"], list(z["classes"])


if __name__ == "__main__":
    prepare(force="--force" in sys.argv)
