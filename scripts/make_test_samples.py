"""Export real, labelled single-ECG CSV files for testing the API and the web UI.

ECGs are taken from PTB-XL's held-out test fold (strat_fold 10), which the model never saw
during training. Only records whose diagnostic codes all belong to one superclass are used,
so each file has an unambiguous cardiologist label.

    python scripts/make_test_samples.py              # 2 ECGs per class -> samples/test_ecgs/
    python scripts/make_test_samples.py --per-class 4 --seed 7

Each file holds ONE recording: columns ecg_id, label, channel-0 ... channel-11 (leads I, II,
III, aVR, aVL, aVF, V1-V6), 1,000 rows (10 s at 100 Hz). The web UI reads the label column and
shows whether the model's prediction matches it. index.csv lists every file with the model's
prediction, so it also shows where the model is wrong (it is ~72% accurate on this fold).
"""

import argparse
import ast
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from src.pipeline.prepare_data import CLASSES, ROOT, _primary_label  # noqa: E402
from src.utils.features import validate_signal  # noqa: E402

OUT_DIR = os.path.join("samples", "test_ecgs")
CHANNELS = [f"channel-{i}" for i in range(12)]


def single_class(scp_str, diag_map):
    """True if every diagnostic code of the record maps to the same superclass."""
    classes = {diag_map[c] for c in ast.literal_eval(scp_str) if c in diag_map}
    return len(classes) == 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--per-class", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import wfdb

    db = pd.read_csv(os.path.join(ROOT, "ptbxl_database.csv"))
    scp = pd.read_csv(os.path.join(ROOT, "scp_statements.csv"), index_col=0)
    diag_map = scp[scp.diagnostic == 1]["diagnostic_class"].to_dict()

    test = db[db.strat_fold == 10].copy()
    test["label"] = test["scp_codes"].apply(lambda s: _primary_label(s, diag_map))
    test = test[test["label"].isin(CLASSES) & test["scp_codes"].apply(lambda s: single_class(s, diag_map))]

    os.environ.setdefault("MODEL_SOURCE", "file")
    from src.utils.model_loader import predict_ecg_signal

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for cls in ["NORM", "MI", "STTC", "CD", "HYP"]:
        picks = test[test.label == cls].sample(n=args.per_class, random_state=args.seed)
        for _, rec in picks.iterrows():
            sig, _ = wfdb.rdsamp(os.path.join(ROOT, rec.filename_lr))
            validate_signal(sig)
            ecg_id = int(rec.ecg_id)
            df = pd.DataFrame(np.round(sig, 4), columns=CHANNELS)
            df.insert(0, "label", cls)
            df.insert(0, "ecg_id", ecg_id)
            name = f"{cls}_{ecg_id}.csv"
            df.to_csv(os.path.join(OUT_DIR, name), index=False)

            pred = predict_ecg_signal(sig)
            rows.append({
                "file": name, "ecg_id": ecg_id, "label": cls, "predicted": pred["predicted_class"],
                "confidence": round(pred["confidence"], 3), "correct": pred["predicted_class"] == cls,
                "age": rec.age, "sex": "M" if rec.sex == 0 else "F",  # PTB-XL: 0 = male, 1 = female
            })

    index = pd.DataFrame(rows)
    index.to_csv(os.path.join(OUT_DIR, "index.csv"), index=False)
    print(index.to_string(index=False))
    print(f"\n{index.correct.sum()}/{len(index)} predicted correctly -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
