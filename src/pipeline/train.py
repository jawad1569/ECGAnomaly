"""ECG Anomaly Detection - Training Pipeline (Milestone 1, with FLAML AutoML).

1. Load PTB-XL (WFDB, 100 Hz), label = diagnostic superclass with highest likelihood.
2. Validate signals and extract 467 features per ECG (src/utils/features.py).
3. Official PTB-XL split: folds 1-9 train (CV), fold 10 held-out test.
4. FLAML AutoML search over lgbm / xgboost / extra_tree / rf (metric: macro-F1).
5. One child MLflow run per estimator family (3+ runs) + a parent run.
6. Register the winner as `ecg_model_v1` only if macro-F1 >= best registered.
"""
import ast
import os
import sys
import tempfile
import time
import warnings

sys.path.insert(0, os.getcwd())
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from flaml import AutoML  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, f1_score  # noqa: E402

from src.utils.features import extract_features, validate_signal  # noqa: E402,F401

warnings.filterwarnings("ignore")

ROOT = os.path.join("dataset", "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")
CACHE = os.path.join("data", "features_v2.npz")
CLASSES = sorted(["NORM", "MI", "STTC", "CD", "HYP"])
MLFLOW_EXPERIMENT = "ECG_AutoML_Phase1"
REGISTERED_MODEL_NAME = "ecg_model_v1"
AUTOML_TIME_BUDGET = int(os.getenv("AUTOML_TIME_BUDGET", "240"))
ESTIMATOR_LIST = ["lgbm", "xgboost", "extra_tree", "rf"]
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))


def load_and_process_data():
    """Return X, y (index into CLASSES), strat_fold. Features cached in data/."""
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        return z["X"], z["y"], z["fold"]
    import wfdb

    db = pd.read_csv(os.path.join(ROOT, "ptbxl_database.csv"))
    scp = pd.read_csv(os.path.join(ROOT, "scp_statements.csv"), index_col=0)
    dm = scp[scp.diagnostic == 1]["diagnostic_class"].to_dict()

    def label(s):
        best, bl = None, -1
        for k, v in ast.literal_eval(s).items():
            if k in dm and v > bl:
                best, bl = dm[k], v
        return best

    db["y"] = db["scp_codes"].apply(label)
    db = db[db.y.isin(CLASSES)].reset_index(drop=True)
    print("records:", len(db), db.y.value_counts().to_dict())

    def one(fn):
        sig, _ = wfdb.rdsamp(os.path.join(ROOT, fn))
        validate_signal(sig)
        return extract_features(sig)

    t = time.time()
    X = np.array(Parallel(n_jobs=-1, batch_size=64)(delayed(one)(f) for f in db.filename_lr))
    print("features", X.shape, f"{time.time() - t:.0f}s")
    y = db.y.map({c: i for i, c in enumerate(CLASSES)}).values
    np.savez(CACHE, X=X, y=y, fold=db.strat_fold.values)
    return X, y, db.strat_fold.values


def _best_registered_f1macro(name):
    try:
        c = mlflow.MlflowClient()
        vals = [
            mlflow.get_run(v.run_id).data.metrics.get("f1_macro")
            for v in c.search_model_versions(f"name='{name}'")
        ]
        vals = [v for v in vals if v is not None]
        return max(vals) if vals else 0.0
    except Exception:
        return 0.0


def _fit(X, y, estimators, budget):
    am = AutoML()
    am.fit(X, y, task="classification", metric="macro_f1", time_budget=budget,
           estimator_list=estimators, n_jobs=-1, eval_method="cv", n_splits=3,
           seed=42, verbose=0, mlflow_logging=False)
    return am


def main():
    X, y, fold = load_and_process_data()
    train, test = fold <= 9, fold == 10
    Xtr, ytr, Xte, yte = X[train], y[train], X[test], y[test]
    norm = CLASSES.index("NORM")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="automl_parent"):
        mlflow.log_params({"time_budget_s": AUTOML_TIME_BUDGET, "estimators": str(ESTIMATOR_LIST),
                           "n_train": len(Xtr), "n_test": len(Xte), "n_features": X.shape[1],
                           "split": "PTB-XL strat_fold 1-9 train / 10 test"})
        per_est = max(20, AUTOML_TIME_BUDGET // len(ESTIMATOR_LIST))
        for est in ESTIMATOR_LIST:
            with mlflow.start_run(run_name=f"trial_{est}", nested=True):
                am = _fit(Xtr, ytr, [est], per_est)
                p = am.predict(Xte)
                f1m, acc = f1_score(yte, p, average="macro"), accuracy_score(yte, p)
                mlflow.log_params({"estimator": est, "time_budget_s": per_est,
                                   "best_config": str(am.best_config)})
                mlflow.log_metrics({"f1_macro": f1m, "accuracy": acc})
                print(f"  [{est:10s}] macroF1={f1m:.4f} acc={acc:.4f}")

        print(f"\nFLAML AutoML search ({AUTOML_TIME_BUDGET}s over {ESTIMATOR_LIST}) ...")
        am = _fit(Xtr, ytr, ESTIMATOR_LIST, AUTOML_TIME_BUDGET)
        p = am.predict(Xte)
        acc, f1m = accuracy_score(yte, p), f1_score(yte, p, average="macro")
        f1ab = f1_score(yte != norm, p != norm)
        print(f"best={am.best_estimator} acc={acc:.4f} macroF1={f1m:.4f} F1abnormal={f1ab:.4f}")
        print(classification_report(yte, p, target_names=CLASSES))
        mlflow.log_param("best_estimator", am.best_estimator)
        mlflow.log_metrics({"accuracy": acc, "f1_macro": f1m, "f1_abnormal": f1ab})

        prev = _best_registered_f1macro(REGISTERED_MODEL_NAME)
        if f1m >= prev:
            model = am.model.estimator if hasattr(am.model, "estimator") else am.model
            mlflow.sklearn.log_model(model, "model", serialization_format="cloudpickle",
                                     registered_model_name=REGISTERED_MODEL_NAME)
            with tempfile.TemporaryDirectory() as d:
                pth = os.path.join(d, "class_names.txt")
                with open(pth, "w") as fh:
                    fh.write("\n".join(CLASSES))
                mlflow.log_artifact(pth)
            print(f"Registered '{REGISTERED_MODEL_NAME}' (macroF1 {f1m:.4f} >= {prev:.4f})")
        else:
            print(f"Not promoted: {f1m:.4f} < registered best {prev:.4f}")
        return f1m


if __name__ == "__main__":
    main()
