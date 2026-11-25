"""ECG Anomaly Detection - Training Pipeline (FLAML AutoML + validated ensemble).

1. Load the prepared, validated feature matrix (src/pipeline/prepare_data.py).
2. Official PTB-XL split: folds 1-9 train, fold 10 held-out test (used once, for the final score).
3. FLAML AutoML search over lgbm / xgboost / extra_tree / rf (metric: macro-F1, 3-fold CV),
   with one child MLflow run per estimator family plus a parent run.
4. Ensemble selection on the PTB-XL validation fold: the AutoML model, a HistGradientBoosting
   model and a logistic regression are fit on folds 1-8 and scored on fold 9; the combination
   (soft-voting average of probabilities) with the best fold-9 accuracy is refit on folds 1-9.
   Fold 10 plays no part in this choice.
5. Score the final model on fold 10 and save it as models/ecg_model_<timestamp>_<tag>.joblib
   with a *_metrics.json next to it.
6. Promotion gate: the new model replaces the served one (models/) and is registered as
   `ecg_model_v1` in MLflow only if its accuracy AND macro-F1 are at least those of the model
   currently served. Otherwise it is kept in models/rejected/ for inspection. The superseded
   model moves to models/archive/ (older versions also stay in git history and old images).
"""
import glob
import itertools
import json
import os
import shutil
import sys
import tempfile
import time
import warnings

sys.path.insert(0, os.getcwd())
import joblib  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
from flaml import AutoML  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, f1_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.pipeline.prepare_data import load_prepared  # noqa: E402
from src.utils.ensemble import SoftVotingEnsemble  # noqa: E402
from src.utils.features import extract_features  # noqa: E402,F401

warnings.filterwarnings("ignore")

MODEL_DIR = "models"
REJECTED_DIR = os.path.join(MODEL_DIR, "rejected")
ARCHIVE_DIR = os.path.join(MODEL_DIR, "archive")
MLFLOW_EXPERIMENT = "ECG_AutoML_Phase1"
REGISTERED_MODEL_NAME = "ecg_model_v1"
AUTOML_TIME_BUDGET = int(os.getenv("AUTOML_TIME_BUDGET", "240"))
ESTIMATOR_LIST = ["lgbm", "xgboost", "extra_tree", "rf"]
VAL_FOLD = 9
LABELS = {"lgbm": "LightGBM", "xgboost": "XGBoost", "extra_tree": "Extra trees", "rf": "Random forest",
          "hgb": "HistGradientBoosting", "logreg": "Logistic regression"}
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))


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


def served_metrics(model_dir=MODEL_DIR):
    """Metrics of the model the API serves (newest models/ecg_model_*_metrics.json), or None."""
    files = sorted(glob.glob(os.path.join(model_dir, "ecg_model_*_metrics.json")))
    if not files:
        return None
    with open(files[-1], encoding="utf-8") as fh:
        return json.load(fh)


def is_better(new, current):
    """Promotion rule: accuracy and macro-F1 must both be at least the served model's."""
    if current is None:
        return True
    return new["accuracy"] >= current.get("accuracy", 0) and new["f1_macro"] >= current.get("f1_macro", 0)


def _fit(X, y, estimators, budget):
    am = AutoML()
    am.fit(X, y, task="classification", metric="macro_f1", time_budget=budget,
           estimator_list=estimators, n_jobs=-1, eval_method="cv", n_splits=3,
           seed=42, verbose=0, mlflow_logging=False)
    return am


def ensemble_candidates(automl_model, automl_name):
    """Diverse members, all runnable with the API's runtime dependencies."""
    return {
        automl_name: clone(automl_model),
        "hgb": HistGradientBoostingClassifier(max_iter=600, learning_rate=0.05, max_leaf_nodes=48,
                                              l2_regularization=1.0, early_stopping=True, random_state=42),
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)),
    }


def select_on_validation(candidates, X, y, fold):
    """Fit each candidate on folds 1-8, return the member combination with the best fold-9 accuracy."""
    tr, va = fold < VAL_FOLD, fold == VAL_FOLD
    probs, results = {}, []
    for name, model in candidates.items():
        probs[name] = clone(model).fit(X[tr], y[tr]).predict_proba(X[va])
    for k in range(1, len(candidates) + 1):
        for combo in itertools.combinations(candidates, k):
            p = np.mean([probs[c] for c in combo], axis=0).argmax(1)
            results.append((combo, accuracy_score(y[va], p), f1_score(y[va], p, average="macro")))
    results.sort(key=lambda r: (r[1], r[2]), reverse=True)
    return results


def main():
    X, y, fold, CLASSES = load_prepared()
    train, test = fold <= 9, fold == 10
    Xtr, ytr, Xte, yte = X[train], y[train], X[test], y[test]
    norm = CLASSES.index("NORM")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="automl_parent"):
        mlflow.log_params({"time_budget_s": AUTOML_TIME_BUDGET, "estimators": str(ESTIMATOR_LIST),
                           "n_train": len(Xtr), "n_test": len(Xte), "n_features": X.shape[1],
                           "split": "PTB-XL strat_fold 1-9 train / 10 test; ensemble chosen on fold 9"})
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
        automl_model = am.model.estimator if hasattr(am.model, "estimator") else am.model
        p = am.predict(Xte)
        print(f"AutoML best={am.best_estimator} acc={accuracy_score(yte, p):.4f} "
              f"macroF1={f1_score(yte, p, average='macro'):.4f}")
        mlflow.log_param("automl_best_estimator", am.best_estimator)

        # ensemble selection on the validation fold (fold 10 is not used here)
        print(f"\nEnsemble selection on fold {VAL_FOLD} ...")
        with mlflow.start_run(run_name="ensemble_selection", nested=True):
            candidates = ensemble_candidates(automl_model, am.best_estimator)
            ranking = select_on_validation(candidates, X, y, fold)
            for combo, vacc, vf1 in ranking:
                print(f"  val {'+'.join(combo):28s} acc={vacc:.4f} macroF1={vf1:.4f}")
            best_combo, val_acc, val_f1 = ranking[0]
            mlflow.log_params({"candidates": "+".join(candidates), "chosen": "+".join(best_combo),
                               "selection": f"best accuracy on PTB-XL fold {VAL_FOLD}"})
            mlflow.log_metrics({"val_accuracy": val_acc, "val_f1_macro": val_f1})

        if len(best_combo) == 1:
            model = clone(candidates[best_combo[0]]).fit(Xtr, ytr)
            tag, label = best_combo[0], LABELS.get(best_combo[0], best_combo[0])
        else:
            model = SoftVotingEnsemble([candidates[c] for c in best_combo], list(best_combo)).fit(Xtr, ytr)
            tag = "ensemble"
            label = "Ensemble (" + " + ".join(LABELS.get(c, c) for c in best_combo) + ")"

        p = model.predict(Xte)
        acc, f1m = accuracy_score(yte, p), f1_score(yte, p, average="macro")
        f1ab = f1_score(yte != norm, p != norm)
        print(f"\nfinal={label} acc={acc:.4f} macroF1={f1m:.4f} F1abnormal={f1ab:.4f}")
        print(classification_report(yte, p, target_names=CLASSES))
        mlflow.log_params({"best_estimator": tag, "final_model": label})
        mlflow.log_metrics({"accuracy": acc, "f1_macro": f1m, "f1_abnormal": f1ab,
                            "val_accuracy": val_acc, "val_f1_macro": val_f1})

        # save with a clear, versioned name; promote only if not worse than the served model
        stamp = time.strftime("%Y%m%d_%H%M%S")
        metrics = {"best_estimator": tag, "estimator_label": label, "members": list(best_combo),
                   "accuracy": acc, "f1_macro": f1m, "f1_abnormal": f1ab, "n_test": int(len(Xte)),
                   "val_accuracy": val_acc, "val_f1_macro": val_f1,
                   "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "mlflow_run_id": mlflow.active_run().info.run_id}
        current = served_metrics()
        promote = is_better(metrics, current)
        target = MODEL_DIR if promote else REJECTED_DIR
        os.makedirs(target, exist_ok=True)
        model_path = os.path.join(target, f"ecg_model_{stamp}_{tag}.joblib")
        if promote:  # move the superseded served model out of the way first
            os.makedirs(ARCHIVE_DIR, exist_ok=True)
            for old in glob.glob(os.path.join(MODEL_DIR, "ecg_model_*")):
                shutil.move(old, ARCHIVE_DIR)
        joblib.dump({"model": model, "classes": CLASSES}, model_path)
        with open(os.path.join(target, f"ecg_model_{stamp}_metrics.json"), "w") as fh:
            json.dump(metrics, fh, indent=2)
        mlflow.log_artifact(model_path)
        mlflow.log_param("promoted", promote)

        if current:
            print(f"served model: acc={current['accuracy']:.4f} macroF1={current['f1_macro']:.4f}")
        if not promote:
            print(f"NOT promoted (worse than the served model): kept in {model_path}")
            return metrics

        print(f"Promoted: {model_path} is now the served model")
        prev = _best_registered_f1macro(REGISTERED_MODEL_NAME)
        if f1m >= prev:
            mlflow.sklearn.log_model(model, "model", serialization_format="cloudpickle",
                                     registered_model_name=REGISTERED_MODEL_NAME)
            with tempfile.TemporaryDirectory() as d:
                pth = os.path.join(d, "class_names.txt")
                with open(pth, "w") as fh:
                    fh.write("\n".join(CLASSES))
                mlflow.log_artifact(pth)
            print(f"Registered '{REGISTERED_MODEL_NAME}' (macroF1 {f1m:.4f} >= {prev:.4f})")
        return metrics


if __name__ == "__main__":
    main()
