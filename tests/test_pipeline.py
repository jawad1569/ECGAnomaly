import numpy as np
import pytest

from src.utils import model_loader
from src.utils.features import extract_features, validate_signal
from src.utils.model_loader import get_model, predict_ecg_signal


def test_feature_extraction():
    feats = extract_features(np.random.randn(1000, 12))
    assert feats.ndim == 1 and feats.shape[0] == 467
    assert np.isfinite(feats).all()


def test_validate_signal_ok():
    validate_signal(np.random.randn(1000, 12))


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((1000, 6)),
        np.zeros((10, 12)),
        np.full((1000, 12), np.nan),
        np.full((1000, 12), 99.0),
    ],
)
def test_validate_signal_rejects(bad):
    with pytest.raises(ValueError):
        validate_signal(bad)


def test_model_loaded():
    model, class_names = get_model()
    assert model is not None
    assert len(class_names) == 5


def test_model_prediction():
    result = predict_ecg_signal(np.random.randn(500, 12))
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["predicted_class"] in ["CD", "HYP", "MI", "NORM", "STTC"]


def test_latest_model_file_picks_newest(tmp_path):
    for name in ["ecg_model_20260101_000000_rf.joblib", "ecg_model_20261231_235959_lgbm.joblib"]:
        (tmp_path / name).write_bytes(b"")
    assert model_loader.latest_model_file(str(tmp_path)).endswith("20261231_235959_lgbm.joblib")
    assert model_loader.latest_model_file(str(tmp_path / "empty")) is None


def test_model_loads_from_file_without_mlflow(monkeypatch):
    """Docker/CI case: no MLflow registry, the committed joblib file is used."""
    monkeypatch.setenv("MODEL_SOURCE", "file")
    model_loader.reset_model_cache()
    try:
        model, class_names = get_model()
        assert model is not None
        assert class_names == ["CD", "HYP", "MI", "NORM", "STTC"]
        assert model_loader.get_model_version().startswith("ecg_model_")
    finally:
        model_loader.reset_model_cache()


def test_missing_model_gives_clear_error(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_SOURCE", "file")
    monkeypatch.setattr(model_loader, "MODEL_DIR", str(tmp_path))
    model_loader.reset_model_cache()
    try:
        with pytest.raises(RuntimeError, match="No ecg_model_"):
            get_model()
        assert get_model(raise_on_error=False) == (None, None)
    finally:
        model_loader.reset_model_cache()


def test_soft_voting_ensemble_averages_member_probabilities():
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    from src.utils.ensemble import SoftVotingEnsemble

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    members = [LogisticRegression(), DecisionTreeClassifier(max_depth=3)]
    ens = SoftVotingEnsemble(members, ["lr", "dt"]).fit(X, y)
    expected = np.mean([m.predict_proba(X) for m in ens.members_], axis=0)
    assert np.allclose(ens.predict_proba(X), expected)
    assert set(ens.predict(X)) <= {0, 1}


def test_promotion_gate_needs_accuracy_and_f1_not_worse():
    from src.pipeline.train import is_better

    served = {"accuracy": 0.724, "f1_macro": 0.633}
    assert is_better({"accuracy": 0.728, "f1_macro": 0.640}, served)
    assert not is_better({"accuracy": 0.730, "f1_macro": 0.620}, served)  # F1 worse
    assert not is_better({"accuracy": 0.720, "f1_macro": 0.650}, served)  # accuracy worse
    assert is_better({"accuracy": 0.5, "f1_macro": 0.5}, None)            # nothing served yet
