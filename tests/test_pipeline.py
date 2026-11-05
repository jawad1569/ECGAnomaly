import numpy as np
import pytest
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
