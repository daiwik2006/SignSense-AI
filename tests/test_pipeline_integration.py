"""Cross-module integration test: the real trained model on the held-out test set.

End-to-end regression guard for the chain:

    data/processed features -> GesturePredictor (preprocess + model + labels)
    -> softmax -> predicted class index -> label string

A trained model is required; the test is skipped with a clear message when it
is missing (models are Git-ignored and built locally).
"""

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _paths():
    return (
        ROOT / "models" / "sign_classifier.keras",
        ROOT / "models" / "labels.json",
        ROOT / "data" / "processed",
    )


@pytest.fixture(scope="module")
def real_pipeline():
    model_path, labels_path, data_dir = _paths()
    if not (model_path.exists() and labels_path.exists()):
        pytest.skip("Trained model not present; run scripts/train_static_model.py first.")

    from app.ml.predictor import GesturePredictor

    # Low threshold: keep confidence filtering out of this test.
    # The threshold gate itself is covered in tests/test_predictor.py.
    predictor = GesturePredictor(model_path, labels_path, confidence_threshold=0.01)
    X = np.load(data_dir / "features_test.npy").astype(np.float32)
    y = np.load(data_dir / "labels_test.npy")
    return predictor, X, y


def test_real_model_accuracy_on_test_split(real_pipeline):
    predictor, X, y = real_pipeline
    pred = np.argmax(predictor._model.predict(X, verbose=0), axis=1)
    accuracy = float(np.mean(pred == y))
    assert accuracy >= 0.95


def test_every_trained_class_predicts_something(real_pipeline):
    predictor, X, _ = real_pipeline
    pred = np.argmax(predictor._model.predict(X, verbose=0), axis=1)
    expected = set(range(len(predictor.index_to_label)))
    assert set(int(i) for i in pred) == expected


def test_predict_features_end_to_end(real_pipeline):
    predictor, X, y = real_pipeline
    from app.ml.predictor import Prediction

    seen_labels = set()
    for features in X:
        pred = predictor.predict_features(features)
        assert pred is not None
        assert isinstance(pred, Prediction)
        assert 0.0 <= pred.confidence <= 1.0
        assert abs(sum(pred.probabilities.values()) - 1.0) < 1e-4
        assert pred.label in predictor.label_to_index
        seen_labels.add(pred.label)
    assert seen_labels == set(predictor.index_to_label.values())