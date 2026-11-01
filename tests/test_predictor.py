"""Tests for the real-time gesture predictor."""

import json
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from app.ml.preprocess import FEATURE_DIM_PER_HAND


@contextmanager
def model_override(predictor, probabilities):
    """Temporarily replace the loaded Keras model with a fake predictor."""
    fake = type(
        "FakeModel",
        (),
        {"predict": lambda self, x, verbose=0: np.array([probabilities])},
    )()
    original = predictor._model
    predictor._model = fake
    try:
        yield
    finally:
        predictor._model = original


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory):
    """Build a tiny real Keras model + label mapping on disk for tests."""
    tmp = tmp_path_factory.mktemp("predictor")
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(FEATURE_DIM_PER_HAND,)),
            tf.keras.layers.Dense(4, activation="softmax"),
        ]
    )
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
    # Fit so weights are "real".
    x = np.random.rand(200, FEATURE_DIM_PER_HAND).astype(np.float32)
    y = np.random.randint(0, 4, size=200)
    model.fit(x, y, epochs=1, verbose=0)
    model_path = tmp / "model.keras"
    model.save(model_path)

    mapping = {"HELLO": 0, "YES": 1, "NO": 2, "OK": 3}
    labels_path = tmp / "labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "label_to_index": mapping,
                "index_to_label": {str(v): k for k, v in mapping.items()},
            }
        ),
        encoding="utf-8",
    )
    return model_path, labels_path, mapping


def test_predictor_loads_and_reports_model_info(trained_model):
    from app.ml.predictor import GesturePredictor

    model_path, labels_path, mapping = trained_model
    predictor = GesturePredictor(model_path, labels_path, confidence_threshold=0.01)
    try:
        assert predictor.index_to_label == {0: "HELLO", 1: "YES", 2: "NO", 3: "OK"}
        assert predictor.label_to_index == mapping
    finally:
        predictor._model = None


def test_missing_model_raises():
    from app.ml.predictor import GesturePredictor, ModelNotLoadedError

    with pytest.raises(ModelNotLoadedError):
        GesturePredictor(
            Path("does/not/exist.keras"),
            Path("also/missing.json"),
            confidence_threshold=0.7,
        )


def test_confident_prediction_passes_high_threshold_does_not(trained_model):
    from app.ml.predictor import GesturePredictor

    model_path, labels_path, _ = trained_model
    # Build a "perfectly confident" one-hot-ish model is overkill; instead
    # verify both paths with a threshold of 0.5 (most outputs will clear it).
    predictor = GesturePredictor(model_path, labels_path, confidence_threshold=0.5)
    try:
        features = np.zeros(FEATURE_DIM_PER_HAND).astype(np.float32)
        with model_override(predictor, [0.6, 0.2, 0.1, 0.1]):
            assert predictor.predict_features(features) is not None
        with model_override(predictor, [0.2, 0.2, 0.3, 0.3]):
            assert predictor.predict_features(features) is None
    finally:
        predictor._model = None


def test_predict_features_shape_and_types(trained_model):
    from app.ml.predictor import GesturePredictor, Prediction

    model_path, labels_path, _ = trained_model
    predictor = GesturePredictor(model_path, labels_path, confidence_threshold=0.01)
    try:
        features = np.zeros(FEATURE_DIM_PER_HAND, dtype=np.float32)
        # Force a known distribution so assertions are deterministic.
        with model_override(predictor, [0.7, 0.1, 0.1, 0.1]):
            pred = predictor.predict_features(features)
        assert isinstance(pred, Prediction)
        assert pred.label == "HELLO"
        assert pred.confidence == pytest.approx(0.7)
        assert set(pred.probabilities.keys()) == {"HELLO", "YES", "NO", "OK"}
        assert abs(sum(pred.probabilities.values()) - 1.0) < 1e-5
    finally:
        predictor._model = None


def test_predict_hand_wires_preprocessing(trained_model):
    from app.ml.detector import DetectedHand, Landmark
    from app.ml.predictor import GesturePredictor

    model_path, labels_path, _ = trained_model
    predictor = GesturePredictor(model_path, labels_path, confidence_threshold=0.01)
    try:
        lm = [Landmark(x=i * 0.01, y=0.02 * i, z=0.0) for i in range(21)]
        hand = DetectedHand(landmarks=lm, handedness="Right", score=0.9)
        with model_override(predictor, [0.5, 0.2, 0.2, 0.1]):
            pred = predictor.predict_hand(hand)
        assert pred is not None and isinstance(pred.label, str)
    finally:
        predictor._model = None


def test_invalid_threshold_rejected(trained_model):
    from app.ml.predictor import GesturePredictor

    model_path, labels_path, _ = trained_model
    with pytest.raises(ValueError):
        GesturePredictor(model_path, labels_path, confidence_threshold=1.5)