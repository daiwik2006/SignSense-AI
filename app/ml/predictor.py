"""Real-time gesture prediction using the trained classifier.

Loads the model and label mapping once and keeps them in memory, so repeated
calls per frame do not re-read or re-load the model. Rendering and the webcam
loop stay outside this module.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf

from app.ml.detector import DetectedHand, DetectionResult
from app.ml.preprocess import PreprocessConfig, Preprocessor

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """A single-frame classification result."""

    label: str
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=dict)


class ModelNotLoadedError(RuntimeError):
    """Raised when a trained model cannot be located or loaded."""


class GesturePredictor:
    """Wraps the Keras classifier together with landmark preprocessing.

    Parameters
    ----------
    model_path:
        Path to the ``.keras`` model artifact.
    labels_path:
        Path to the ``labels.json`` produced by training/preprocessing.
    confidence_threshold:
        Minimum top-class probability for a frame to be considered meaningful.
        Below this threshold the frame yields ``None`` (treated as "unknown").
    """

    def __init__(
        self,
        model_path: str | Path,
        labels_path: str | Path,
        confidence_threshold: float = 0.7,
        max_hands: int = 1,
    ) -> None:
        if not Path(model_path).exists():
            raise ModelNotLoadedError(
                f"Model not found at {model_path}. Train a static model first "
                "with:\n  python scripts/preprocess_data.py\n  "
                "python scripts/train_static_model.py"
            )
        if not Path(labels_path).exists():
            raise ModelNotLoadedError(
                f"Label mapping not found at {labels_path}. Run preprocessing first."
            )
        if not (0.0 < confidence_threshold <= 1.0):
            raise ValueError("confidence_threshold must be in (0, 1].")

        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        self.confidence_threshold = confidence_threshold
        self.preprocessor = Preprocessor(PreprocessConfig(max_hands=max_hands))

        self._model = tf.keras.models.load_model(self.model_path)
        with open(self.labels_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.index_to_label: Dict[int, str] = {
            int(k): v for k, v in data["index_to_label"].items()
        }
        self.label_to_index: Dict[str, int] = {
            k: int(v) for k, v in data["label_to_index"].items()
        }
        expected_dim = self.preprocessor.feature_dim()
        expected_input = self._expected_input_dim()
        if expected_input != expected_dim:
            raise ModelNotLoadedError(
                f"Model expects {expected_input}-dim input but preprocessor "
                f"produces {expected_dim}-dim features."
            )
        logger.info(
            "Predictor ready: %d classes, threshold=%.2f",
            len(self.index_to_label),
            self.confidence_threshold,
        )

    def _expected_input_dim(self) -> int:
        """Determine the model's expected feature dimension robustly.

        Keras 3 models loaded from ``.keras`` do not always expose
        ``model.input`` (it errors when the model is a fresh load). The input
        shape property and the first layer's input tensor are used instead.
        """
        try:
            shape = self._model.input_shape
            return int(shape[-1])
        except (AttributeError, TypeError):
            pass
        for layer in self._model.layers:
            try:
                return int(layer.input.shape[-1])
            except (AttributeError, TypeError):
                continue
        raise ModelNotLoadedError("Could not determine the model's input dimension.")

    def predict_features(self, features: np.ndarray) -> Optional[Prediction]:
        """Classify one preprocessed feature vector.

        Returns ``None`` when confidence is below the threshold.
        """
        if features is None:
            return None
        x = np.asarray(features, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        probs = self._model.predict(x, verbose=0)[0]
        pred_index = int(np.argmax(probs))
        confidence = float(probs[pred_index])
        if confidence < self.confidence_threshold:
            return None
        label = self.index_to_label.get(pred_index, "UNKNOWN")
        return Prediction(
            label=label,
            confidence=confidence,
            probabilities={
                self.index_to_label[i]: float(p) for i, p in enumerate(probs)
            },
        )

    def predict_hand(self, hand: DetectedHand) -> Optional[Prediction]:
        """Preprocess one detected hand and classify it."""
        features = self.preprocessor.normalize_single_hand(hand)
        return self.predict_features(features)

    def predict_hands(self, hands: List[DetectedHand]) -> Optional[Prediction]:
        """Preprocess detected hands and classify the frame."""
        features = self.preprocessor.normalize_hands(hands)
        return self.predict_features(features)

    def predict_detection(self, detection: DetectionResult) -> Optional[Prediction]:
        """Classify a frame's detection result (hands)."""
        return self.predict_hands(detection.hands)