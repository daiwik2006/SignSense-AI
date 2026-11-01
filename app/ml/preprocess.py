"""Landmark normalization for gesture classification.

The exact pipeline used here is mirrored in the data-collection and training
scripts, so features produced during training are identical to those produced
during inference. See ``docs/model.md`` for the full description.

Single-hand feature vector (v1, 63 dimensions):

1. Take the 21 (x, y, z) MediaPipe landmarks for the detection's primary hand.
2. Subtract the wrist landmark (index 0) from every landmark so the hand is
   described relative to its own wrist (translation invariance).
3. Divide by the distance between the wrist (index 0) and the middle-finger MCP
   (index 9), a hand-scale measure (scale / distance invariance).
4. Flatten the resulting 21 x 3 array row-major into a single 63-d vector.

Every returned vector therefore describes *shape* of the hand, not its
position in the frame, its distance from the camera, or the image resolution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from app.ml.detector import DetectedHand

logger = logging.getLogger(__name__)

NUM_LANDMARKS = 21
COORDS_PER_LANDMARK = 3
FEATURE_DIM_PER_HAND = NUM_LANDMARKS * COORDS_PER_LANDMARK  # 63

ANCHOR_LANDMARK_INDEX = 0  # wrist
SCALE_LANDMARK_INDEX = 9  # middle finger MCP, used as scale reference

_MIN_SCALE = 1e-6


class PreprocessingError(ValueError):
    """Raised when landmarks cannot be converted into features."""


@dataclass
class PreprocessConfig:
    """Configuration for landmark normalization.

    Attributes
    ----------
    use_z:
        Include the depth component. MediaPipe's ``z`` is normalized and helps
        distinguish bent vs extended fingers, so it is enabled by default.
    anchor_index:
        Landmark index used as the origin for translation normalization.
    scale_index:
        Landmark index whose distance from the anchor defines the hand scale.
    max_hands:
        Number of hands expected by the model. ``1`` for the v1 static model.
    """

    use_z: bool = True
    anchor_index: int = ANCHOR_LANDMARK_INDEX
    scale_index: int = SCALE_LANDMARK_INDEX
    max_hands: int = 1


@dataclass
class Preprocessor:
    """Normalizes :class:`DetectedHand` objects into feature vectors.

    The instance holds a :class:`PreprocessConfig` that is also written to
    disk next to the label mapping so training and inference stay aligned.
    """

    config: PreprocessConfig = field(default_factory=PreprocessConfig)

    def _validate(self, landmarks: np.ndarray) -> None:
        if landmarks.shape != (NUM_LANDMARKS, COORDS_PER_LANDMARK):
            raise PreprocessingError(
                f"Expected landmarks of shape ({NUM_LANDMARKS}, "
                f"{COORDS_PER_LANDMARK}) but got {landmarks.shape}."
            )
        if not np.all(np.isfinite(landmarks)):
            raise PreprocessingError("Landmarks contain NaN or infinite values.")

    def normalize_single_hand(self, hand: DetectedHand) -> Optional[np.ndarray]:
        """Convert one detected hand into a (63,) feature vector.

        Returns ``None`` when the hand cannot be normalized (degenerate scale
        or malformed landmarks) so callers can skip it safely.
        """
        if hand is None or hand.landmarks is None:
            return None
        try:
            lm = np.array(
                [
                    [p.x, p.y, p.z if self.config.use_z else 0.0]
                    for p in hand.landmarks
                ],
                dtype=np.float32,
            )
        except (AttributeError, TypeError):
            raise PreprocessingError("Malformed landmark data.") from None
        self._validate(lm)

        anchor = lm[self.config.anchor_index]
        relative = lm - anchor

        scale_vec = lm[self.config.scale_index] - anchor
        scale = float(np.linalg.norm(scale_vec))
        if scale < _MIN_SCALE:
            logger.warning("Hand scale too small (%s); skipping hand.", scale)
            return None

        normalized = relative / scale
        return normalized.reshape(-1).astype(np.float32)

    def normalize_hands(
        self, hands: List[DetectedHand]
    ) -> Optional[np.ndarray]:
        """Normalize detected hands into the feature vector for the model.

        For ``max_hands == 1`` (v1) this returns the feature vector of the
        primary detected hand: MediaPipe reports hands in decreasing detection
        confidence order.

        For ``max_hands == 2`` (extension path) both hands are concatenated in
        a fixed order (Left then Right by handedness label) followed by a
        2-element presence indicator, and a missing hand is zero-filled. This
        deterministic ordering makes two-handed gestures representable without
        depending on detection order. The v1 model is trained on 63-dim
        vectors only; the two-hand format is not yet part of the trained model.
        """
        if not hands:
            return None

        if self.config.max_hands == 1:
            hand = hands[0]
            return self.normalize_single_hand(hand)

        ordered: List[Optional[np.ndarray]] = [None, None]
        for hand in hands:
            index = 0 if hand.handedness.upper() in ("LEFT", "UNKNOWN") else 1
            if ordered[index] is None:
                ordered[index] = self.normalize_single_hand(hand)

        parts: List[np.ndarray] = []
        for index in range(self.config.max_hands):
            vec = ordered[index]
            if vec is None:
                parts.append(np.zeros(FEATURE_DIM_PER_HAND, dtype=np.float32))
                parts.append(np.zeros(1, dtype=np.float32))
            else:
                parts.append(vec)
                parts.append(np.ones(1, dtype=np.float32))
        return np.concatenate(parts, axis=0).astype(np.float32)

    def feature_dim(self) -> int:
        """Return the feature-vector length consumed by the model."""
        if self.config.max_hands == 1:
            return FEATURE_DIM_PER_HAND
        return self.config.max_hands * (FEATURE_DIM_PER_HAND + 1)

    def to_dict(self) -> dict:
        """Serialize the configuration for storage with the model."""
        return {
            "use_z": self.config.use_z,
            "anchor_index": self.config.anchor_index,
            "scale_index": self.config.scale_index,
            "max_hands": self.config.max_hands,
            "feature_dim": self.feature_dim(),
        }