"""MediaPipe hand-landmark detection.

This module isolates all MediaPipe interaction so the rest of the codebase
never touches MediaPipe directly. Detection is kept separate from the webcam
loop: a camera frame is produced elsewhere and passed into
:meth:`HandLandmarkDetector.process`.

The implementation is hand-agnostic and supports an arbitrary number of hands
(the configured limit). Downstream code can therefore consume
:class:`DetectedHand` objects for one-hand or two-hand pipelines.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions import hands as mp_hands

logger = logging.getLogger(__name__)


@dataclass
class Landmark:
    """A single 3D hand landmark in normalized coordinates."""

    x: float
    y: float
    z: float


@dataclass
class DetectedHand:
    """A detected hand with its 21 landmarks and handedness label."""

    landmarks: List[Landmark]  # exactly 21 landmarks, ordered as MediaPipe defines
    handedness: str  # "Left" or "Right"
    score: float  # MediaPipe handedness classification confidence


@dataclass
class DetectionResult:
    """Container for all hands detected in a single frame."""

    hands: List[DetectedHand]

    def __bool__(self) -> bool:
        return bool(self.hands)


class HandLandmarkDetector:
    """Wrapper around the MediaPipe ``Hands`` solution.

    Parameters
    ----------
    max_num_hands:
        Maximum number of hands to detect in one frame. Set to ``2`` when
        two-handed signs are enabled.
    min_detection_confidence:
        Minimum confidence for the initial palm/hand detection.
    min_tracking_confidence:
        Minimum confidence to keep tracking an already-detected hand.
    static_image_mode:
        Use ``True`` for still images; ``False`` for video streams (enables
        tracking which is faster).
    model_complexity:
        0 (light), 1 (full), or 2 (heavy). Keep 1 for a balance of quality and
        CPU speed.
    """

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
        model_complexity: int = 1,
    ) -> None:
        if max_num_hands < 1:
            raise ValueError("max_num_hands must be at least 1.")
        self.max_num_hands = max_num_hands
        self._hands = mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
        )
        logger.debug("Initialized HandLandmarkDetector (max hands=%s)", max_num_hands)

    def process(self, image_bgr: np.ndarray) -> DetectionResult:
        """Detect hands in one BGR frame and return normalized landmarks.

        The image is converted to RGB internally because MediaPipe expects RGB.
        The caller passes the raw frame as-is, preserving the usual webcam
        convention of BGR frames.
        """
        if image_bgr is None:
            return DetectionResult(hands=[])
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False  # speed optimization
        try:
            results = self._hands.process(rgb)
        finally:
            rgb.flags.writeable = True

        return self._to_detection_result(results)

    def _to_detection_result(self, results: object) -> DetectionResult:
        hands: List[DetectedHand] = []
        if results.multi_hand_landmarks is None:
            return DetectionResult(hands=hands)

        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            landmarks = [
                Landmark(x=lm.x, y=lm.y, z=lm.z) for lm in hand_landmarks.landmark
            ]
            handedness: str = "Unknown"
            score: float = 0.0
            if results.multi_handedness is not None and idx < len(
                results.multi_handedness
            ):
                label = results.multi_handedness[idx].classification[0]
                handedness = str(label.label)
                score = float(label.score)
            hands.append(
                DetectedHand(landmarks=landmarks, handedness=handedness, score=score)
            )
        return DetectionResult(hands=hands)

    def close(self) -> None:
        """Release MediaPipe resources."""
        if hasattr(self._hands, "close"):
            self._hands.close()
        logger.debug("Closed HandLandmarkDetector")

    def __enter__(self) -> "HandLandmarkDetector":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# MediaPipe hand connection pairs for drawing.
_HAND_CONNECTIONS = tuple(mp_hands.HAND_CONNECTIONS)

_CIRCLE_CENTER = (0, 4, 8, 12, 16, 20)  # landmark indices drawn as circles


def draw_hand_landmarks(
    image_bgr: np.ndarray,
    detection: DetectionResult,
    *,
    landmark_color: Tuple[int, int, int] = (0, 255, 0),
    connection_color: Tuple[int, int, int] = (0, 165, 255),
) -> np.ndarray:
    """Draw landmarks, connections, and handedness labels onto a BGR image.

    The input image is modified in place (OpenCV convention) and also
    returned for convenience.
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must be a 3-channel BGR image.")

    h, w, _ = image_bgr.shape

    for hand in detection.hands:
        pts: List[Tuple[int, int]] = []
        for lm in hand.landmarks:
            px = int(lm.x * w)
            py = int(lm.y * h)
            pts.append((min(max(px, 0), w - 1), min(max(py, 0), h - 1)))

        # Connections
        for a, b in _HAND_CONNECTIONS:
            cv2.line(image_bgr, pts[a], pts[b], connection_color, 2, cv2.LINE_AA)

        # Landmark dots (finger tips slightly larger)
        for i, (px, py) in enumerate(pts):
            radius = 6 if i in _CIRCLE_CENTER else 4
            cv2.circle(
                image_bgr, (px, py), radius, landmark_color, -1, cv2.LINE_AA
            )

        # Handedness label
        label = f"{hand.handedness} ({hand.score:.2f})"
        cv2.putText(
            image_bgr,
            label,
            (pts[0][0], max(pts[0][1] - 12, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return image_bgr