"""Tests for the MediaPipe hand-landmark detector and drawing helpers.

MediaPipe itself can only be verified end-to-end with a webcam, so these
tests exercise the parts that wrap it: the result-mapping logic and the
drawing code, using mock MediaPipe result objects.
"""

import numpy as np
import pytest

from app.ml.detector import (
    DetectionResult,
    DetectedHand,
    HandLandmarkDetector,
    Landmark,
    draw_hand_landmarks,
)


class _FakeLandmark:
    def __init__(self, x=0.1, y=0.2, z=0.3):
        self.x = x
        self.y = y
        self.z = z


class _FakeClassification:
    def __init__(self, label, score):
        self.label = label
        self.score = score


class _FakeClassList:
    def __init__(self, classification):
        self.classification = [classification]


class _FakeHandedness:
    def __init__(self, classification):
        self.classification = classification


def _make_fake_landmark_list():
    return type("LMList", (), {"landmark": [_FakeLandmark(i / 100.0, 0.1, 0.0) for i in range(21)]})


class _FakeResults:
    def __init__(self, n_hands=1):
        self.multi_hand_landmarks = [
            _make_fake_landmark_list() for _ in range(n_hands)
        ]
        self.multi_handedness = [
            _FakeHandedness([_FakeClassification("Left", 0.9)]),
            _FakeHandedness([_FakeClassification("Right", 0.8)]),
        ][:n_hands]


class _FakeResultsNoHands:
    multi_hand_landmarks = None
    multi_handedness = None


def test_no_hands_returns_empty_result():
    detector = HandLandmarkDetector(max_num_hands=1)
    try:
        result = detector._to_detection_result(_FakeResultsNoHands())
        assert result.hands == []
        assert not result
    finally:
        detector.close()


def test_single_hand_maps_21_landmarks():
    detector = HandLandmarkDetector(max_num_hands=1)
    try:
        result = detector._to_detection_result(_FakeResults(n_hands=1))
        assert len(result.hands) == 1
        hand = result.hands[0]
        assert len(hand.landmarks) == 21
        assert hand.handedness == "Left"
        assert hand.landmarks[0].x == pytest.approx(0.0)
    finally:
        detector.close()


def test_two_hands_supported_order_preserved():
    detector = HandLandmarkDetector(max_num_hands=2)
    try:
        result = detector._to_detection_result(_FakeResults(n_hands=2))
        assert len(result.hands) == 2
        assert [h.handedness for h in result.hands] == ["Left", "Right"]
    finally:
        detector.close()


def test_max_num_hands_validation():
    with pytest.raises(ValueError):
        HandLandmarkDetector(max_num_hands=0)


def test_process_handles_none_frame():
    detector = HandLandmarkDetector(max_num_hands=1)
    try:
        assert detector.process(None).hands == []
    finally:
        detector.close()


def test_draw_landmarks_mutates_and_returns_image():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    detection = DetectionResult(
        hands=[
            DetectedHand(
                landmarks=[Landmark(x=0.5, y=0.5, z=0.0)] * 21,
                handedness="Right",
                score=0.95,
            )
        ]
    )
    out = draw_hand_landmarks(img, detection)
    assert out is img
    assert img.sum() > 0  # something was drawn


def test_draw_landmarks_rejects_bad_shape():
    img = np.zeros((100, 100), dtype=np.uint8)
    detection = DetectionResult(hands=[])
    with pytest.raises(ValueError):
        draw_hand_landmarks(img, detection)