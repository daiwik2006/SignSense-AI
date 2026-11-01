"""Tests for landmark normalization (preprocessing)."""

import numpy as np
import pytest

from app.ml.detector import DetectedHand, HandLandmarkDetector, Landmark
from app.ml.preprocess import (
    FEATURE_DIM_PER_HAND,
    PreprocessConfig,
    Preprocessor,
    PreprocessingError,
)


def _make_hand(landmarks=None, handedness="Right", score=0.9):
    if landmarks is None:
        landmarks = [
            Landmark(x=i * 0.01, y=i * 0.02, z=-0.0) for i in range(21)
        ]
    return DetectedHand(landmarks=landmarks, handedness=handedness, score=score)


def test_feature_vec_is_wrist_relative_and_size_63():
    lm = [Landmark(x=1.0 + i, y=5.0 + i, z=0.5) for i in range(21)]
    hand = _make_hand(lm)
    vec = Preprocessor().normalize_single_hand(hand)
    assert vec is not None
    assert vec.shape == (FEATURE_DIM_PER_HAND,)
    assert np.allclose(vec[:3], [0.0, 0.0, 0.0])  # wrist maps to origin


def test_scale_invariance():
    # Same finger configuration at two scales must produce similar vectors.
    def hand_at_scale(scale):
        lm = []
        for i in range(21):
            lm.append(Landmark(x=scale * (i + 1), y=scale * (2 * i), z=0.0))
        return _make_hand(lm)

    p = Preprocessor()
    a = p.normalize_single_hand(hand_at_scale(1.0))
    b = p.normalize_single_hand(hand_at_scale(5.0))
    assert a is not None and b is not None
    assert np.allclose(a, b, atol=1e-5)


def test_translation_invariance():
    base = [Landmark(x=0.1 + 0.02 * i, y=0.2, z=0.05) for i in range(21)]
    shifted = [Landmark(x=0.9 + 0.02 * i, y=0.8, z=0.95) for i in range(21)]
    p = Preprocessor()
    a = p.normalize_single_hand(_make_hand(base))
    b = p.normalize_single_hand(_make_hand(shifted))
    assert a is not None and b is not None
    assert np.allclose(a, b, atol=1e-6)


def test_returned_vector_is_unit_scaled():
    # landmark 9 (middle MCP) should be at unit distance from wrist after norm.
    lm = [Landmark(x=0.0, y=0.0, z=0.0) for _ in range(21)]
    # wrist at (0,0,0); middle MCP (index 9) moved to (0, 3, 0)
    lm[9] = Landmark(x=0.0, y=3.0, z=0.0)
    lm[8] = Landmark(x=1.0, y=1.0, z=0.0)
    vec = Preprocessor().normalize_single_hand(_make_hand(lm))
    assert vec is not None
    mcp9 = vec[9 * 3 : 9 * 3 + 3]
    assert np.linalg.norm(mcp9) == pytest.approx(1.0, abs=1e-6)


def test_degenerate_scale_returns_none():
    all_zero = [Landmark(x=0.0, y=0.0, z=0.0) for _ in range(21)]
    assert Preprocessor().normalize_single_hand(_make_hand(all_zero)) is None


def test_no_hands_returns_none():
    assert Preprocessor().normalize_hands([]) is None


def test_primary_hand_used_for_single_hand():
    lm = [Landmark(x=0.01 * i, y=0.02 * i, z=0.0) for i in range(21)]
    hand = _make_hand(lm, handedness="Left")
    vec = Preprocessor().normalize_hands([hand])
    assert vec.shape == (FEATURE_DIM_PER_HAND,)
    assert vec[:3] == pytest.approx([0.0, 0.0, 0.0])


def test_use_z_false_zeroes_z():
    lm = [Landmark(x=i * 0.01, y=0.0, z=0.02 * i) for i in range(21)]
    p_no_z = Preprocessor(PreprocessConfig(use_z=False))
    p_z = Preprocessor(PreprocessConfig(use_z=True))
    a = p_no_z.normalize_single_hand(_make_hand(lm))
    b = p_z.normalize_single_hand(_make_hand(lm))
    assert a is not None and b is not None
    assert np.allclose(a[2::3], 0.0)  # z features are zero when disabled
    assert not np.allclose(b[2::3], 0.0)  # varied z is preserved when enabled


def test_malformed_landmark_count_raises():
    hand = _make_hand([Landmark(x=0.0, y=0.0, z=0.0) for _ in range(5)])
    with pytest.raises(PreprocessingError):
        Preprocessor().normalize_single_hand(hand)


def test_two_hand_feature_order_and_missing_fill():
    lm = [Landmark(x=0.01 * i, y=0.02 * i, z=0.0) for i in range(21)]
    right = _make_hand(lm, handedness="Right")
    # Left-handed vector differs from right (mirror on x)
    lm2 = [Landmark(x=-0.01 * i, y=0.02 * i, z=0.0) for i in range(21)]
    left = _make_hand(lm2, handedness="Left")

    p = Preprocessor(PreprocessConfig(max_hands=2))
    vec = p.normalize_hands([right, left])
    assert vec is not None
    assert vec.shape == (2 * (FEATURE_DIM_PER_HAND + 1),)
    # Left features first (presence flag 1), then right features (flag 1).
    assert vec[FEATURE_DIM_PER_HAND] == pytest.approx(1.0)
    assert vec[-1] == pytest.approx(1.0)
    # Only a left hand: left flag present, right block zero-filled (deterministic).
    vec2 = p.normalize_hands([left])
    assert vec2 is not None
    assert vec2[FEATURE_DIM_PER_HAND] == pytest.approx(1.0)  # left present
    assert vec2[-1] == pytest.approx(0.0)  # right absent -> zero fill flag
    # The right-hand feature block is zero-filled.
    right_block = vec2[FEATURE_DIM_PER_HAND + 1 : 2 * FEATURE_DIM_PER_HAND]
    assert np.allclose(right_block, 0.0)


def test_feature_dim_reflects_config():
    assert Preprocessor().feature_dim() == FEATURE_DIM_PER_HAND
    assert (
        Preprocessor(PreprocessConfig(max_hands=2)).feature_dim()
        == 2 * (FEATURE_DIM_PER_HAND + 1)
    )


def test_to_dict_serializable():
    d = Preprocessor().to_dict()
    assert d["feature_dim"] == FEATURE_DIM_PER_HAND
    assert set(["use_z", "anchor_index", "scale_index", "max_hands"]) <= set(d)


def test_integration_with_real_detector_process():
    import numpy as np

    detector = HandLandmarkDetector(max_num_hands=1)
    try:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        detection = detector.process(blank)
        assert Preprocessor().normalize_hands(detection.hands) is None
    finally:
        detector.close()