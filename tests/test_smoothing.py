"""Tests for the temporal smoothing/debounce logic."""

import pytest

from app.ml.smoothing import AcceptedPrediction, SmoothingConfig, TemporalSmoothing


def test_holds_steady_sign_emit_single_token():
    s = TemporalSmoothing(SmoothingConfig(window_size=5, min_stability=0.6, cooldown_frames=10))
    accepted = []
    for _ in range(30):  # hold HELLO for 30 frames
        token = s.process("HELLO", 0.9)
        if token is not None:
            accepted.append(token.label)
    assert accepted.count("HELLO") == 1
    assert len(accepted) == 1


def test_change_of_sign_produces_new_token():
    s = TemporalSmoothing(SmoothingConfig(window_size=5, min_stability=0.6, cooldown_frames=10))
    first = None
    for _ in range(10):
        token = s.process("HELLO", 0.9)
        first = token or first
    assert first is not None and first.label == "HELLO"

    for _ in range(10):  # keep holding HELLO
        s.process("HELLO", 0.9)

    second = None
    for _ in range(10):  # switch to YES
        token = s.process("YES", 0.9)
        second = token or second
    assert second is not None and second.label == "YES"


def test_low_confidence_is_ignored():
    s = TemporalSmoothing(SmoothingConfig(window_size=3, min_stability=0.66, min_confidence=0.7))
    s.process("HELLO", 0.3)
    s.process("HELLO", 0.2)
    s.process("HELLO", 0.1)
    # confidence below threshold every frame -> "None" dropped from window
    assert s.process("HELLO", 0.5) is None
    assert s._dominant() is None


def test_requires_majority():
    s = TemporalSmoothing(SmoothingConfig(window_size=4, min_stability=0.75))
    s.process("A", 0.9)
    s.process("A", 0.9)
    s.process("B", 0.9)
    s.process("B", 0.9)
    assert s.process("A", 0.9) is None  # tie -> not stable

    t = TemporalSmoothing(SmoothingConfig(window_size=4, min_stability=0.5, min_confidence=0.01))
    t.process("A", 0.9)
    t.process("A", 0.9)  # accepts A at this point
    t.process("B", 0.9)  # suppressed while A still dominates window
    token = t.process("B", 0.9)
    token = t.process("B", 0.9)  # B becomes dominant -> accepted
    assert token is not None and token.label == "B"


def test_release_then_reshow_allows_repeat():
    s = TemporalSmoothing(SmoothingConfig(window_size=3, min_stability=0.66, cooldown_frames=5))
    for _ in range(3):
        s.process("HELLO", 0.9)
    assert s.last_accepted == "HELLO"
    # hand lost -> release (two frames needed to flush the window)
    for _ in range(3):
        s.process(None, 0.0)
    assert s.last_accepted is None
    # re-show the sign -> allowed again
    token = None
    for _ in range(3):
        tok = s.process("HELLO", 0.9)
        token = tok or token
    assert token is not None and token.label == "HELLO"


def test_mean_confidence_reported():
    s = TemporalSmoothing(SmoothingConfig(window_size=3, min_stability=0.66, min_confidence=0.01))
    s.process("A", 0.8)
    token = s.process("A", 0.9)  # accepted at this point
    assert token is not None
    assert token.mean_confidence == pytest.approx(0.85)
    assert token.votes == 2


def test_invalid_config_raises():
    with pytest.raises(ValueError):
        SmoothingConfig(window_size=0).validate()
    with pytest.raises(ValueError):
        SmoothingConfig(min_stability=1.5).validate()
    with pytest.raises(ValueError):
        SmoothingConfig(min_confidence=0.0).validate()
    with pytest.raises(ValueError):
        SmoothingConfig(cooldown_frames=-1).validate()


def test_reset_clears_state():
    s = TemporalSmoothing(SmoothingConfig(window_size=3, min_stability=0.66))
    s.process("A", 0.9)
    s.process("A", 0.9)
    s.reset()
    assert not s.window
    assert s.last_accepted is None