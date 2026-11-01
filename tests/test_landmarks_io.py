"""Tests for the landmark CSV I/O module."""

import numpy as np
import pytest

from app.ml.preprocess import FEATURE_DIM_PER_HAND
from app.utils.landmarks_io import (
    TOTAL_COLS,
    landmark_row,
    load_landmarks,
    row_count,
    write_header,
)


def make_flat(fill=0.1):
    return np.full(FEATURE_DIM_PER_HAND, fill, dtype=np.float32)


def test_landmark_row_layout(tmp_path):
    row = landmark_row("HELLO", "Right", 123, make_flat())
    assert len(row) == TOTAL_COLS
    assert row[0] == "HELLO"
    assert row[1] == "Right"
    assert row[2] == 123


def test_landmark_row_rejects_wrong_size(tmp_path):
    with pytest.raises(ValueError):
        landmark_row("HELLO", "Right", 1, np.zeros(10))


def test_write_header_then_count(tmp_path):
    p = tmp_path / "landmarks.csv"
    write_header(p)
    assert row_count(p) == 0
    with p.open("a", encoding="utf-8") as f:
        f.write(",".join(str(v) for v in landmark_row("YES", "Left", 1, make_flat())) + "\n")
        f.write(",".join(str(v) for v in landmark_row("NO", "Right", 2, make_flat())) + "\n")
    assert row_count(p) == 2


def test_load_roundtrip(tmp_path):
    p = tmp_path / "landmarks.csv"
    write_header(p)
    flat = make_flat(0.25)
    with p.open("a", encoding="utf-8") as f:
        f.write(",".join(str(v) for v in landmark_row("OK", "Right", 999, flat)) + "\n")
    labels, handedness, timestamps, features = load_landmarks(p)
    assert labels == ["OK"]
    assert handedness == ["Right"]
    assert timestamps == [999]
    assert features.shape == (1, FEATURE_DIM_PER_HAND)
    assert np.allclose(features[0], 0.25)


def test_load_missing_file(tmp_path):
    labels, handedness, timestamps, features = load_landmarks(tmp_path / "missing.csv")
    assert labels == [] and handedness == [] and timestamps == []
    assert features.shape == (0, FEATURE_DIM_PER_HAND)


def test_load_skips_malformed_rows(tmp_path):
    p = tmp_path / "landmarks.csv"
    write_header(p)
    good = landmark_row("OK", "Right", 1, make_flat())
    with p.open("a", encoding="utf-8") as f:
        f.write(",".join(str(v) for v in good) + "\n")
        f.write("MALFORMED,ROW\n")
        f.write(",".join(str(v) for v in good) + "\n")
    labels, _, _, features = load_landmarks(p)
    assert len(labels) == 2
    assert features.shape[0] == 2