"""Shared CSV utilities for the landmark dataset.

The raw dataset is stored as a single headerless CSV file with a consistent
column layout so that training scripts can ``numpy.loadtxt`` it directly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from app.ml.preprocess import COORDS_PER_LANDMARK, NUM_LANDMARKS

logger = logging.getLogger(__name__)

# Column layout: label(str), handedness(str), timestamp_ms(int), 63 floats
METADATA_COLS = 3
FEATURE_COLS = NUM_LANDMARKS * COORDS_PER_LANDMARK  # 63
TOTAL_COLS = METADATA_COLS + FEATURE_COLS  # 66


def landmark_row(
    label: str,
    handedness: str,
    timestamp_ms: int,
    landmarks_flat: np.ndarray,
) -> list:
    """Build one CSV row from a single sample.

    ``landmarks_flat`` must be a flat array of length ``FEATURE_COLS``.
    """
    if landmarks_flat.size != FEATURE_COLS:
        raise ValueError(
            f"landmarks_flat must have {FEATURE_COLS} elements, got {landmarks_flat.size}"
        )
    return [label, handedness, timestamp_ms] + landmarks_flat.tolist()


def write_header(path: Path) -> None:
    """Write a CSV header line and create the parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("label,handedness,timestamp_ms,")
        parts = [f"lm{i}_{c}" for i in range(NUM_LANDMARKS) for c in ("x", "y", "z")]
        f.write(",".join(parts) + "\n")


def row_count(path: Path) -> int:
    """Return the number of data rows (excluding header) in the CSV."""
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    return max(total_lines - 1, 0)  # subtract header


def load_landmarks(
    path: Path,
) -> tuple[list[str], list[str], list[int], np.ndarray]:
    """Load all rows from a landmarks CSV.

    Returns
    -------
    labels : list of str
    handedness : list of str
    timestamps : list of int
    features : ndarray, shape (N, 63)
    """
    if not path.exists():
        return [], [], [], np.empty((0, FEATURE_COLS), dtype=np.float64)
    with path.open("r", encoding="utf-8") as f:
        header = f.readline()  # skip header
        data = f.readlines()
    if not data:
        return [], [], [], np.empty((0, FEATURE_COLS), dtype=np.float64)

    labels: list[str] = []
    handedness_list: list[str] = []
    timestamps: list[int] = []
    rows: list[list[float]] = []

    for line in data:
        parts = line.strip().split(",")
        if len(parts) != TOTAL_COLS:
            logger.warning("Skipping malformed row (expected %d cols, got %d)", TOTAL_COLS, len(parts))
            continue
        labels.append(parts[0])
        handedness_list.append(parts[1])
        try:
            timestamps.append(int(parts[2]))
        except ValueError:
            timestamps.append(0)
        rows.append([float(x) for x in parts[METADATA_COLS:]])

    features = np.array(rows, dtype=np.float64) if rows else np.empty((0, FEATURE_COLS), dtype=np.float64)
    return labels, handedness_list, timestamps, features