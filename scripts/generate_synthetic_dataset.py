"""Generate a SYNTHETIC landmark dataset for pipeline smoke-testing.

STATUS: Development utility ONLY.

This script does NOT produce a real sign-language dataset and must not be used
to claim real-world accuracy. It exists so every pipeline stage
(collect -> preprocess -> train -> evaluate -> predict) can be exercised
without a webcam. Real performance is only obtainable from data collected
with ``scripts/collect_data.py``.

Usage:
    python scripts/generate_synthetic_dataset.py [--samples 200] [--classes 4]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from app.utils.landmarks_io import landmark_row, write_header  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "landmarks_synthetic.csv"


def _base_hand_kpts(fingertips_y: list[float | None]) -> np.ndarray:
    """Build a (21,3) template using the canonical MediaPipe layout.

    ``fingertips_y`` provides the normalized y offset (0 = up at wrist level)
    for tips TIP(4), TIP(8), TIP(12), TIP(16), TIP(20). ``None`` means the
    finger is curled over the palm.
    """
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[0] = [0.0, 0.0, 0.0]
    pts[1] = [0.05, 0.03, 0.05]
    pts[5] = [0.10, 0.02, 0.02]
    pts[9] = [0.14, 0.01, 0.00]
    pts[13] = [0.15, 0.04, 0.00]
    pts[17] = [0.12, 0.07, 0.02]
    pts[2] = [0.08, 0.06, 0.05]
    pts[3] = [0.13, 0.08, 0.06]
    pts[4] = [0.16, 0.10, 0.08]

    fingers = [
        (5, 6, 7, 8, fingertips_y[0]),
        (9, 10, 11, 12, fingertips_y[1]),
        (13, 14, 15, 16, fingertips_y[2]),
        (17, 18, 19, 20, fingertips_y[3]),
    ]
    for mcp, pip, dip, tip, tip_y in fingers:
        if tip_y is None:
            y = 0.02  # curled over the palm
        else:
            y = tip_y
        pts[pip] = pts[mcp] + [0.02, y * 0.45, 0.01]
        pts[dip] = pts[mcp] + [0.03, y * 0.75, 0.01]
        pts[tip] = pts[mcp] + [0.04, y, 0.02]

    # Thumb tip follows fingertips_y[4].
    thumb_y = fingertips_y[4] if fingertips_y[4] is not None else 0.04
    pts[4] = [0.16, thumb_y, 0.08]
    return pts.astype(np.float32)


def default_templates() -> dict[str, list[float | None]]:
    """A small set of clearly-separated synthetic gesture templates."""
    return {
        "OPEN_HAND": [0.15] * 5,
        "FIST": [None] * 5,
        "PEACE": [None, 0.15, 0.15, None, None],
        "THUMBS_UP": [0.15, None, None, None, None],
    }


def run(output: Path, n_samples: int, classes: int, seed: int) -> None:
    templates = default_templates()
    names = list(templates.keys())[:classes]
    rng = np.random.RandomState(seed)

    if output.exists():
        output.unlink()
    write_header(output)
    print(f"Generating {n_samples} samples per class into {output}")

    for name in names:
        base = _base_hand_kpts(templates[name])
        for _ in range(n_samples):
            pts = base + rng.normal(0.0, 0.015, size=(21, 3)).astype(np.float32)
            row = landmark_row(
                name, "Right", int(rng.randint(0, 1_000_000)), pts.reshape(-1)
            )
            with output.open("a", encoding="utf-8") as f:
                f.write(",".join(str(v) for v in row) + "\n")
    print("Done. Labels:", ", ".join(names))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=150)
    parser.add_argument("--classes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.output, args.samples, args.classes, args.seed)