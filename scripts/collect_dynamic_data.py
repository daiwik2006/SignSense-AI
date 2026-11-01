"""EXPERIMENTAL: dynamic (sequence) gesture data collection prototype.

STATUS: prototype. The supported pipeline in this project is the STATIC
gesture classifier. This script exists to define the dynamic data format and
is only worth using once you want to train a sequence (LSTM/GRU) model.

It records the most recent ``--frames`` landmark windows (rolling buffer)
each time a hand is visible and writes each accepted window to:

    data/dynamic/<label>/sample_XXX.npz     # {"seq": (T, 63), "label": label}

Usage:
    python scripts/collect_dynamic_data.py --label WAVE --frames 20 --samples 30

Press Q/ESC to quit early. A sample is recorded on each SPACE press while a
hand is detected, driven by the markers shown on screen.

Requires the steps described in docs/dynamic-signs.md.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.ml.detector import HandLandmarkDetector  # noqa: E402
from app.ml.preprocess import FEATURE_DIM_PER_HAND, PreprocessConfig, Preprocessor  # noqa: E402
from app.video.camera import CameraError, CameraStream  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WINDOW_NAME = "Sign Language Detection - Dynamic Data Collection"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect dynamic landmark sequences.")
    parser.add_argument("--label", required=True, help="Gesture label (uppercase).")
    parser.add_argument("--frames", type=int, default=20, help="Sequence length T.")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "dynamic")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    label = args.label.strip().upper()
    if not re.match(r"^[A-Z0-9_]{1,24}$", label):
        print("ERROR: label must be 1-24 chars (A-Z, 0-9, _).", file=sys.stderr)
        return 1
    if args.frames < 5 or args.frames > 120:
        print("ERROR: --frames must be in [5, 120].", file=sys.stderr)
        return 1

    out_dir = args.out / label
    out_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = Preprocessor(PreprocessConfig(max_hands=1))
    window: deque = deque(maxlen=args.frames)

    saved = 0
    try:
        with HandLandmarkDetector(max_num_hands=1) as detector, CameraStream(
            index=args.index, width=640, height=480
        ) as cam:
            while True:
                frame = cam.read()
                if frame is None:
                    print("ERROR: lost camera stream.", file=sys.stderr)
                    return 1
                detection = detector.process(frame)
                vec = preprocessor.normalize_hands(detection.hands)

                if vec is not None:
                    window.append(vec.copy())
                    status = f"BUFFERING {len(window)}/{args.frames}"
                else:
                    window.append(None)
                    status = "NO HAND"

                cv2.putText(frame, f"Label: {label}", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Saved: {saved}/{args.samples}", (12, 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, status, (12, 88),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0) if vec is not None else (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, "SPACE = save window   Q/ESC = quit", (12, 420),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if key == ord(" "):
                    seq = [v for v in window if v is not None]
                    if len(seq) < args.frames:
                        print("Skipped: window contains gaps (re-show the hand).")
                        continue
                    arr = np.stack(seq).astype(np.float32)
                    assert arr.shape == (args.frames, FEATURE_DIM_PER_HAND)
                    ts = time.strftime("%Y%m%d%H%M%S")
                    fname = out_dir / f"{ts}_{saved:04d}.npz"
                    np.savez_compressed(fname, seq=arr, label=label)
                    saved += 1
                    print(f"Saved {saved}/{args.samples} -> {fname}")
                    if saved >= args.samples:
                        break
    except CameraError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cv2.destroyAllWindows()

    print(f"Done. {saved} sequences under {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())