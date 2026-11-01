"""Interactive landmark dataset collection tool.

Collects normalized *raw* MediaPipe landmarks (not video) for gestures you
define. Data is appended to ``data/raw/landmarks.csv``; existing samples are
never silently overwritten.

Usage:
    python scripts/collect_data.py --label HELLO --samples 200
    python scripts/collect_data.py        # interactive prompts

Controls inside the preview window:
    Q / ESC   quit the collection session
    SPACE     skip the current frame (do not capture a sample)

Instructions:
    To make the classifier robust, record each label from multiple angles,
    distances, and lighting conditions, and slightly vary your hand pose.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.ml.detector import (  # noqa: E402
    HandLandmarkDetector,
    draw_hand_landmarks,
    DetectedHand,
)
from app.utils.landmarks_io import FEATURE_COLS, landmark_row, row_count, write_header  # noqa: E402
from app.video.camera import CameraError, CameraStream  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "landmarks.csv"
VOCABULARY_PATH = PROJECT_ROOT / "configs" / "vocabulary.json"
WINDOW_NAME = "Sign Language Detection - Data Collection"

LABEL_RE = re.compile(r"^[A-Z0-9_]{1,24}$")


def load_vocabulary() -> list[str]:
    """Read the configured gesture labels from ``configs/vocabulary.json``."""
    if not VOCABULARY_PATH.exists():
        return []
    try:
        data = json.loads(VOCABULARY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"ERROR: vocabulary.json is not valid JSON: {exc}", file=sys.stderr)
        return []
    labels = data.get("gestures", [])
    return [str(x) for x in labels if isinstance(x, str)]


def validate_label(label: str, vocabulary: list[str]) -> bool:
    """A label is valid when it matches the allowed pattern and, if a
    vocabulary is configured, it is part of that vocabulary."""
    if not LABEL_RE.match(label):
        print(
            "ERROR: label must be 1-24 characters; only uppercase letters, "
            "digits, and underscores are allowed.",
            file=sys.stderr,
        )
        return False
    if vocabulary and label not in vocabulary:
        print(
            f"ERROR: '{label}' is not in configs/vocabulary.json. "
            f"Available gestures: {', '.join(vocabulary)}. "
            "Add it to the file if you want to train it.",
            file=sys.stderr,
        )
        return False
    return True


def prompt_label(vocabulary: list[str]) -> str:
    """Ask the user for a gesture label, validating against the vocabulary."""
    if vocabulary:
        print("Configured gestures (configs/vocabulary.json):")
        print("  " + ", ".join(vocabulary))
    while True:
        label = input("\nGesture label to collect: ").strip().upper()
        if validate_label(label, vocabulary):
            return label
        print("Please enter a valid label.")


def prompt_samples() -> int:
    while True:
        raw = input("Number of samples to collect: ").strip()
        try:
            n = int(raw)
        except ValueError:
            print("Please enter an integer.", file=sys.stderr)
            continue
        if n < 1 or n > 5000:
            print("Sample count must be between 1 and 5000.", file=sys.stderr)
            continue
        return n


def _single_hand_features(detection) -> DetectedHand | None:
    return detection.hands[0] if detection.hands else None


def run(label: str, n_samples: int, output: Path, camera_index: int) -> int:
    existing = row_count(output)
    if existing:
        print(f"NOTE: {output.name} already has {existing} samples (appending).")
    if not output.exists():
        write_header(output)
        print(f"Created {output} with column header.")

    try:
        with HandLandmarkDetector(max_num_hands=1) as detector, CameraStream(
            index=camera_index, width=640, height=480
        ) as cam:
            captured = 0
            skip_note = False
            print(
                "\nShow the gesture now. Press SPACE to skip, Q/ESC to quit.\n"
                "Capturing...\n"
            )
            while captured < n_samples:
                frame = cam.read()
                if frame is None:
                    print("ERROR: lost camera stream.", file=sys.stderr)
                    return 1

                detection = detector.process(frame)
                draw_hand_landmarks(frame, detection)

                hand = _single_hand_features(detection)
                status = "NO HAND DETECTED" if hand is None else "READY"
                status_color = (0, 0, 255) if hand is None else (0, 255, 0)

                cv2.putText(
                    frame,
                    f"Label: {label}",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"Captured: {captured}/{n_samples}",
                    (12, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    status,
                    (12, 88),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    status_color,
                    2,
                    cv2.LINE_AA,
                )
                if skip_note:
                    cv2.putText(
                        frame,
                        "Skipped",
                        (12, 118),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 165, 255),
                        2,
                        cv2.LINE_AA,
                    )
                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    print("Quit requested. Stopping collection.")
                    break
                if key == ord(" "):
                    skip_note = True
                    continue
                skip_note = False

                if hand is None:
                    continue

                flat = np.array(
                    [[p.x, p.y, p.z] for p in hand.landmarks], dtype=np.float32
                ).reshape(-1)
                assert flat.size == FEATURE_COLS

                row = landmark_row(label, hand.handedness, int(time.time() * 1000), flat)
                with output.open("a", encoding="utf-8") as f:
                    f.write(",".join(str(v) for v in row) + "\n")
                captured += 1
                print(f"Captured {captured}/{n_samples}", end="\r")
    except CameraError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted. Quitting.")
    finally:
        cv2.destroyAllWindows()

    total = row_count(output)
    print(f"\nDone. {captured} samples written for label '{label}'.")
    print(f"Dataset now has {total} samples total in {output}.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect landmark samples.")
    parser.add_argument("--label", type=str, default=None, help="Gesture label.")
    parser.add_argument("--samples", type=int, default=None, help="Number of samples.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument("--index", type=int, default=0, help="Camera index.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    vocab = load_vocabulary()
    if not vocab:
        print("WARNING: configs/vocabulary.json is missing or empty; "
              "any valid label pattern will be accepted.")
    label = args.label
    samples = args.samples
    if label is None:
        label = prompt_label(vocab)
    if not validate_label(label, vocab):
        sys.exit(1)
    if samples is None:
        samples = prompt_samples()
    if samples < 1 or samples > 5000:
        print("ERROR: sample count must be between 1 and 5000.", file=sys.stderr)
        sys.exit(1)
    sys.exit(run(label, samples, args.output, args.index))