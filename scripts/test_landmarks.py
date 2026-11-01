"""Live hand-landmark overlay demo using the webcam.

Usage:
    python scripts/test_landmarks.py [--index 0] [--two-hands]

Press Q (or ESC) in the preview window to quit.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from app.ml.detector import (  # noqa: E402
    HandLandmarkDetector,
    draw_hand_landmarks,
)
from app.video.camera import CameraError, CameraStream  # noqa: E402

WINDOW_NAME = "Sign Language Detection - Hand Landmarks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show MediaPipe hand landmarks.")
    parser.add_argument("--index", type=int, default=0, help="Camera index (default 0).")
    parser.add_argument("--two-hands", action="store_true", help="Detect up to two hands.")
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    max_hands = 2 if args.two_hands else 1
    try:
        with HandLandmarkDetector(max_num_hands=max_hands) as detector:
            with CameraStream(index=args.index, width=640, height=480) as cam:
                prev_time = time.perf_counter()
                while True:
                    frame = cam.read()
                    if frame is None:
                        print("ERROR: Lost the video stream. Exiting.", file=sys.stderr)
                        return 1

                    detection = detector.process(frame)
                    draw_hand_landmarks(frame, detection)

                    now = time.perf_counter()
                    fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
                    prev_time = now

                    num_hands = len(detection.hands)
                    cv2.putText(
                        frame,
                        f"FPS: {fps:5.1f}  Hands: {num_hands}",
                        (12, 34),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(WINDOW_NAME, frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):
                        print("Quitting...")
                        break
    except CameraError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted. Quitting...")
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(run())