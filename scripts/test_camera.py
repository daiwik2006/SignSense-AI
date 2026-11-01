"""Test the webcam independently of any ML model.

Usage:
    python scripts/test_camera.py [--index 0] [--width 640] [--height 480]

Displays the live webcam feed with an FPS counter.
Press Q (or ESC) in the preview window to quit.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running directly without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from app.video.camera import CameraError, CameraStream  # noqa: E402

WINDOW_NAME = "Sign Language Detection - Camera Test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test the webcam feed.")
    parser.add_argument("--index", type=int, default=0, help="Camera index (default 0).")
    parser.add_argument("--width", type=int, default=640, help="Requested frame width.")
    parser.add_argument("--height", type=int, default=480, help="Requested frame height.")
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    try:
        with CameraStream(index=args.index, width=args.width, height=args.height) as cam:
            prev_time = time.perf_counter()
            while True:
                frame = cam.read()
                if frame is None:
                    print("ERROR: Lost the video stream. Exiting.", file=sys.stderr)
                    return 1

                now = time.perf_counter()
                fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
                prev_time = now

                cv2.putText(
                    frame,
                    f"FPS: {fps:5.1f}",
                    (12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):  # Q or ESC
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