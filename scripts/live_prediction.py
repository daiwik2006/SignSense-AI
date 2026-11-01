"""Live real-time prediction demo using the webcam.

Combines: webcam -> MediaPipe -> preprocessing -> classifier -> confidence
filter -> temporal smoothing. Displays prediction, confidence, and FPS.

Usage:
    python scripts/live_prediction.py [--model models/sign_classifier.keras]
                                      [--labels models/labels.json]

Press Q (or ESC) to quit. Press ENTER/SPACE while a prediction is shown to
print the accepted token to the console (useful for demo/notes).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from app.ml.detector import HandLandmarkDetector, draw_hand_landmarks  # noqa: E402
from app.ml.predictor import GesturePredictor, ModelNotLoadedError  # noqa: E402
from app.ml.smoothing import SmoothingConfig, TemporalSmoothing  # noqa: E402
from app.video.camera import CameraError, CameraStream  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WINDOW_NAME = "Sign Language Detection - Live Prediction"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live gesture prediction.")
    parser.add_argument(
        "--model", type=Path, default=PROJECT_ROOT / "models" / "sign_classifier.keras"
    )
    parser.add_argument(
        "--labels", type=Path, default=PROJECT_ROOT / "models" / "labels.json"
    )
    parser.add_argument("--index", type=int, default=0, help="Camera index.")
    parser.add_argument("--threshold", type=float, default=0.7, help="Confidence threshold.")
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    try:
        predictor = GesturePredictor(
            args.model, args.labels, confidence_threshold=args.threshold
        )
    except ModelNotLoadedError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    smoother = TemporalSmoothing(
        SmoothingConfig(window_size=10, min_stability=0.6, cooldown_frames=15)
    )

    try:
        with HandLandmarkDetector(max_num_hands=1) as detector, CameraStream(
            index=args.index, width=640, height=480
        ) as cam:
            prev = time.perf_counter()
            while True:
                frame = cam.read()
                if frame is None:
                    print("ERROR: lost camera stream.", file=sys.stderr)
                    return 1

                detection = detector.process(frame)
                draw_hand_landmarks(frame, detection)

                pred = predictor.predict_detection(detection)
                label = pred.label if pred else None
                conf = pred.confidence if pred else 0.0
                accepted = smoother.process(label, conf)

                now = time.perf_counter()
                fps = 1.0 / (now - prev) if now > prev else 0.0
                prev = now

                display_label = "Unknown" if pred is None else f"{pred.label}"
                display_conf = f"{pred.confidence * 100:.0f}%"

                cv2.putText(
                    frame, f"Predict: {display_label} ({display_conf})",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA,
                )
                cv2.putText(
                    frame, f"FPS: {fps:4.1f}",
                    (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA,
                )
                if accepted is not None:
                    cv2.putText(
                        frame, f"ACCEPTED: {accepted.label}",
                        (12, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 255), 2, cv2.LINE_AA,
                    )

                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    print("Quitting...")
                    break
                if key in (ord("\n"), ord(" ")):
                    if accepted is not None:
                        print(f"[token] {accepted.label} (votes={accepted.votes})")
                    else:
                        print("[enter] no stable token yet")
    except CameraError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(run())