"""API routes: health, model info, and WebSocket inference.

The WebSocket endpoint performs the full inference pipeline on each received
frame: JPEG decode -> MediaPipe detection -> preprocessing -> classifier ->
confidence filter -> temporal smoothing. The annotated frame is returned so
the browser does not need any computer-vision logic.

Frames are processed strictly in memory and never written to disk.
"""

from __future__ import annotations

import base64
import binascii
import logging
import time

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.config import AppSettings
from app.ml.detector import HandLandmarkDetector, draw_hand_landmarks
from app.ml.predictor import GesturePredictor, ModelNotLoadedError
from app.ml.smoothing import SmoothingConfig, TemporalSmoothing

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_FRAME_BYTES = 3_000_000  # 640x480 JPEG is typically well under 100 KB


class InferenceService:
    """Holds the long-lived detector, predictor, and per-connection smoothing."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.detector: HandLandmarkDetector | None = None
        self.predictor: GesturePredictor | None = None
        self.error: str | None = None
        self._load_error = False

    def ensure_loaded(self) -> None:
        """Load inference components once; report a clear error if untrained."""
        if self.predictor is not None:
            return
        if self._load_error:
            return
        if self.detector is None:
            self.detector = HandLandmarkDetector(max_num_hands=1)
        try:
            self.predictor = GesturePredictor(
                self.settings.model_path,
                self.settings.labels_path,
                confidence_threshold=self.settings.confidence_threshold,
            )
        except ModelNotLoadedError as exc:
            self._load_error = True
            self.error = str(exc)
            logger.error("Model could not be loaded: %s", exc)

    @property
    def model_ready(self) -> bool:
        return self.predictor is not None


_service: InferenceService | None = None


def get_service() -> InferenceService:
    global _service
    if _service is None:
        raise RuntimeError("InferenceService not initialized (startup incomplete).")
    return _service


def init_service(settings: AppSettings) -> InferenceService:
    global _service
    _service = InferenceService(settings)
    return _service


def _decode_frame(payload: bytes) -> np.ndarray | None:
    """Decode a JPEG (or other cv2-readable) frame from raw bytes."""
    if not payload:
        return None
    if len(payload) > MAX_FRAME_BYTES:
        logger.warning("Rejecting oversized frame (%d bytes)", len(payload))
        return None
    arr = np.frombuffer(payload, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


def _encode_jpeg(frame: np.ndarray, quality: int = 60) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


@router.get("/health")
def health() -> dict:
    service = get_service()
    return {
        "status": "ok",
        "app_env": service.settings.app_env,
        "model_ready": service.model_ready,
    }


@router.get("/api/model/info")
def model_info() -> dict:
    service = get_service()
    predictor = service.predictor
    if predictor is None:
        return {
            "model_path": str(service.settings.model_path),
            "model_ready": False,
            "num_classes": None,
            "labels": [],
            "feature_dim": None,
            "confidence_threshold": service.settings.confidence_threshold,
            "error": service.error,
        }
    return {
        "model_path": str(service.settings.model_path),
        "model_ready": True,
        "num_classes": len(predictor.index_to_label),
        "labels": [predictor.index_to_label[i] for i in sorted(predictor.index_to_label)],
        "feature_dim": predictor.preprocessor.feature_dim(),
        "confidence_threshold": predictor.confidence_threshold,
    }


@router.websocket("/ws/inference")
async def ws_inference(websocket: WebSocket) -> None:
    await websocket.accept()
    service = get_service()
    service.ensure_loaded()

    if service.predictor is None or service.detector is None:
        await websocket.send_json(
            {
                "type": "error",
                "message": (
                    "No trained model available. Run the dataset preparation and "
                    "training scripts first (see README)."
                ),
            }
        )
        await websocket.close(code=1011)
        return

    smoother = TemporalSmoothing(
        SmoothingConfig(
            window_size=service.settings.smoothing_window_size,
            min_stability=service.settings.smoothing_min_stability,
            # The predictor already applies the confidence threshold; the
            # smoother only needs a tiny guard for malformed inputs.
            min_confidence=0.01,
            cooldown_frames=service.settings.smoothing_cooldown_frames,
        )
    )

    logger.info("Inference websocket connected")
    prev_time = time.perf_counter()
    fps = 0.0
    try:
        while True:
            payload = await websocket.receive()

            if payload.get("type") == "websocket.disconnect":
                break

            message = payload.get("text")
            bytes_payload = payload.get("bytes")

            if message is None and bytes_payload is None:
                continue

            if message is not None:
                # JSON frame message: {"type":"frame","image":"..."}
                import json

                try:
                    msg = json.loads(message)
                    if msg.get("type") != "frame" or not msg.get("image"):
                        continue
                    data = msg["image"]
                    if data.startswith("data:"):
                        data = data.split(",", 1)[1]
                    frame = _decode_frame(base64.b64decode(data))
                except (json.JSONDecodeError, ValueError, binascii.Error) as exc:
                    logger.debug("Ignoring malformed frame message: %s", exc)
                    continue
            else:
                frame = _decode_frame(bytes_payload)

            if frame is None:
                await websocket.send_json(
                    {"type": "error", "message": "Could not decode frame."}
                )
                continue

            detection = service.detector.process(frame)
            draw_hand_landmarks(frame, detection)
            pred = service.predictor.predict_detection(detection)

            label = pred.label if pred else None
            confidence = pred.confidence if pred else 0.0
            accepted = smoother.process(label, confidence)

            now = time.perf_counter()
            fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
            prev_time = now

            response = {
                "type": "prediction",
                "label": label,
                "confidence": confidence,
                "probabilities": pred.probabilities if pred else {},
                "accepted_label": accepted.label if accepted else None,
                "accepted_confidence": accepted.mean_confidence if accepted else None,
                "fps": round(fps, 1),
                "annotated": _encode_jpeg(frame, quality=60),
            }
            await websocket.send_json(response)
    except WebSocketDisconnect:
        logger.info("Inference websocket disconnected")
    finally:
        # Per-connection smoothing state holds no sensitive data; nothing to flush.
        pass