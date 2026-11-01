"""Application configuration and settings loading.

Secrets/credentials are never hard-coded. All deployment-specific values come
from environment variables (via ``.env``) with safe defaults appropriate for
local development. See ``.env.example`` for available keys.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VOCABULARY = PROJECT_ROOT / "configs" / "vocabulary.json"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "sign_classifier.keras"
DEFAULT_LABELS_PATH = PROJECT_ROOT / "models" / "labels.json"


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %s", key, raw, default)
        return default


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %s", key, raw, default)
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class AppSettings:
    """Runtime settings loaded from environment variables."""

    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_env: str = "development"
    app_debug: bool = True
    cors_origins: List[str] = field(default_factory=list)
    model_path: Path = DEFAULT_MODEL_PATH
    labels_path: Path = DEFAULT_LABELS_PATH
    vocabulary_path: Path = DEFAULT_VOCABULARY
    confidence_threshold: float = 0.7
    smoothing_window_size: int = 10
    smoothing_min_stability: float = 0.6
    smoothing_cooldown_frames: int = 15
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480

    @classmethod
    def from_env(cls, dotenv_path: Path | None = None) -> "AppSettings":
        load_dotenv(dotenv_path if dotenv_path else PROJECT_ROOT / ".env")

        raw_origins = os.getenv("CORS_ORIGINS", "")
        cors = [o.strip() for o in raw_origins.split(",") if o.strip()]

        return cls(
            app_host=os.getenv("APP_HOST", "0.0.0.0"),
            app_port=_env_int("APP_PORT", 8000),
            app_env=os.getenv("APP_ENV", "development"),
            app_debug=_env_bool("APP_DEBUG", True),
            cors_origins=cors,
            model_path=Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))),
            labels_path=Path(os.getenv("LABELS_PATH", str(DEFAULT_LABELS_PATH))),
            vocabulary_path=Path(
                os.getenv("VOCABULARY_PATH", str(DEFAULT_VOCABULARY))
            ),
            confidence_threshold=_env_float("CONFIDENCE_THRESHOLD", 0.7),
            smoothing_window_size=_env_int("SMOOTHING_WINDOW_SIZE", 10),
            smoothing_min_stability=_env_float("SMOOTHING_MIN_STABILITY", 0.6),
            smoothing_cooldown_frames=_env_int("SMOOTHING_COOLDOWN_FRAMES", 15),
            camera_index=_env_int("CAMERA_INDEX", 0),
            camera_width=_env_int("CAMERA_WIDTH", 640),
            camera_height=_env_int("CAMERA_HEIGHT", 480),
        )

    def validate(self) -> None:
        """Fail fast on clearly invalid settings."""
        if not (0.0 < self.confidence_threshold <= 1.0):
            raise ValueError("CONFIDENCE_THRESHOLD must be in (0, 1].")
        if self.smoothing_window_size < 1:
            raise ValueError("SMOOTHING_WINDOW_SIZE must be >= 1.")
        if not (0.0 < self.smoothing_min_stability <= 1.0):
            raise ValueError("SMOOTHING_MIN_STABILITY must be in (0, 1].")
        if self.smoothing_cooldown_frames < 0:
            raise ValueError("SMOOTHING_COOLDOWN_FRAMES must be >= 0.")
        if self.app_env == "production":
            if not self.cors_origins:
                raise ValueError(
                    "CORS_ORIGINS must be set explicitly in production "
                    "(wildcard origins are never allowed)."
                )
            if "*" in self.cors_origins:
                raise ValueError("Wildcard CORS origin '*' is not allowed.")
            if self.app_debug:
                raise ValueError("APP_DEBUG must be false in production.")


def load_vocabulary(path: Path | None = None) -> List[str]:
    """Load the configured gesture vocabulary from JSON.

    Returns an empty list when the file is absent or invalid (callers decide
    whether to fall back to free-form labels).
    """
    vocab_path = path or DEFAULT_VOCABULARY
    if not vocab_path.exists():
        logger.warning("Vocabulary file not found at %s", vocab_path)
        return []
    try:
        data = json.loads(vocab_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Could not parse vocabulary %s: %s", vocab_path, exc)
        return []
    gestures = data.get("gestures", [])
    return [str(g) for g in gestures if isinstance(g, str)]