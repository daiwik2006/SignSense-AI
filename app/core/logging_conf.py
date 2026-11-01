"""Safe logging configuration.

Webcam/image data is never logged. Only structural information (frame counts,
model loading, connection events) is recorded.
"""

from __future__ import annotations

import logging
import sys

FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(FORMAT))
        root.addHandler(handler)
    root.setLevel(level)
    # Suppress noisy third-party logs at DEBUG.
    for noisy in ("PIL", "urllib3", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)