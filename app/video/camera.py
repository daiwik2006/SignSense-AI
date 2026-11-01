"""Safe OpenCV webcam handling.

This module provides a context-managed camera wrapper so that the webcam is
always released properly, even when an exception occurs. Camera access is
independent of the machine-learning pipeline; scripts and the application can
share this single implementation.
"""

from __future__ import annotations

import logging
from typing import Iterator, Optional

import cv2

logger = logging.getLogger(__name__)


class CameraError(RuntimeError):
    """Raised when the webcam cannot be opened or read."""


class CameraStream:
    """Context manager around an OpenCV ``VideoCapture``.

    Example
    -------
    >>> with CameraStream(index=0) as cam:
    ...     for frame in cam:
    ...         _process(frame)
    """

    def __init__(
        self,
        index: int = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        api: Optional[int] = None,
    ) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.api = api
        self._capture: Optional[cv2.VideoCapture] = None

    def open(self) -> "CameraStream":
        """Open the camera and raise :class:`CameraError` on failure."""
        kwargs: dict = {}
        if self.api is not None:
            kwargs["apiPreference"] = int(self.api)
        capture = cv2.VideoCapture(self.index, **kwargs)
        if not capture.isOpened():
            capture.release()
            raise CameraError(
                f"Could not open camera index {self.index}. "
                "Check that the webcam is connected, not in use by another "
                "application, and the CAMERA_INDEX/CAMERA_API values are correct."
            )
        if self.width is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height is not None:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._capture = capture
        logger.info("Opened camera %s (%sx%s)", self.index, self.width, self.height)
        return self

    def read(self) -> Optional[object]:
        """Read one BGR frame.

        Returns ``None`` when the stream ends or a frame could not be decoded
        (rather than raising) so callers can decide how to react.
        """
        if self._capture is None:
            raise CameraError("Camera is not open. Call open() first.")
        ok, frame = self._capture.read()
        if not ok:
            logger.warning("Failed to read a frame from camera %s", self.index)
            return None
        return frame

    def __enter__(self) -> "CameraStream":
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    def release(self) -> None:
        """Release the camera and free resources."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            logger.info("Released camera %s", self.index)

    def __iter__(self) -> Iterator[object]:
        """Iterate over frames until the stream ends."""
        while True:
            frame = self.read()
            if frame is None:
                break
            yield frame