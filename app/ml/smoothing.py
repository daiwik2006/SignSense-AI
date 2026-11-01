"""Temporal smoothing/debounce for prediction streams.

Problem: a classifier produces one prediction per video frame. Holding a sign
for two seconds at 30 FPS would otherwise emit the same word ~60 times.
Smoothing solves this with a rolling majority vote, a stability check, and a
cooldown, so a held gesture produces a single accepted token until the user
changes or releases the gesture.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple

FrameEntry = Tuple[Optional[str], float]  # (label, confidence)


@dataclass
class SmoothingConfig:
    window_size: int = 10
    min_stability: float = 0.6
    min_confidence: float = 0.7
    cooldown_frames: int = 15
    release_on_hand_loss: bool = True

    def validate(self) -> None:
        if self.window_size < 1:
            raise ValueError("window_size must be >= 1")
        if not (0.0 < self.min_stability <= 1.0):
            raise ValueError("min_stability must be in (0, 1]")
        if not (0.0 < self.min_confidence <= 1.0):
            raise ValueError("min_confidence must be in (0, 1]")
        if self.cooldown_frames < 0:
            raise ValueError("cooldown_frames must be >= 0")


@dataclass
class AcceptedPrediction:
    """A token that passed the stability and confidence checks."""

    label: str
    votes: int
    window_size: int
    mean_confidence: float


class TemporalSmoothing:
    """Turn a raw per-frame prediction stream into debounced tokens."""

    def __init__(self, config: Optional[SmoothingConfig] = None) -> None:
        self.config = config or SmoothingConfig()
        self.config.validate()
        self.window: deque = deque(maxlen=self.config.window_size)
        self._last_accepted: Optional[str] = None
        self._cooldown_remaining: int = 0

    def process(
        self, label: Optional[str], confidence: Optional[float] = None
    ) -> Optional[AcceptedPrediction]:
        """Push one frame prediction and return an accepted token or ``None``.

        Parameters
        ----------
        label:
            Predicted class label, or ``None`` when the hand is absent or the
            confidence was too low.
        confidence:
            Confidence in ``[0, 1]`` for ``label``.
        """
        if label is not None and confidence is not None:
            if confidence < self.config.min_confidence:
                label = None
        self.window.append((label, confidence if label is not None else 0.0))

        dominant = self._dominant()
        if dominant is None:
            if self.config.release_on_hand_loss:
                self._last_accepted = None
            self._cooldown_remaining = 0
            return None

        vote_count = self._count(dominant)
        required = math.ceil(self.config.min_stability * self.config.window_size)
        if vote_count < required:
            return None

        # A dominant label that matches the last accepted token is suppressed
        # during the cooldown, and suppressed forever afterwards as long as the
        # hand stays put (a held sign must emit only one token).
        if dominant == self._last_accepted:
            self._cooldown_remaining = max(self._cooldown_remaining - 1, 0)
            return None

        self._last_accepted = dominant
        self._cooldown_remaining = self.config.cooldown_frames
        mean_conf = self._mean_confidence(dominant)
        return AcceptedPrediction(
            label=dominant,
            votes=vote_count,
            window_size=self.config.window_size,
            mean_confidence=mean_conf,
        )

    def _dominant(self) -> Optional[str]:
        if not self.window:
            return None
        counter = Counter(entry[0] for entry in self.window)
        label, _ = counter.most_common(1)[0]
        return None if label is None else label

    def _count(self, label: str) -> int:
        return sum(1 for entry in self.window if entry[0] == label)

    def _mean_confidence(self, label: str) -> float:
        entries = [c for (l, c) in self.window if l == label]
        if not entries:
            return 0.0
        return sum(entries) / len(entries)

    def reset(self) -> None:
        """Clear the window and internal acceptance state."""
        self.window.clear()
        self._last_accepted = None
        self._cooldown_remaining = 0

    @property
    def last_accepted(self) -> Optional[str]:
        return self._last_accepted