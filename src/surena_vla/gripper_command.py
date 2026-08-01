"""Dependency-free continuous hand-command qualification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np


class HandIntent(str, Enum):
    OPEN = "open"
    CLOSE = "close"


@dataclass(frozen=True)
class GripperCommandConfig:
    """Hysteresis, dwell, normalization, and optional EMA configuration."""

    close_threshold: float = 0.65
    release_threshold: float = 0.35
    close_dwell_ticks: int = 2
    release_dwell_ticks: int = 2
    filter_alpha: float | None = None
    # Local Bridge/RLDS actions use 0=close, 1=open; convert continuously to
    # a close-strength signal without clipping or binarizing.
    normalization_scale: float = -1.0
    normalization_offset: float = 1.0

    def __post_init__(self):
        if not self.release_threshold < self.close_threshold:
            raise ValueError("release_threshold must be strictly less than close_threshold")
        if self.close_dwell_ticks < 1 or self.release_dwell_ticks < 1:
            raise ValueError("close/release dwell ticks must be at least 1")
        if self.filter_alpha is not None and not 0.0 < self.filter_alpha <= 1.0:
            raise ValueError("filter_alpha must be in (0, 1] or None")
        values = (self.close_threshold, self.release_threshold,
                  self.normalization_scale, self.normalization_offset)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("command configuration values must be finite")


@dataclass(frozen=True)
class CommandSample:
    raw_command: float | None
    normalized_command: float | None
    filtered_command: float | None
    qualified_intent: HandIntent
    close_counter: int
    release_counter: int
    valid: bool
    intent_changed: bool
    error: str | None = None


class GripperCommandProcessor:
    """Pure deterministic processor; updates correspond to controller ticks.

    EMA uses ``y[t] = alpha*x[t] + (1-alpha)*y[t-1]``. Invalid samples
    retain the previous qualified intent and cannot create a transition.
    """

    def __init__(self, config: GripperCommandConfig | None = None):
        self.config = config or GripperCommandConfig()
        self.reset()

    def reset(self) -> None:
        self.raw_command = None
        self.normalized_command = None
        self.filtered_command = None
        self.qualified_intent = HandIntent.OPEN
        self.close_counter = 0
        self.release_counter = 0

    @staticmethod
    def _scalar(command) -> float:
        arr = np.asarray(command)
        if arr.ndim != 0:
            raise ValueError(f"gripper command must be a scalar, got shape {arr.shape}")
        value = float(arr)
        if not math.isfinite(value):
            raise ValueError("gripper command must be finite")
        return value

    def update(self, command) -> CommandSample:
        previous = self.qualified_intent
        try:
            raw = self._scalar(command)
        except (TypeError, ValueError) as exc:
            self.close_counter = self.release_counter = 0
            return self._sample(False, str(exc), False)
        cfg = self.config
        normalized = raw * cfg.normalization_scale + cfg.normalization_offset
        if not math.isfinite(normalized):
            self.close_counter = self.release_counter = 0
            return self._sample(False, "normalized gripper command is not finite", False)
        self.raw_command, self.normalized_command = raw, normalized
        if cfg.filter_alpha is None or self.filtered_command is None:
            self.filtered_command = normalized
        else:
            a = cfg.filter_alpha
            self.filtered_command = a * normalized + (1.0 - a) * self.filtered_command
        value = self.filtered_command
        if value >= cfg.close_threshold:
            self.release_counter = 0
            if previous is HandIntent.OPEN:
                self.close_counter += 1
                if self.close_counter >= cfg.close_dwell_ticks:
                    self.qualified_intent, self.close_counter = HandIntent.CLOSE, 0
            else:
                self.close_counter = 0
        elif value <= cfg.release_threshold:
            self.close_counter = 0
            if previous is HandIntent.CLOSE:
                self.release_counter += 1
                if self.release_counter >= cfg.release_dwell_ticks:
                    self.qualified_intent, self.release_counter = HandIntent.OPEN, 0
            else:
                self.release_counter = 0
        else:
            self.close_counter = self.release_counter = 0
        return self._sample(True, None, self.qualified_intent is not previous)

    def _sample(self, valid: bool, error: str | None, changed: bool) -> CommandSample:
        return CommandSample(self.raw_command, self.normalized_command,
                             self.filtered_command, self.qualified_intent,
                             self.close_counter, self.release_counter,
                             valid, changed, error)
