"""Module F - adaptive quantum windowing controller.

The controller decides how many of the most recent labelled observations the
PQC is re-fitted on at each streaming step.

Let e_t be the (optionally EMA-smoothed) absolute one-step prediction error:

    e_t = alpha * |y_t - yhat_t| + (1 - alpha) * e_{t-1}      (alpha = 1 -> raw error)

Rules (W = current window):

* regime-shift response:  e_t > tau_high  ->  W <- max(W_min, floor(W * gamma_down))
* stable-region response: e_t < tau_low for ``patience_low`` consecutive steps
                                         ->  W <- min(W_max, ceil(W * gamma_up))
* otherwise: hold W.

Anti-oscillation safeguards:

1. the dead-band [tau_low, tau_high] provides hysteresis;
2. after any change the window is frozen for ``cooldown`` steps;
3. expansion additionally requires ``patience_low`` consecutive low-error steps,
   so the controller shrinks quickly but grows cautiously.

Causality: :meth:`AdaptiveWindowController.update` must be called with the error
of a prediction that was produced *before* the target was observed.  The
controller keeps no reference to the data, so it cannot look ahead.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from src.config import AdaptiveWindowConfig


@dataclass(frozen=True)
class WindowEvent:
    """One controller decision, recorded for every streaming step."""

    step: int
    time_index: int
    error: float
    smoothed_error: float
    old_window: int
    new_window: int
    action: str

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def effective_window(window: int, available: int) -> int:
    """Clamp a window to the number of labelled observations available.

    Raises:
        ValueError: If no observations are available yet.
    """
    if available < 1:
        raise ValueError("no labelled observations are available for training")
    return min(window, available)


class FixedWindowPolicy:
    """Constant-window policy with the same interface as the adaptive controller."""

    def __init__(self, window: int) -> None:
        if window < 1:
            raise ValueError("fixed window must be >= 1")
        self._window = int(window)
        self._history: list[WindowEvent] = []

    @property
    def window(self) -> int:
        return self._window

    @property
    def history(self) -> tuple[WindowEvent, ...]:
        return tuple(self._history)

    def update(self, error: float, time_index: int) -> int:
        _validate_error(error)
        self._history.append(
            WindowEvent(
                step=len(self._history),
                time_index=int(time_index),
                error=float(error),
                smoothed_error=float(error),
                old_window=self._window,
                new_window=self._window,
                action="fixed",
            )
        )
        return self._window

    def effective_window(self, available: int) -> int:
        return effective_window(self._window, available)


class AdaptiveWindowController:
    """Error-driven window controller (see module docstring for the rules)."""

    def __init__(self, cfg: AdaptiveWindowConfig | None = None) -> None:
        if cfg is None:
            cfg = AdaptiveWindowConfig()
        self._cfg = cfg
        self._window = int(cfg.w_base)
        self._smoothed: float | None = None
        self._steps_since_change = cfg.cooldown  # allow an immediate first change
        self._low_streak = 0
        self._history: list[WindowEvent] = []

    @property
    def config(self) -> AdaptiveWindowConfig:
        return self._cfg

    @property
    def window(self) -> int:
        return self._window

    @property
    def history(self) -> tuple[WindowEvent, ...]:
        return tuple(self._history)

    @property
    def n_changes(self) -> int:
        return sum(1 for e in self._history if e.new_window != e.old_window)

    def shrunk(self, window: int) -> int:
        """Window after a regime-shift response."""
        return max(self._cfg.w_min, math.floor(window * self._cfg.gamma_down))

    def expanded(self, window: int) -> int:
        """Window after a stable-region response."""
        return min(self._cfg.w_max, math.ceil(window * self._cfg.gamma_up))

    def update(self, error: float, time_index: int) -> int:
        """Ingest the latest absolute prediction error and return the new window."""
        _validate_error(error)
        cfg = self._cfg
        alpha = cfg.error_smoothing
        self._smoothed = (
            float(error)
            if self._smoothed is None
            else alpha * float(error) + (1.0 - alpha) * self._smoothed
        )
        e = self._smoothed
        old = self._window
        in_cooldown = self._steps_since_change < cfg.cooldown

        if e > cfg.tau_high:
            self._low_streak = 0
            candidate = self.shrunk(old)
            action = "cooldown" if in_cooldown and candidate != old else "shrink"
        elif e < cfg.tau_low:
            self._low_streak += 1
            if self._low_streak < cfg.patience_low:
                candidate, action = old, "hold_low_patience"
            else:
                candidate = self.expanded(old)
                action = "cooldown" if in_cooldown and candidate != old else "expand"
        else:
            self._low_streak = 0
            candidate, action = old, "hold"

        if action == "cooldown" or candidate == old:
            new = old
            if action in {"shrink", "expand"}:
                action = f"{action}_at_bound"
        else:
            new = candidate
            self._low_streak = 0

        self._steps_since_change = 0 if new != old else self._steps_since_change + 1
        self._window = new
        self._history.append(
            WindowEvent(
                step=len(self._history),
                time_index=int(time_index),
                error=float(error),
                smoothed_error=float(e),
                old_window=old,
                new_window=new,
                action=action,
            )
        )
        return new

    def effective_window(self, available: int) -> int:
        """Current window clamped to the available labelled history."""
        return effective_window(self._window, available)


def _validate_error(error: float) -> None:
    if not isinstance(error, (int, float)) or not math.isfinite(error) or error < 0:
        raise ValueError(f"prediction error must be a finite non-negative number, got {error!r}")
