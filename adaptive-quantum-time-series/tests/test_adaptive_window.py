"""Unit tests for Module F: Adaptive Quantum Windowing Controller."""

import pytest

from src.adaptive_window import AdaptiveWindowController, FixedWindowPolicy
from src.config import AdaptiveWindowConfig


def test_adaptive_window_bounds_and_initialization() -> None:
    cfg = AdaptiveWindowConfig(w_base=20, w_min=5, w_max=40)
    controller = AdaptiveWindowController(cfg)

    assert controller.window == 20
    assert controller.effective_window(available=10) == 10
    assert controller.effective_window(available=50) == 20


def test_adaptive_window_regime_shift_shrink() -> None:
    cfg = AdaptiveWindowConfig(
        w_base=20,
        w_min=5,
        w_max=40,
        tau_high=0.25,
        tau_low=0.05,
        gamma_down=0.5,
        cooldown=0,
    )
    controller = AdaptiveWindowController(cfg)

    # Large error triggers shrink: floor(20 * 0.5) = 10
    new_w = controller.update(error=0.5, time_index=10)
    assert new_w == 10
    assert controller.window == 10


def test_adaptive_window_stable_expansion_with_patience() -> None:
    cfg = AdaptiveWindowConfig(
        w_base=10,
        w_min=5,
        w_max=30,
        tau_high=0.25,
        tau_low=0.08,
        gamma_up=1.5,
        patience_low=2,
        cooldown=0,
    )
    controller = AdaptiveWindowController(cfg)

    # First low error: not expanded yet because patience = 2
    w1 = controller.update(error=0.02, time_index=1)
    assert w1 == 10

    # Second low error: triggers expansion -> ceil(10 * 1.5) = 15
    w2 = controller.update(error=0.03, time_index=2)
    assert w2 == 15


def test_adaptive_window_hysteresis_and_cooldown() -> None:
    cfg = AdaptiveWindowConfig(
        w_base=20,
        w_min=5,
        w_max=40,
        tau_high=0.30,
        tau_low=0.10,
        gamma_down=0.5,
        cooldown=2,
    )
    controller = AdaptiveWindowController(cfg)

    # Step 1: Shrink
    w1 = controller.update(error=0.5, time_index=1)
    assert w1 == 10

    # Step 2: In cooldown, even with huge error window stays frozen
    w2 = controller.update(error=0.8, time_index=2)
    assert w2 == 10

    # Step 3: Still in cooldown
    w3 = controller.update(error=0.8, time_index=3)
    assert w3 == 10


def test_fixed_window_policy() -> None:
    policy = FixedWindowPolicy(window=15)
    assert policy.window == 15
    assert policy.update(error=0.99, time_index=10) == 15
    assert policy.effective_window(available=8) == 8
    assert policy.effective_window(available=100) == 15
