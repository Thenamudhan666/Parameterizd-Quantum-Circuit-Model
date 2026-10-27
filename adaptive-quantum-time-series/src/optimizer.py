"""Module E (part 2) - optimizers for the PQC parameters.

* :class:`AdamOptimizer`  - gradient-based; uses exact parameter-shift gradients
                            from :meth:`PQCObjective.value_and_grad`.
* :class:`SPSAOptimizer`  - simultaneous-perturbation stochastic approximation
                            (two loss evaluations per iteration, any P).
* :class:`COBYLAOptimizer`- derivative-free trust-region method from SciPy.
                            Each "iteration" is one loss evaluation.

All optimizers share ``run(objective, x0, n_steps, callback)`` and keep their
internal state (Adam moments, SPSA iteration counter) between calls, which is
what the streaming loop needs for warm-started online updates.  A callback
returning ``True`` requests early stopping.

Numerical failures (non-finite loss/gradient) are caught and reported through
``OptimizationResult.success = False`` while returning the last finite
parameters; they are never silently ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from scipy.optimize import minimize

from src.config import OptimizerConfig
from src.loss_function import NumericalError

Callback = Callable[[int, np.ndarray, float], bool | None]


class Objective(Protocol):
    def __call__(self, params: np.ndarray) -> float: ...

    def value_and_grad(self, params: np.ndarray) -> tuple[float, np.ndarray]: ...


@dataclass(frozen=True)
class OptimizationResult:
    """Outcome of an optimizer run."""

    params: np.ndarray
    best_params: np.ndarray
    best_loss: float
    losses: tuple[float, ...]
    n_iterations: int
    n_function_evals: int
    success: bool
    message: str


class _EarlyStop(Exception):
    """Internal signal used to terminate SciPy's COBYLA from a callback."""


def _clip(grad: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(grad))
    return grad * (max_norm / norm) if norm > max_norm else grad


class AdamOptimizer:
    """Adam (Kingma & Ba, 2015) with gradient-norm clipping."""

    name = "adam"

    def __init__(self, cfg: OptimizerConfig) -> None:
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self._m: np.ndarray | None = None
        self._v: np.ndarray | None = None
        self._t = 0

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Return updated parameters (inputs are not modified)."""
        c = self.cfg
        g = _clip(np.asarray(grad, dtype=float), c.grad_clip)
        if self._m is None or self._m.shape != g.shape:
            self._m, self._v, self._t = np.zeros_like(g), np.zeros_like(g), 0
        self._t += 1
        self._m = c.beta1 * self._m + (1 - c.beta1) * g
        self._v = c.beta2 * self._v + (1 - c.beta2) * g * g
        m_hat = self._m / (1 - c.beta1**self._t)
        v_hat = self._v / (1 - c.beta2**self._t)
        return params - c.learning_rate * m_hat / (np.sqrt(v_hat) + c.epsilon)

    def run(
        self, objective: Objective, x0: np.ndarray, n_steps: int, callback: Callback | None = None
    ) -> OptimizationResult:
        x = np.asarray(x0, dtype=float).copy()
        losses: list[float] = []
        best_x, best_loss = x.copy(), np.inf
        evals = 0
        for i in range(n_steps):
            try:
                loss, grad = objective.value_and_grad(x)
            except NumericalError as exc:
                return OptimizationResult(x, best_x, best_loss, tuple(losses), i, evals, False, str(exc))
            evals += 1
            losses.append(loss)
            if loss < best_loss:
                best_x, best_loss = x.copy(), loss
            x = self.step(x, grad)
            if callback is not None and callback(i, x, loss):
                return OptimizationResult(x, best_x, best_loss, tuple(losses), i + 1, evals, True, "early stop")
        return OptimizationResult(x, best_x, best_loss, tuple(losses), n_steps, evals, True, "completed")


class SPSAOptimizer:
    """SPSA (Spall, 1998) with standard gain sequences.

    a_k = a / (k + 1 + A)^alpha,  c_k = c / (k + 1)^gamma,
    g_k = [L(x + c_k D) - L(x - c_k D)] / (2 c_k) * D,  D ~ Rademacher.

    The recorded per-iteration loss is the mean of the two perturbed losses
    (an estimate; no extra evaluation is spent on the exact value).
    """

    name = "spsa"

    def __init__(self, cfg: OptimizerConfig, seed: int) -> None:
        self.cfg = cfg
        self._seed = seed
        self.reset()

    def reset(self) -> None:
        self._rng = np.random.default_rng(self._seed)
        self._k = 0

    def run(
        self, objective: Objective, x0: np.ndarray, n_steps: int, callback: Callback | None = None
    ) -> OptimizationResult:
        c = self.cfg
        x = np.asarray(x0, dtype=float).copy()
        losses: list[float] = []
        best_x, best_loss = x.copy(), np.inf
        evals = 0
        for i in range(n_steps):
            a_k = c.spsa_a / (self._k + 1 + c.spsa_stability) ** c.spsa_alpha
            c_k = c.spsa_c / (self._k + 1) ** c.spsa_gamma
            delta = self._rng.choice([-1.0, 1.0], size=x.shape)
            try:
                f_plus = objective(x + c_k * delta)
                f_minus = objective(x - c_k * delta)
            except NumericalError as exc:
                return OptimizationResult(x, best_x, best_loss, tuple(losses), i, evals, False, str(exc))
            evals += 2
            self._k += 1
            estimate = 0.5 * (f_plus + f_minus)
            losses.append(estimate)
            if estimate < best_loss:
                best_x, best_loss = x.copy(), estimate
            grad = _clip((f_plus - f_minus) / (2.0 * c_k) * delta, c.grad_clip)
            x = x - a_k * grad
            if callback is not None and callback(i, x, estimate):
                return OptimizationResult(x, best_x, best_loss, tuple(losses), i + 1, evals, True, "early stop")
        return OptimizationResult(x, best_x, best_loss, tuple(losses), n_steps, evals, True, "completed")


class COBYLAOptimizer:
    """SciPy COBYLA; ``n_steps`` is the maximum number of loss evaluations."""

    name = "cobyla"

    def __init__(self, cfg: OptimizerConfig) -> None:
        self.cfg = cfg

    def reset(self) -> None:  # stateless between runs
        return None

    def run(
        self, objective: Objective, x0: np.ndarray, n_steps: int, callback: Callback | None = None
    ) -> OptimizationResult:
        losses: list[float] = []
        best = {"x": np.asarray(x0, dtype=float).copy(), "loss": np.inf}
        stop = {"flag": False}

        def wrapped(x: np.ndarray) -> float:
            if stop["flag"]:
                raise _EarlyStop
            loss = objective(x)
            losses.append(loss)
            if loss < best["loss"]:
                best["x"], best["loss"] = np.array(x, dtype=float), loss
            if callback is not None and callback(len(losses) - 1, np.array(x, dtype=float), loss):
                stop["flag"] = True
            return loss

        try:
            res = minimize(
                wrapped,
                np.asarray(x0, dtype=float),
                method="COBYLA",
                options={"maxiter": int(max(n_steps, 1)), "rhobeg": self.cfg.cobyla_rhobeg},
            )
            success, message, final = bool(res.success), str(res.message), np.asarray(res.x)
        except _EarlyStop:
            success, message, final = True, "early stop", best["x"]
        except NumericalError as exc:
            success, message, final = False, str(exc), best["x"]
        if not np.all(np.isfinite(final)):
            success, message, final = False, "COBYLA returned non-finite parameters", best["x"]
        return OptimizationResult(
            final, best["x"], best["loss"], tuple(losses), len(losses), len(losses), success, message
        )


PQCOptimizer = AdamOptimizer | SPSAOptimizer | COBYLAOptimizer


def make_optimizer(cfg: OptimizerConfig, seed: int) -> PQCOptimizer:
    """Factory returning a fresh optimizer instance."""
    if cfg.name == "adam":
        return AdamOptimizer(cfg)
    if cfg.name == "spsa":
        return SPSAOptimizer(cfg, seed)
    if cfg.name == "cobyla":
        return COBYLAOptimizer(cfg)
    raise ValueError(f"unknown optimizer '{cfg.name}'")
