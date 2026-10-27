"""Module E (part 1) - loss functions and analytic gradients for the PQC.

Objective (normalised units):

    L(params) = (1/M) * sum_i (yhat_i - y_i)^2  +  lambda * ||theta||^2

where yhat_i = sum_q w_q E_q(x_i, theta) + b.  The optional L2 term acts on
the circuit angles only and is a simple circuit-complexity penalty that keeps
the ansatz close to the identity.

Gradient (chain rule; dE/dtheta from the parameter-shift rule):

    dL/dtheta_j = (2/M) sum_i r_i sum_q w_q dE_q(x_i)/dtheta_j + 2 lambda theta_j
    dL/dw_q     = (2/M) sum_i r_i E_q(x_i)
    dL/db       = (2/M) sum_i r_i                       with r_i = yhat_i - y_i
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from src.quantum_circuit import PQCModel


class NumericalError(RuntimeError):
    """Raised when a loss, gradient or expectation value is not finite."""


def mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Mean squared error with shape and finiteness checks."""
    p = np.asarray(predictions, dtype=float)
    t = np.asarray(targets, dtype=float)
    if p.shape != t.shape:
        raise ValueError(f"shape mismatch: predictions {p.shape} vs targets {t.shape}")
    if p.size == 0:
        raise ValueError("cannot compute MSE of empty arrays")
    value = float(np.mean((p - t) ** 2))
    if not np.isfinite(value):
        raise NumericalError("MSE is not finite")
    return value


class PQCObjective:
    """Differentiable MSE objective of a :class:`PQCModel` on a fixed batch."""

    def __init__(
        self, model: "PQCModel", angles: np.ndarray, targets: np.ndarray, l2_reg: float = 0.0
    ) -> None:
        a = np.atleast_2d(np.asarray(angles, dtype=float))
        y = np.asarray(targets, dtype=float).reshape(-1)
        if a.shape[0] != y.shape[0] or y.size == 0:
            raise ValueError(f"need matching, non-empty batches; got {a.shape[0]} vs {y.size}")
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(y))):
            raise ValueError("training batch contains non-finite values")
        if l2_reg < 0:
            raise ValueError("l2_reg must be >= 0")
        self.model = model
        self.angles = a
        self.targets = y
        self.l2_reg = float(l2_reg)
        self.n_value_calls = 0
        self.n_gradient_calls = 0

    def _penalty(self, theta: np.ndarray) -> float:
        return self.l2_reg * float(theta @ theta)

    def __call__(self, params: np.ndarray) -> float:
        """Loss value only (used by derivative-free optimizers)."""
        theta, _, _ = self.model.split(params)
        preds = self.model.predict(self.angles, params)
        self.n_value_calls += 1
        loss = mse(preds, self.targets) + self._penalty(theta)
        if not np.isfinite(loss):
            raise NumericalError("loss is not finite")
        return loss

    def value_and_grad(self, params: np.ndarray) -> tuple[float, np.ndarray]:
        """Loss value and exact gradient (parameter-shift + chain rule)."""
        theta, w, b = self.model.split(params)
        e, jac = self.model.expectations_and_jacobian(self.angles, theta)
        preds = e @ w + b
        resid = preds - self.targets
        m = resid.size
        g_theta = (2.0 / m) * np.einsum("i,iqp,q->p", resid, jac, w) + 2.0 * self.l2_reg * theta
        g_w = (2.0 / m) * (resid @ e)
        g_b = (2.0 / m) * resid.sum()
        grad = np.concatenate([g_theta, g_w, [g_b]])
        loss = mse(preds, self.targets) + self._penalty(theta)
        self.n_gradient_calls += 1
        if not (np.isfinite(loss) and np.all(np.isfinite(grad))):
            raise NumericalError("loss or gradient is not finite")
        return loss, grad
