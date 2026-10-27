"""Module G - offline pre-training and the integrated streaming loop.

Two clearly separated phases:

1. **Offline training** (:func:`train_offline_pqc`): full-batch optimisation on
   the training split; the validation split is used only for model selection
   (best-validation parameters are kept) and optional early stopping.

2. **Online adaptation** (:func:`run_online`): a prequential *test-then-train*
   loop over the test split.  At stream step k with target time t:

       a) predict yhat_t from features built from z_{t-L..t-1}
          using parameters fitted only on targets with time < t;
       b) observe y_t, compute e_t = |y_t - yhat_t|;
       c) the window policy updates W from e_t;
       d) the learner is re-fitted (warm start, ``steps_per_update`` optimizer
          steps) on the W most recent labelled pairs, whose targets lie in
          (t - W, t];
       e) advance to t + 1.

   Hence no future target is ever used to produce a prediction or to choose
   an input window.  Every step records the window bounds so this can be
   verified (see ``tests/test_training.py``).

Computational cost of online adaptation with Adam: each update evaluates
(2P + 1) * W_eff circuits per optimizer step (parameter-shift), so a smaller
window is also cheaper.  The total number of circuit evaluations is recorded.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from src.adaptive_window import AdaptiveWindowController, FixedWindowPolicy, WindowEvent
from src.config import OptimizerConfig, TrainingConfig
from src.loss_function import NumericalError, PQCObjective, mse
from src.optimizer import make_optimizer
from src.quantum_circuit import PQCModel


# --------------------------------------------------------------------------- #
# Offline training
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OfflineResult:
    """Outcome of offline PQC pre-training."""

    params: np.ndarray
    final_params: np.ndarray
    initial_params: np.ndarray
    history: tuple[dict[str, float], ...]
    best_iteration: int
    best_val_loss: float
    n_iterations: int
    n_function_evals: int
    n_circuit_evaluations: int
    train_time_s: float
    success: bool
    message: str


def fit_readout_head(
    model: PQCModel, params: np.ndarray, angles: np.ndarray, targets: np.ndarray, ridge: float = 1e-6
) -> np.ndarray:
    """Closed-form ridge initialisation of the linear readout head (w, b).

    Solves min_{w,b} ||E w + b - y||^2 + ridge ||w||^2 for the *initial*
    circuit angles, giving the variational optimizer a sensible starting
    point.  Returns a new parameter vector.
    """
    theta, _, _ = model.split(params)
    e = model.expectations(angles, theta)
    design = np.hstack([e, np.ones((e.shape[0], 1))])
    reg = ridge * np.eye(design.shape[1])
    reg[-1, -1] = 0.0
    coef = np.linalg.solve(design.T @ design + reg, design.T @ np.asarray(targets, dtype=float))
    if not np.all(np.isfinite(coef)):
        raise NumericalError("readout-head initialisation produced non-finite values")
    return np.concatenate([theta, coef])


def train_offline_pqc(
    model: PQCModel,
    angles_train: np.ndarray,
    y_train: np.ndarray,
    angles_val: np.ndarray,
    y_val: np.ndarray,
    cfg: TrainingConfig,
    seed: int,
) -> OfflineResult:
    """Pre-train the PQC on the training split, selecting by validation MSE."""
    rng = np.random.default_rng(seed)
    evals_before = model.estimator.n_circuit_evaluations
    start = time.perf_counter()
    params0 = fit_readout_head(model, model.init_params(rng, cfg.init_scale), angles_train, y_train)
    objective = PQCObjective(model, angles_train, y_train, cfg.optimizer.l2_reg)
    optimizer = make_optimizer(cfg.optimizer, seed)

    def losses_at(p: np.ndarray) -> tuple[float, float]:
        return mse(model.predict(angles_train, p), y_train), mse(model.predict(angles_val, p), y_val)

    tr0, va0 = losses_at(params0)
    history: list[dict[str, float]] = [{"iteration": 0, "train_loss": tr0, "val_loss": va0}]
    best = {"params": params0.copy(), "val": va0, "iteration": 0, "stale": 0}

    def callback(i: int, x: np.ndarray, _loss: float) -> bool:
        if (i + 1) % cfg.eval_every and i + 1 != cfg.iterations:
            return False
        tr, va = losses_at(x)
        history.append({"iteration": i + 1, "train_loss": tr, "val_loss": va})
        if va < best["val"]:
            best.update(params=x.copy(), val=va, iteration=i + 1, stale=0)
        else:
            best["stale"] += 1
        patience = cfg.early_stopping_patience
        return bool(patience and best["stale"] >= patience)

    result = optimizer.run(objective, params0, cfg.iterations, callback)
    elapsed = time.perf_counter() - start
    return OfflineResult(
        params=best["params"],
        final_params=result.params,
        initial_params=params0,
        history=tuple(history),
        best_iteration=int(best["iteration"]),
        best_val_loss=float(best["val"]),
        n_iterations=result.n_iterations,
        n_function_evals=result.n_function_evals,
        n_circuit_evaluations=model.estimator.n_circuit_evaluations - evals_before,
        train_time_s=elapsed,
        success=result.success,
        message=result.message,
    )


# --------------------------------------------------------------------------- #
# Online learners
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class UpdateInfo:
    """Result of one online re-fit."""

    loss: float
    n_iterations: int
    success: bool
    message: str = ""


class OnlineLearner(Protocol):
    """Interface shared by the PQC and the classical baselines."""

    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def update(self, X: np.ndarray, y: np.ndarray) -> UpdateInfo: ...

    def checkpoint_arrays(self) -> dict[str, np.ndarray]: ...

    @property
    def n_parameters(self) -> int: ...


class PQCOnlineLearner:
    """Warm-started online re-fitting of a PQC on the current window."""

    def __init__(
        self,
        model: PQCModel,
        params: np.ndarray,
        opt_cfg: OptimizerConfig,
        steps_per_update: int,
        seed: int,
    ) -> None:
        self.model = model
        self.params = np.asarray(params, dtype=float).copy()
        self._opt_cfg = opt_cfg
        self._optimizer = make_optimizer(opt_cfg, seed)
        self._steps = steps_per_update

    @property
    def n_parameters(self) -> int:
        return self.model.n_params

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X, self.params)

    def update(self, X: np.ndarray, y: np.ndarray) -> UpdateInfo:
        objective = PQCObjective(self.model, X, y, self._opt_cfg.l2_reg)
        result = self._optimizer.run(objective, self.params, self._steps)
        if result.success and np.all(np.isfinite(result.params)):
            self.params = result.params
        loss = result.losses[-1] if result.losses else float("nan")
        return UpdateInfo(loss, result.n_iterations, result.success, result.message)

    def checkpoint_arrays(self) -> dict[str, np.ndarray]:
        return {"params": self.params.copy()}


# --------------------------------------------------------------------------- #
# Streaming loop
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StepRecord:
    """Everything that happened at one streaming step."""

    step: int
    sample_index: int
    time_index: int
    y_true: float
    y_pred: float
    abs_error: float
    window: int
    effective_window: int
    window_start_time: int
    window_end_time: int
    update_loss: float
    update_success: bool
    predict_time_s: float
    update_time_s: float


@dataclass(frozen=True)
class OnlineResult:
    """Outcome of one model's pass over the stream."""

    name: str
    records: tuple[StepRecord, ...]
    window_events: tuple[WindowEvent, ...]
    n_updates: int
    n_failed_updates: int
    n_optimizer_iterations: int
    predict_time_s: float
    update_time_s: float
    checkpoints: tuple[str, ...] = field(default_factory=tuple)

    def column(self, name: str) -> np.ndarray:
        return np.array([getattr(r, name) for r in self.records])


WindowPolicy = AdaptiveWindowController | FixedWindowPolicy


def run_online(
    name: str,
    learner: OnlineLearner,
    policy: WindowPolicy | None,
    features: np.ndarray,
    targets: np.ndarray,
    target_time: np.ndarray,
    stream_idx: np.ndarray,
    update_every: int = 1,
    checkpoint_every: int = 0,
    checkpoint_dir: Path | None = None,
) -> OnlineResult:
    """Run the causal test-then-train loop (see module docstring).

    Args:
        name: Model name used in records and checkpoint file names.
        learner: Model implementing :class:`OnlineLearner`.
        policy: Window policy; ``None`` disables online updates (frozen model).
        features: (N, d) features for every supervised sample.
        targets: (N,) normalised targets.
        target_time: (N,) strictly increasing time index of each target.
        stream_idx: Consecutive sample indices to stream over.
        update_every: Re-fit every ``update_every`` steps.
        checkpoint_every: Save learner state every k steps (0 disables).
        checkpoint_dir: Directory for checkpoints (required if enabled).
    """
    stream = np.asarray(stream_idx, dtype=int)
    if stream.size == 0:
        raise ValueError("stream is empty")
    if np.any(np.diff(target_time) <= 0):
        raise ValueError("target_time must be strictly increasing")
    if np.any(np.diff(stream) != 1):
        raise ValueError("stream indices must be consecutive")
    if update_every < 1:
        raise ValueError("update_every must be >= 1")
    if checkpoint_every and checkpoint_dir is None:
        raise ValueError("checkpoint_dir is required when checkpoint_every > 0")

    records: list[StepRecord] = []
    checkpoints: list[str] = []
    n_updates = n_failed = n_iters = 0
    total_pred = total_upd = 0.0
    for step, s in enumerate(stream):
        t = int(target_time[s])
        t0 = time.perf_counter()
        y_hat = float(learner.predict(features[s : s + 1])[0])
        pred_time = time.perf_counter() - t0
        if not np.isfinite(y_hat):
            raise NumericalError(f"{name}: non-finite prediction at time {t}")
        err = abs(float(targets[s]) - y_hat)

        window = eff = 0
        w_start = w_end = -1
        upd_loss, upd_ok, upd_time = float("nan"), True, 0.0
        if policy is not None:
            window = policy.update(err, t)
            # samples 0..s have targets observed at or before time t
            eff = policy.effective_window(available=s + 1)
            if step % update_every == 0:
                win = np.arange(s - eff + 1, s + 1)
                w_start, w_end = int(target_time[win[0]]), int(target_time[win[-1]])
                t1 = time.perf_counter()
                info = learner.update(features[win], targets[win])
                upd_time = time.perf_counter() - t1
                upd_loss, upd_ok = info.loss, info.success
                n_updates += 1
                n_failed += int(not info.success)
                n_iters += info.n_iterations
        total_pred += pred_time
        total_upd += upd_time
        records.append(
            StepRecord(
                step=step,
                sample_index=int(s),
                time_index=t,
                y_true=float(targets[s]),
                y_pred=y_hat,
                abs_error=err,
                window=window,
                effective_window=eff,
                window_start_time=w_start,
                window_end_time=w_end,
                update_loss=float(upd_loss),
                update_success=bool(upd_ok),
                predict_time_s=pred_time,
                update_time_s=upd_time,
            )
        )
        if checkpoint_every and (step + 1) % checkpoint_every == 0:
            assert checkpoint_dir is not None
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            path = checkpoint_dir / f"{name}_step{step + 1}.npz"
            np.savez(path, time_index=t, **learner.checkpoint_arrays())
            checkpoints.append(str(path))

    events = policy.history if policy is not None else ()
    return OnlineResult(
        name=name,
        records=tuple(records),
        window_events=tuple(events),
        n_updates=n_updates,
        n_failed_updates=n_failed,
        n_optimizer_iterations=n_iters,
        predict_time_s=total_pred,
        update_time_s=total_upd,
        checkpoints=tuple(checkpoints),
    )
