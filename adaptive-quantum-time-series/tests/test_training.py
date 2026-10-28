"""Unit and integration tests for Module G & H: training and streaming loops."""

import numpy as np
import pytest

from src.adaptive_window import AdaptiveWindowController
from src.classical_baselines import MLPOnlineLearner, train_offline_mlp
from src.config import MLPConfig, OptimizerConfig, QuantumConfig, TrainingConfig
from src.loss_function import PQCObjective
from src.quantum_circuit import PQCModel
from src.trainer import (
    PQCOnlineLearner,
    fit_readout_head,
    run_online,
    train_offline_pqc,
)


def test_fit_readout_head() -> None:
    cfg = QuantumConfig(n_qubits=2, ansatz_layers=1)
    model = PQCModel(cfg)

    angles = np.zeros((10, 2))
    targets = np.full(10, 2.5)

    params = np.zeros(model.n_params)
    init_params = fit_readout_head(model, params, angles, targets)

    preds = model.predict(angles, init_params)
    # The linear head should achieve close to 2.5
    np.testing.assert_allclose(preds, 2.5, atol=1e-3)


def test_offline_pqc_pretraining_convergence() -> None:
    cfg_q = QuantumConfig(n_qubits=2, ansatz_layers=1)
    model = PQCModel(cfg_q)

    # Simple synthetic task
    angles = np.array([[0.1, 0.2], [0.5, 0.8], [1.0, 1.2], [1.5, 0.5]])
    targets = np.sin(angles[:, 0])

    train_cfg = TrainingConfig(
        optimizer=OptimizerConfig(name="adam", learning_rate=0.1),
        iterations=10,
        eval_every=1,
    )

    res = train_offline_pqc(model, train_cfg, angles, targets, angles, targets, seed=42)
    assert res.success
    # history records iteration 0 (initial) through iteration 10
    assert len(res.history) == 11
    assert res.best_val_loss <= res.history[0]["val_loss"]


def test_streaming_loop_temporal_causality() -> None:
    cfg_q = QuantumConfig(n_qubits=2, ansatz_layers=1)
    model = PQCModel(cfg_q)
    params = np.ones(model.n_params)

    opt_cfg = OptimizerConfig(name="adam", learning_rate=0.05)
    learner = PQCOnlineLearner(model, params, opt_cfg, steps_per_update=1, seed=42)
    policy = AdaptiveWindowController()

    N = 25
    features = np.random.uniform(0, np.pi, size=(N, 2))
    targets = np.random.randn(N)
    target_time = np.arange(10, 10 + N)
    stream_idx = np.arange(5, N)

    result = run_online(
        name="test_pqc",
        learner=learner,
        policy=policy,
        features=features,
        targets=targets,
        target_time=target_time,
        stream_idx=stream_idx,
    )

    assert len(result.records) == len(stream_idx)
    # Verify strict causality on every step
    for rec in result.records:
        if rec.window_start_time != -1:
            # Training window bounds must be <= current target time
            assert rec.window_start_time <= rec.time_index
            assert rec.window_end_time <= rec.time_index


def test_mlp_baseline_conforms_to_online_learner() -> None:
    cfg = MLPConfig(hidden_layers=(8,), epochs=10, online_epochs=1)
    X = np.random.randn(20, 3)
    y = np.random.randn(20)

    learner = train_offline_mlp(cfg, X, y, seed=42)
    assert learner.n_parameters > 0

    pred = learner.predict(X[:2])
    assert pred.shape == (2,)

    info = learner.update(X[:5], y[:5])
    assert info.success
    assert np.isfinite(info.loss)
