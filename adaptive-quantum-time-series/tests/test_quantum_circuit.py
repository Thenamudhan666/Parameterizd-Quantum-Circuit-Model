"""Unit tests for Module D: Parameterized Quantum Circuit (PQC)."""

import numpy as np
import pytest

from src.config import QuantumConfig
from src.quantum_circuit import PQCModel


def test_pqc_construction_and_parameters() -> None:
    cfg = QuantumConfig(
        n_qubits=3,
        ansatz_layers=2,
        rotation_gates=("ry", "rz"),
        entangler="cx",
    )
    model = PQCModel(cfg)

    # 3 qubits x 2 gates/qubit x 2 layers = 12 rotation angles
    # + 3 readout weights + 1 bias = 16 total parameters
    assert model.n_qubits == 3
    assert model.n_angles == 12
    assert model.n_readout == 3
    assert model.n_params == 16


def test_pqc_forward_prediction() -> None:
    cfg = QuantumConfig(n_qubits=2, ansatz_layers=1, rotation_gates=("ry",))
    model = PQCModel(cfg)

    angles = np.array([[0.5, 1.0], [1.2, 0.3]])
    params = np.ones(model.n_params)

    preds = model.predict(angles, params)
    assert preds.shape == (2,)
    assert np.all(np.isfinite(preds))


def test_parameter_shift_gradient_matches_finite_difference() -> None:
    from src.loss_function import PQCObjective

    cfg = QuantumConfig(n_qubits=2, ansatz_layers=1, rotation_gates=("ry",))
    model = PQCModel(cfg)

    np.random.seed(42)
    angles = np.random.uniform(0, np.pi, size=(4, 2))
    targets = np.array([0.5, -0.2, 0.8, -0.5])
    params = np.random.uniform(-0.5, 0.5, size=model.n_params)

    objective = PQCObjective(model, angles, targets)
    loss, grad = objective.value_and_grad(params)

    # Numerical finite-difference check across all parameters
    eps = 1e-5
    num_grad = np.zeros_like(params)
    for i in range(len(params)):
        p_plus = params.copy()
        p_plus[i] += eps
        p_minus = params.copy()
        p_minus[i] -= eps
        num_grad[i] = (objective(p_plus) - objective(p_minus)) / (2 * eps)

    np.testing.assert_allclose(grad, num_grad, atol=1e-4, rtol=1e-3)
