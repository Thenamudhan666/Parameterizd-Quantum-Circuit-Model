"""Module D - parameterized quantum circuit (PQC) regression model.

Circuit structure (n qubits, L_eff = max(ansatz_layers, n_encoding_blocks)):

    for k in range(L_eff):
        if k < n_blocks:  U_enc(x; block k)            # data (re-)uploading
        U_layer(theta_k) = prod_q prod_g R_g(theta_{k,q,g})  ;  entangler(CX/CZ)

Readout: expectation values of Pauli-Z on the readout qubits

    E_q(x, theta) = <psi(x, theta)| Z_q |psi(x, theta)>   in [-1, 1]

Output transformation (documented, trainable, classical):

    yhat(x) = sum_q w_q E_q(x, theta) + b

so the prediction is a numerical value in normalised units that can exceed the
[-1, 1] range of a single expectation value (needed after amplitude shifts).
All input information flows through the quantum circuit; the head only
rescales measured expectation values.

Gradients: every trainable angle theta_j appears in exactly one Pauli rotation
R_P(theta) = exp(-i theta P / 2), so the exact parameter-shift rule holds:

    dE/dtheta_j = [E(theta + pi/2 e_j) - E(theta - pi/2 e_j)] / 2

All (2P + 1) x M circuit evaluations of one gradient are submitted as a single
broadcasted Estimator PUB, which Qiskit Aer evaluates efficiently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from src.config import QuantumConfig
from src.loss_function import NumericalError
from src.quantum_encoder import (
    EncodingPlan,
    append_encoding_block,
    apply_entangler,
    entangling_pairs,
)

SHIFT = np.pi / 2.0


@dataclass(frozen=True)
class PQCStructure:
    """The symbolic circuit and its parameter bookkeeping."""

    circuit: QuantumCircuit
    x: ParameterVector
    theta: ParameterVector
    plan: EncodingPlan
    n_layers: int
    observables: tuple[SparsePauliOp, ...]
    readout_qubits: tuple[int, ...]


def build_pqc(cfg: QuantumConfig, n_features: int) -> PQCStructure:
    """Construct the parameterized circuit for ``n_features`` input angles."""
    n = cfg.n_qubits
    plan = EncodingPlan.create(n_features, n)
    n_layers = max(cfg.ansatz_layers, plan.n_blocks)
    x = ParameterVector("x", n_features)
    theta = ParameterVector("theta", n_layers * n * len(cfg.rotation_gates))
    qc = QuantumCircuit(n, name="AdaptivePQC")
    ansatz_pairs = entangling_pairs(n, cfg.ansatz_entanglement)
    k = 0
    for layer in range(n_layers):
        if layer < plan.n_blocks:
            append_encoding_block(
                qc, x, plan.blocks[layer], cfg.encoding, cfg.encoding_entanglement, cfg.entangler
            )
            qc.barrier()
        for q in range(n):
            for gate in cfg.rotation_gates:
                getattr(qc, gate)(theta[k], q)
                k += 1
        apply_entangler(qc, ansatz_pairs, cfg.entangler)
        qc.barrier()
    readout = cfg.resolved_readout_qubits
    observables = tuple(
        SparsePauliOp.from_sparse_list([("Z", [q], 1.0)], num_qubits=n) for q in readout
    )
    return PQCStructure(qc, x, theta, plan, n_layers, observables, readout)


class ExpectationEstimator:
    """Batched exact expectation values through a Qiskit EstimatorV2 primitive.

    ``backend="aer"`` uses :class:`qiskit_aer.primitives.EstimatorV2`, i.e. an
    AerSimulator with the statevector method and ``precision=0`` (exact, no
    shot noise).  ``backend="statevector"`` uses Qiskit's reference
    :class:`qiskit.primitives.StatevectorEstimator`.  Both are noiseless
    simulations and do **not** represent physical hardware.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        observables: tuple[SparsePauliOp, ...],
        backend: str,
        seed: int,
        max_batch_size: int,
    ) -> None:
        self._circuit = circuit
        self._obs = [[o] for o in observables]  # shape (R, 1) broadcasts against (B,)
        self._n_obs = len(observables)
        self._max_batch = int(max_batch_size)
        self.backend = backend
        self.n_calls = 0
        self.n_circuit_evaluations = 0
        if backend == "aer":
            try:
                from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
            except ImportError as exc:  # pragma: no cover - depends on environment
                raise ImportError(
                    "qiskit-aer is required for backend 'aer'; install it via "
                    "'pip install -r requirements.txt' or set quantum.backend: statevector"
                ) from exc
            self._estimator = AerEstimatorV2(
                options={
                    "default_precision": 0.0,
                    "backend_options": {"method": "statevector", "seed_simulator": seed},
                }
            )
        elif backend == "statevector":
            from qiskit.primitives import StatevectorEstimator

            self._estimator = StatevectorEstimator(default_precision=0.0, seed=seed)
        else:
            raise ValueError(f"unknown backend '{backend}'")

    def run(self, values: np.ndarray) -> np.ndarray:
        """Evaluate all observables for every row of circuit-parameter values.

        Args:
            values: (B, n_circuit_parameters) in ``circuit.parameters`` order.

        Returns:
            (B, R) expectation values.
        """
        vals = np.asarray(values, dtype=float)
        if vals.ndim != 2 or vals.shape[1] != self._circuit.num_parameters:
            raise ValueError(
                f"expected values of shape (B, {self._circuit.num_parameters}), got {vals.shape}"
            )
        if not np.all(np.isfinite(vals)):
            raise NumericalError("non-finite circuit parameters supplied to the estimator")
        chunks = []
        for start in range(0, vals.shape[0], self._max_batch):
            chunk = vals[start : start + self._max_batch]
            result = self._estimator.run([(self._circuit, self._obs, chunk)]).result()
            evs = np.asarray(result[0].data.evs, dtype=float).reshape(self._n_obs, chunk.shape[0])
            chunks.append(evs.T)
            self.n_calls += 1
        out = np.concatenate(chunks, axis=0)
        self.n_circuit_evaluations += vals.shape[0]
        if not np.all(np.isfinite(out)):
            raise NumericalError("estimator returned non-finite expectation values")
        return out


class PQCModel:
    """Variational quantum regressor: circuit + estimator + linear readout head.

    The flat parameter vector is ``params = [theta (P), w (R), b (1)]``.
    """

    def __init__(self, cfg: QuantumConfig, n_features: int, seed: int = 0) -> None:
        self.config = cfg
        self.structure = build_pqc(cfg, n_features)
        self.n_features = n_features
        qc = self.structure.circuit
        position = {p: i for i, p in enumerate(qc.parameters)}
        self._x_cols = np.array([position[p] for p in self.structure.x], dtype=int)
        self._theta_cols = np.array([position[p] for p in self.structure.theta], dtype=int)
        self.estimator = ExpectationEstimator(
            qc, self.structure.observables, cfg.backend, seed, cfg.max_batch_size
        )

    # ------------------------------------------------------------------ sizes
    @property
    def circuit(self) -> QuantumCircuit:
        return self.structure.circuit

    @property
    def n_theta(self) -> int:
        return len(self.structure.theta)

    @property
    def n_readout(self) -> int:
        return len(self.structure.observables)

    @property
    def n_params(self) -> int:
        return self.n_theta + self.n_readout + 1

    # ----------------------------------------------------------- parameters
    def split(self, params: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        p = np.asarray(params, dtype=float)
        if p.shape != (self.n_params,):
            raise ValueError(f"expected {self.n_params} parameters, got shape {p.shape}")
        return p[: self.n_theta], p[self.n_theta : self.n_theta + self.n_readout], float(p[-1])

    def init_params(self, rng: np.random.Generator, init_scale: float) -> np.ndarray:
        """Small random angles (near-identity ansatz) and a uniform readout head."""
        theta = rng.uniform(-np.pi, np.pi, size=self.n_theta) * init_scale
        w = np.full(self.n_readout, 1.0 / self.n_readout)
        return np.concatenate([theta, w, [0.0]])

    # ------------------------------------------------------------ execution
    def _values(self, angles: np.ndarray, thetas: np.ndarray) -> np.ndarray:
        """Circuit-parameter matrix for every (theta row, sample) combination."""
        a = np.atleast_2d(np.asarray(angles, dtype=float))
        if a.shape[1] != self.n_features:
            raise ValueError(f"expected {self.n_features} input features, got {a.shape[1]}")
        th = np.atleast_2d(thetas)
        m, k = a.shape[0], th.shape[0]
        values = np.empty((k * m, self.circuit.num_parameters))
        values[:, self._x_cols] = np.tile(a, (k, 1))
        values[:, self._theta_cols] = np.repeat(th, m, axis=0)
        return values

    def expectations(self, angles: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """(M, R) Pauli-Z expectation values for each input row."""
        return self.estimator.run(self._values(angles, theta))

    def expectations_and_jacobian(
        self, angles: np.ndarray, theta: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Expectation values (M, R) and their parameter-shift Jacobian (M, R, P)."""
        p = self.n_theta
        th = np.asarray(theta, dtype=float)
        shifts = np.eye(p) * SHIFT
        thetas = np.vstack([th[None, :], th + shifts, th - shifts])
        m = np.atleast_2d(angles).shape[0]
        all_e = self.estimator.run(self._values(angles, thetas)).reshape(2 * p + 1, m, -1)
        jac = 0.5 * (all_e[1 : p + 1] - all_e[p + 1 :])  # (P, M, R)
        return all_e[0], np.transpose(jac, (1, 2, 0))

    def predict(self, angles: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Numerical predictions (normalised units) for each input row."""
        theta, w, b = self.split(params)
        return self.expectations(angles, theta) @ w + b

    def statevector(self, angles_row: np.ndarray, theta: np.ndarray) -> Statevector:
        """Exact statevector for one input (used for entanglement analysis)."""
        binding = dict(zip(self.circuit.parameters, self._values(angles_row, theta)[0]))
        return Statevector(self.circuit.assign_parameters(binding))
