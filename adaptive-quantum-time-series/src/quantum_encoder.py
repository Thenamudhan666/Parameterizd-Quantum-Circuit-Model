"""Module C (quantum part) - classical-to-quantum feature maps.

Features arrive as rotation angles (see ``preprocessing.AngleMapper``).  An
:class:`EncodingPlan` decides which feature is written onto which qubit:

* d == n_qubits : one block, feature i -> qubit i.
* d <  n_qubits : one block, features repeated cyclically (qubit q gets
                  feature q mod d) so every qubit carries data.
* d >  n_qubits : ``ceil(d / n)`` *re-uploading* blocks; block k carries
                  features k*n ... k*n+n-1, each block followed by a trainable
                  ansatz layer (data re-uploading).  Unused slots in the last
                  block are left empty.

Supported encodings (per block, per qubit q with angle a_q):

* ``ry``    : RY(a_q)
* ``ry_rz`` : RY(a_q) RZ(a_q)
* ``zz``    : H, P(2 a_q), then for each entangled pair (i, j):
              CX(i,j) P(2 (pi - a_i)(pi - a_j)) CX(i,j)
              (the second-order Pauli-Z evolution of Havlicek et al. 2019,
              written out gate-by-gate to avoid deprecated library classes).

For ``ry``/``ry_rz`` an optional entangling layer (CX or CZ, linear or circular)
follows each block.  ``zz`` provides its own entanglement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from qiskit.circuit import Parameter, ParameterVector, QuantumCircuit


def entangling_pairs(n_qubits: int, pattern: str) -> list[tuple[int, int]]:
    """Qubit pairs for a linear or circular nearest-neighbour topology."""
    if pattern == "none" or n_qubits < 2:
        return []
    pairs = [(i, i + 1) for i in range(n_qubits - 1)]
    if pattern == "circular" and n_qubits > 2:
        pairs.append((n_qubits - 1, 0))
    elif pattern not in {"linear", "circular"}:
        raise ValueError(f"unknown entanglement pattern '{pattern}'")
    return pairs


def apply_entangler(qc: QuantumCircuit, pairs: list[tuple[int, int]], gate: str) -> None:
    """Append two-qubit entangling gates in place."""
    for a, b in pairs:
        if gate == "cx":
            qc.cx(a, b)
        elif gate == "cz":
            qc.cz(a, b)
        else:
            raise ValueError(f"unknown entangling gate '{gate}'")


@dataclass(frozen=True)
class EncodingPlan:
    """Assignment of feature indices to qubits for each encoding block."""

    n_qubits: int
    n_features: int
    blocks: tuple[tuple[int | None, ...], ...]

    @classmethod
    def create(cls, n_features: int, n_qubits: int) -> "EncodingPlan":
        if n_features < 1 or n_qubits < 1:
            raise ValueError("n_features and n_qubits must be >= 1")
        if n_features <= n_qubits:
            block = tuple(q % n_features for q in range(n_qubits))
            return cls(n_qubits, n_features, (block,))
        n_blocks = math.ceil(n_features / n_qubits)
        blocks = []
        for k in range(n_blocks):
            blocks.append(
                tuple(
                    (k * n_qubits + q) if k * n_qubits + q < n_features else None
                    for q in range(n_qubits)
                )
            )
        return cls(n_qubits, n_features, tuple(blocks))

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)


def append_encoding_block(
    qc: QuantumCircuit,
    x: ParameterVector,
    block: tuple[int | None, ...],
    encoding: str,
    entanglement: str,
    entangler: str,
) -> None:
    """Append one encoding block (in place) using input parameters ``x``."""
    n = qc.num_qubits
    angle: dict[int, Parameter] = {q: x[f] for q, f in enumerate(block) if f is not None}
    if encoding in {"ry", "ry_rz"}:
        for q, a in angle.items():
            qc.ry(a, q)
            if encoding == "ry_rz":
                qc.rz(a, q)
        apply_entangler(qc, entangling_pairs(n, entanglement), entangler)
        return
    if encoding == "zz":
        for q, a in angle.items():
            qc.h(q)
            qc.p(2.0 * a, q)
        pattern = entanglement if entanglement != "none" else "linear"
        for i, j in entangling_pairs(n, pattern):
            if i in angle and j in angle:
                qc.cx(i, j)
                qc.p(2.0 * (math.pi - angle[i]) * (math.pi - angle[j]), j)
                qc.cx(i, j)
        return
    raise ValueError(f"unknown encoding '{encoding}'")
