# Agent Memory & Architectural Guide for `adaptive-quantum-time-series`

> **Quick Context for Agents:**
> Refer to [`../AGENTS.md`](file:///c:/Users/thena/QML%20RESEARCH%20PAPER/MODEL/AGENTS.md) for the complete root blueprint.

### Essential Quick Links & File Roles:
- **Core PQC Model:** [`src/quantum_circuit.py`](file:///c:/Users/thena/QML%20RESEARCH%20PAPER/MODEL/adaptive-quantum-time-series/src/quantum_circuit.py) — Defines `PQCModel`, exact analytical gradients via Parameter-Shift Rule, and Qiskit Aer's `EstimatorV2`.
- **Adaptive Windowing Controller:** [`src/adaptive_window.py`](file:///c:/Users/thena/QML%20RESEARCH%20PAPER/MODEL/adaptive-quantum-time-series/src/adaptive_window.py) — Dynamic window expansion and contraction controller with hysteresis safeguards.
- **Pre-Training & Streaming:** [`src/trainer.py`](file:///c:/Users/thena/QML%20RESEARCH%20PAPER/MODEL/adaptive-quantum-time-series/src/trainer.py) — `train_offline_pqc` with closed-form readout ridge initialization and `run_online` test-then-train loop.
- **Classical Baselines:** [`src/classical_baselines.py`](file:///c:/Users/thena/QML%20RESEARCH%20PAPER/MODEL/adaptive-quantum-time-series/src/classical_baselines.py) — Scikit-Learn `MLPOnlineLearner` and PyTorch `LSTMOnlineLearner`.
- **Evaluation:** [`src/evaluation.py`](file:///c:/Users/thena/QML%20RESEARCH%20PAPER/MODEL/adaptive-quantum-time-series/src/evaluation.py) — Error metrics, regime adaptation delay, and reporting.
- **Visuals:** [`src/visualization.py`](file:///c:/Users/thena/QML%20RESEARCH%20PAPER/MODEL/adaptive-quantum-time-series/src/visualization.py) — 6 publication-ready figures + circuit export + entanglement entropy.
- **CLI Entry:** [`src/main.py`](file:///c:/Users/thena/QML%20RESEARCH%20PAPER/MODEL/adaptive-quantum-time-series/src/main.py) — Modes: `smoke`, `train`, `benchmark`, `generate`.

### Execution Commands:
```bash
# Run all tests
.\.venv\Scripts\python.exe -m pytest -v

# Run smoke test
.\.venv\Scripts\python.exe -m src.main --mode smoke

# Run multi-seed benchmark
.\.venv\Scripts\python.exe -m src.main --mode benchmark --config config/config.yaml
```

### High-Impact Improvements to Explore:
1. **NISQ Noise Models & Error Mitigation:** Test under depolarizing noise with ZNE or TREX.
2. **Bayesian Change-Point Detection (BOCPD):** Upgrade heuristic error thresholds to posterior run-length estimation.
3. **Multi-Step Ahead Forecasting:** Direct multi-qubit readout or recursive rollout.
4. **Real-World Non-Stationary Time Series:** Ingest real financial/sensor datasets.
