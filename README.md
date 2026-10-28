# Parameterized Quantum Circuit (PQC) Model for Adaptive Quantum Time Series Regression

An end-to-end Quantum Machine Learning (QML) framework implementing **Parameterized Quantum Circuits (PQC)** with **data re-uploading**, **variational ansatz layers**, **parameter-shift rule analytical gradients**, and **adaptive window conditioning** to eliminate spectral leakage in non-periodic time series analysis and regression.

---

## Architecture Overview

```
Time Series Data x[t] ──► [Adaptive Window / Preprocessing] ──► Feature Vector x
                                                                      │
      ┌───────────────────────────────────────────────────────────────┘
      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Parameterized Quantum Circuit (PQC)                  │
│                                                                         │
│   |0⟩ ──► [ U_enc(x; 0) ] ──► [ U_var(θ_0) ] ──► [CX/CZ] ──► ... ──► ⟨Z_0⟩
│   |0⟩ ──► [ U_enc(x; 1) ] ──► [ U_var(θ_1) ] ──► [CX/CZ] ──► ... ──► ⟨Z_1⟩
│   ...                                                                ... 
│   |0⟩ ──► [ U_enc(x; n) ] ──► [ U_var(θ_n) ] ──► [CX/CZ] ──► ... ──► ⟨Z_q⟩
└────────────────────────────────────┬────────────────────────────────────┘
                                     │  Expectation Values E_q(x, θ) ∈ [-1, 1]
                                     ▼
                      ┌──────────────────────────────┐
                      │    Trainable Readout Head    │
                      │  y_hat = Σ_q w_q E_q(x, θ) + b│
                      └──────────────┬───────────────┘
                                     ▼
                           Predicted Value y_hat
```

### Key Pillars:
1. **Data Re-Uploading & Expressibility**:
   Interleaved encoding blocks $U_{\text{enc}}(x; \text{block } k)$ and variational layers $U_{\text{layer}}(\theta_k)$ overcome the single-qubit expressibility barrier, enabling approximation of complex Fourier frequency spectra.
2. **Analytical Quantum Gradients (Parameter-Shift Rule)**:
   Exact partial derivatives with respect to variational parameters are computed without numerical finite-difference approximation:
   $$\frac{\partial E}{\partial \theta_j} = \frac{E(\theta + \frac{\pi}{2} e_j) - E(\theta - \frac{\pi}{2} e_j)}{2}$$
3. **Adaptive Window Conditioning**:
   Dynamic tapering (Hann, raised-cosine, Tukey, Gaussian) mitigating boundary cliff discontinuities and Dirichlet sinc sidelobes before quantum amplitude or angle encoding.
4. **Classical Trainable Readout**:
   Measured Pauli-$Z$ expectations are mapped through a trainable linear head:
   $$\hat{y}(x) = \sum_{q} w_q E_q(x, \theta) + b$$
   ensuring the model can predict outside $[-1, 1]$ while strictly routing all non-linear information processing through the quantum circuit.

---

## Project Structure

```
Parameterizd-Quantum-Circuit-Model/
├── adaptive-quantum-time-series/
│   ├── config/
│   │   ├── config.yaml          # Full training & architecture configuration
│   │   └── smoke.yaml           # Fast smoke test configuration
│   ├── experiments/
│   │   └── run_benchmarks.py    # Multi-seed benchmarking experiment runner
│   ├── outputs/
│   │   ├── plots/               # Generated high-resolution publication figures
│   │   └── reports/             # JSON summary and Markdown evaluation reports
│   ├── src/
│   │   ├── __init__.py          # Module initialization
│   │   ├── adaptive_window.py   # Adaptive spectral windowing (leakage mitigation)
│   │   ├── classical_baselines.py # Scikit-Learn MLP & PyTorch LSTM baselines
│   │   ├── config.py            # Typed dataclass configs & validation
│   │   ├── data_generator.py    # Synthetic & benchmark time series generators
│   │   ├── evaluation.py        # Accuracy & regime-shift adaptation metrics
│   │   ├── loss_function.py     # Loss formulations (MSE, regularized objective)
│   │   ├── main.py              # Central orchestrator CLI (smoke/train/benchmark)
│   │   ├── optimizer.py         # Quantum-aware optimizers (Adam, SPSA, COBYLA)
│   │   ├── preprocessing.py     # Rolling window features, scalers, & angle mapping
│   │   ├── quantum_circuit.py   # PQC circuit generator & expectation evaluator
│   │   ├── quantum_encoder.py   # Angle/amplitude encoding & entanglement topology
│   │   ├── trainer.py           # Training loop, streaming adaptation & checkpoints
│   │   └── visualization.py     # Multi-panel plots, error thresholds, & circuit export
│   ├── tests/                   # 23 automated unit and integration tests (pytest)
│   ├── pytest.ini               # Pytest configuration
│   └── requirements.txt         # Python package dependencies
├── .gitignore                   # Excludes virtual environments & build artifacts
└── README.md                    # Project documentation
```

---

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Thenamudhan666/Parameterizd-Quantum-Circuit-Model.git
   cd Parameterizd-Quantum-Circuit-Model/adaptive-quantum-time-series
   ```

2. **Create and activate a virtual environment**:
   ```bash
   py -3.12 -m venv .venv
   # Windows PowerShell
   .venv\Scripts\Activate.ps1
   # Linux / macOS
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage & Execution Modes

### 1. Rapid Smoke Test
Execute an end-to-end pipeline run (data generation, PQC pre-training, online adaptation, baselines, plots, and report generation):
```bash
python -m src.main --mode smoke
```

### 2. Multi-Seed Benchmark Experiments
Run comprehensive evaluation comparing **Adaptive-Window PQC**, **Fixed-Window PQCs ($W=10, 30, 50$)**, **Frozen PQC**, **Classical MLP**, and **Classical LSTM** across multiple random seeds:
```bash
python -m src.main --mode benchmark --config config/config.yaml
# Or via the experiment script
python experiments/run_benchmarks.py --config config/config.yaml
```

### 3. Single-Seed PQC Training
```bash
python -m src.main --mode train --config config/config.yaml
```

### 4. Synthetic Data Generation & Plotting
```bash
python -m src.main --mode generate --config config/config.yaml
```

### 5. Automated Test Suite
Run the 23 unit and integration tests:
```bash
pytest -v
```

---

## Benchmark Performance (Smoke Evaluation / Seed 7)

Evaluated on non-stationary regime-switching time series with frequency jumps and amplitude scaling:

| Model | MSE (Physical Units) | RMSE | MAE | MAPE (%) | Mean Adaptation Delay | Total Runtime |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Fixed-Window PQC ($W=15$)** | **0.0448** | 0.2117 | 0.1601 | 30.04% | 1.0 step | 4.26s |
| **Fixed-Window PQC ($W=10$)** | **0.0489** | 0.2212 | 0.1699 | 32.06% | 1.0 step | 3.92s |
| **Frozen PQC (No Adaptation)** | 0.0611 | 0.2472 | 0.1973 | 38.31% | 1.5 steps | 0.81s |
| **Classical LSTM** | 0.0765 | 0.2767 | 0.2125 | 42.80% | 2.5 steps | 0.70s |
| **Fixed-Window PQC ($W=5$)** | 0.0975 | 0.3122 | 0.2362 | 47.73% | 1.0 step | 3.35s |
| **Adaptive-Window PQC** | 0.1030 | 0.3209 | 0.2378 | 48.17% | 1.0 step | 2.28s |
| **Classical MLP** | 0.3868 | 0.6219 | 0.5060 | 89.62% | 6.5 steps | 0.33s |

*Note: Results obtained via exact statevector simulation using Qiskit Aer with analytical parameter-shift gradient computation. The Adaptive-Window PQC dynamically adjusts temporal horizon $W_t$ in response to real-time prediction error $\epsilon_t$, adapting to regime shifts in 1.0 step with reduced circuit evaluation overhead.*

---

## Generated Visualizations & Reports

All artifacts are automatically exported to `adaptive-quantum-time-series/outputs/`:
- `plots/01_time_series_regimes.png`: Non-stationary synthetic signal with labeled regime shifts.
- `plots/02_predictions_comparison.png`: PQC vs classical baseline trajectory comparison.
- `plots/03_adaptive_window_dynamics.png`: Dynamic window resizing ($W_t$) and error thresholds ($\tau_{\text{low}}, \tau_{\text{high}}$).
- `plots/04_benchmark_model_comparison.png`: Bar chart comparison of MSE and regime recovery delay.
- `plots/05_quantum_circuit_diagram.png` / `.txt`: Parameterized quantum circuit architecture.
- `plots/06_quantum_entanglement_entropy.png`: Bipartite von Neumann entanglement entropy dynamics over time.
- `reports/summary.json` & `reports/report.md`: Complete numerical evaluation tables.

---

## Dependencies
- **Qiskit** ($\ge 2.0, < 3$)
- **Qiskit Aer** ($\ge 0.17$)
- **NumPy** ($\ge 2.0$)
- **SciPy** ($\ge 1.14$)
- **Scikit-learn** ($\ge 1.5$)
- **Matplotlib** ($\ge 3.9$)
- **PyYAML** ($\ge 6.0$)
- **PyTorch** ($\ge 2.0$)
- **Pytest** ($\ge 8.0$)

---

## License
MIT License
