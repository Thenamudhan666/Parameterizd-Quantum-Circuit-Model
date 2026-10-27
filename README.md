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
│   ├── src/
│   │   ├── __init__.py          # Module initialization
│   │   ├── adaptive_window.py   # Adaptive spectral windowing (leakage mitigation)
│   │   ├── config.py            # Typed dataclass configs & validation
│   │   ├── data_generator.py    # Synthetic & benchmark time series generators
│   │   ├── loss_function.py     # Loss formulations (MSE, Huber, Spectral Loss)
│   │   ├── optimizer.py         # Quantum-aware optimizers (Adam, SGD, Momentum)
│   │   ├── preprocessing.py     # Rolling window features & normalization
│   │   ├── quantum_circuit.py   # PQC circuit generator & expectation evaluator
│   │   ├── quantum_encoder.py   # Angle/amplitude encoding & entanglement topology
│   │   └── trainer.py           # Training loop, validation, metrics & checkpointing
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
   python -m venv .venv
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

## Quickstart

Run a fast smoke test using `smoke.yaml`:
```bash
python -m src.trainer --config config/smoke.yaml
```

To run complete training with full configuration:
```bash
python -m src.trainer --config config/config.yaml
```

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

---

## License
MIT License
