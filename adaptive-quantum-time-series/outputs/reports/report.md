# Evaluation Report: smoke

## 1. Executive Summary & Benchmark Comparison

| Model | MSE | RMSE | MAE | MAPE (%) | Mean Adaptation Delay (steps) | Total Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Adaptive-Window PQC** | 0.1030 ± 0.0000 | 0.3209 ± 0.0000 | 0.2378 ± 0.0000 | 48.17% | 1.0 | 2.28s |
| **Fixed-Window PQC (W=15)** | 0.0448 ± 0.0000 | 0.2117 ± 0.0000 | 0.1601 ± 0.0000 | 30.04% | 1.0 | 4.26s |
| **Fixed-Window PQC (W=10)** | 0.0489 ± 0.0000 | 0.2212 ± 0.0000 | 0.1699 ± 0.0000 | 32.06% | 1.0 | 3.92s |
| **Frozen PQC (No Adaptation)** | 0.0611 ± 0.0000 | 0.2472 ± 0.0000 | 0.1973 ± 0.0000 | 38.31% | 1.5 | 0.81s |
| **Classical LSTM** | 0.0765 ± 0.0000 | 0.2767 ± 0.0000 | 0.2125 ± 0.0000 | 42.80% | 2.5 | 0.70s |
| **Fixed-Window PQC (W=5)** | 0.0975 ± 0.0000 | 0.3122 ± 0.0000 | 0.2362 ± 0.0000 | 47.73% | 1.0 | 3.35s |
| **Classical MLP** | 0.3868 ± 0.0000 | 0.6219 ± 0.0000 | 0.5060 ± 0.0000 | 89.62% | 6.5 | 0.33s |

## 2. Regime Shift Analysis

This table evaluates how quickly models adapt following known non-stationary regime transitions (frequency jumps, amplitude spikes, variance shifts):

### Seed 7 Regime Shift Details

| Model | Transition | Pre-Shift MAE | Post-Shift MAE | Delay Steps | Recovered |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Adaptive-Window PQC | stable -> frequency_jump | nan | 0.1367 | 1 | True |
| Adaptive-Window PQC | frequency_jump -> amplitude_up | 0.1203 | 0.3310 | 1 | True |
| Fixed-Window PQC (W=5) | stable -> frequency_jump | nan | 0.1367 | 1 | True |
| Fixed-Window PQC (W=5) | frequency_jump -> amplitude_up | 0.1203 | 0.3341 | 1 | True |
| Fixed-Window PQC (W=10) | stable -> frequency_jump | nan | 0.1079 | 1 | True |
| Fixed-Window PQC (W=10) | frequency_jump -> amplitude_up | 0.1199 | 0.2945 | 1 | True |
| Fixed-Window PQC (W=15) | stable -> frequency_jump | nan | 0.1140 | 1 | True |
| Fixed-Window PQC (W=15) | frequency_jump -> amplitude_up | 0.1110 | 0.2856 | 1 | True |
| Frozen PQC (No Adaptation) | stable -> frequency_jump | nan | 0.1149 | 1 | True |
| Frozen PQC (No Adaptation) | frequency_jump -> amplitude_up | 0.1570 | 0.2573 | 2 | True |
| Classical MLP | stable -> frequency_jump | nan | 0.6421 | 5 | True |
| Classical MLP | frequency_jump -> amplitude_up | 0.4933 | 0.9159 | 8 | True |
| Classical LSTM | stable -> frequency_jump | nan | 0.4776 | 4 | True |
| Classical LSTM | frequency_jump -> amplitude_up | 0.1354 | 0.1083 | 1 | True |

## 3. Methodological Discussion & Limitations

- **Simulation Scope**: Evaluated on statevector simulation (Qiskit Aer / Statevector). Does not claim quantum computational supremacy over classical hardware.
- **Parameter-Shift Rule**: Exact analytical quantum gradients were estimated across the parameterized circuit ansatz.
- **Dynamic Windowing**: Shortening the window size after a regime boundary prevents stale observations from corrupting the gradient updates, while expanding the window in stationary phases reduces variance.
