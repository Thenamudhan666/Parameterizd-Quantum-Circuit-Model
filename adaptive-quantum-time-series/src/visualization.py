"""Module J - Comprehensive visualization and plotting.

Generates high-resolution publication-quality figures:
1. Synthetic time-series with marked regime boundaries;
2. Predictions vs Ground Truth trajectory comparison;
3. Offline pre-training loss convergence curves;
4. Adaptive window-size dynamics over streaming steps;
5. Prediction error trajectory with adaptation thresholds;
6. Baseline benchmark comparison bar charts;
7. Quantum circuit diagram export;
8. Quantum entanglement entropy dynamics across the time series.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # Headless backend for robust script/CI execution
import matplotlib.pyplot as plt
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import partial_trace

from src.data_generator import SyntheticSeries
from src.evaluation import AggregatedModelMetrics, ModelMetrics
from src.preprocessing import SeriesScaler
from src.quantum_circuit import PQCModel
from src.trainer import OfflineResult, OnlineResult


# --------------------------------------------------------------------------- #
# Style helper
# --------------------------------------------------------------------------- #
def _apply_theme() -> None:
    plt.rcParams.update({
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
        "axes.edgecolor": "#CCCCCC",
        "axes.linewidth": 0.8,
        "grid.color": "#E5E5E5",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
    })


# --------------------------------------------------------------------------- #
# 1. Raw Time Series with Regimes
# --------------------------------------------------------------------------- #
def plot_time_series_regimes(
    series: SyntheticSeries,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot the synthetic time series with shaded regime segments and change points."""
    _apply_theme()
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=200)

    t = series.time
    y = series.values
    clean = series.clean

    ax.plot(t, y, label="Noisy Observation $X(t)$", color="#3498DB", alpha=0.6, lw=1.2)
    ax.plot(t, clean, label="Clean Latent Signal", color="#2C3E50", lw=1.5, ls="--")

    colors = ["#E8F8F5", "#FEF9E7", "#FDEDEC", "#EAF2F8", "#F4ECF7", "#EAFAF1"]
    cp_all = [0] + list(series.change_points) + [len(t)]

    for idx in range(len(cp_all) - 1):
        start = cp_all[idx]
        end = cp_all[idx + 1]
        c = colors[idx % len(colors)]
        lbl = series.regime_labels[idx] if idx < len(series.regime_labels) else f"Regime {idx}"
        ax.axvspan(start, end, color=c, alpha=0.5, label=f"[{start}:{end}] {lbl}")
        if idx > 0:
            ax.axvline(start, color="#E74C3C", ls=":", lw=1.5)

    ax.set_title("Non-Stationary Time Series with Regime Shifts", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Time Step $t$", fontsize=11)
    ax.set_ylabel("Signal Value", fontsize=11)
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------- #
# 2. Ground Truth vs Predictions Comparison
# --------------------------------------------------------------------------- #
def plot_predictions_comparison(
    results: Sequence[OnlineResult],
    scaler: SeriesScaler,
    change_points: Sequence[int] = (),
    save_path: Path | None = None,
) -> plt.Figure:
    """Compare predictions of multiple models against true values in physical units."""
    _apply_theme()
    fig, ax = plt.subplots(figsize=(12, 5), dpi=200)

    # Use first result for true targets
    primary = results[0]
    times = [r.time_index for r in primary.records]
    y_true_orig = scaler.inverse_transform([r.y_true for r in primary.records])

    ax.plot(times, y_true_orig, label="Ground Truth", color="#111111", lw=1.8, zorder=5)

    palette = ["#E74C3C", "#2980B9", "#27AE60", "#8E44AD", "#F39C12", "#16A085"]
    for i, res in enumerate(results):
        c = palette[i % len(palette)]
        y_pred = scaler.inverse_transform([r.y_pred for r in res.records])
        is_adaptive = "adaptive" in res.name.lower()
        ax.plot(
            times,
            y_pred,
            label=res.name,
            color=c,
            lw=2.0 if is_adaptive else 1.2,
            ls="-" if is_adaptive else "--",
            alpha=0.9 if is_adaptive else 0.7,
        )

    for cp in change_points:
        if min(times) <= cp <= max(times):
            ax.axvline(cp, color="#7F8C8D", ls=":", lw=1.2, alpha=0.8)

    ax.set_title("Streaming Forecast Comparison Under Regime Transitions", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Time Step $t$", fontsize=11)
    ax.set_ylabel("Physical Signal Value", fontsize=11)
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------- #
# 3. Offline Pre-Training Loss Curve
# --------------------------------------------------------------------------- #
def plot_loss_curves(
    offline_res: OfflineResult,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot offline pre-training loss history."""
    _apply_theme()
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)

    iters = [h["iteration"] for h in offline_res.history]
    tr_loss = [h["train_loss"] for h in offline_res.history]
    val_loss = [h["val_loss"] for h in offline_res.history if "val_loss" in h and not np.isnan(h["val_loss"])]
    val_iters = [h["iteration"] for h in offline_res.history if "val_loss" in h and not np.isnan(h["val_loss"])]

    ax.plot(iters, tr_loss, label="Training Loss (MSE)", color="#2980B9", lw=1.8)
    if val_loss:
        ax.plot(val_iters, val_loss, label="Validation Loss", color="#E67E22", lw=1.6, ls="--")

    ax.axvline(offline_res.best_iteration, color="#27AE60", ls=":", label=f"Best Model (iter {offline_res.best_iteration})")
    ax.set_title("PQC Offline Pre-Training Convergence", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Optimization Iteration", fontsize=11)
    ax.set_ylabel("Loss (MSE in normalised units)", fontsize=11)
    ax.set_yscale("log")
    ax.grid(True, which="both")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------- #
# 4. Adaptive Window History & Error Thresholds
# --------------------------------------------------------------------------- #
def plot_adaptive_window_dynamics(
    result: OnlineResult,
    tau_low: float,
    tau_high: float,
    w_min: int,
    w_max: int,
    change_points: Sequence[int] = (),
    save_path: Path | None = None,
) -> plt.Figure:
    """Two-panel plot: (top) prediction error & thresholds; (bottom) adaptive window dynamics."""
    _apply_theme()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, dpi=200)

    times = [r.time_index for r in result.records]
    errs = [r.abs_error for r in result.records]
    windows = [r.window for r in result.records]

    # Panel 1: Error & Thresholds
    ax1.plot(times, errs, color="#E74C3C", lw=1.2, label="|Error| $\\epsilon_t$")
    ax1.axhline(tau_high, color="#C0392B", ls="--", lw=1.3, label=f"$\\tau_{{high}} = {tau_high}$ (Shrink Threshold)")
    ax1.axhline(tau_low, color="#27AE60", ls="--", lw=1.3, label=f"$\\tau_{{low}} = {tau_low}$ (Expand Threshold)")
    for cp in change_points:
        if min(times) <= cp <= max(times):
            ax1.axvline(cp, color="#7F8C8D", ls=":", lw=1.2)

    ax1.set_title("Prediction Error vs Adaptation Thresholds", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Absolute Error", fontsize=10)
    ax1.grid(True)
    ax1.legend(loc="upper right", fontsize=8.5)

    # Panel 2: Adaptive Window
    ax2.step(times, windows, where="post", color="#8E44AD", lw=1.8, label="Active Window Size $W_t$")
    ax2.axhline(w_max, color="#95A5A6", ls=":", label=f"$W_{{max}} = {w_max}$")
    ax2.axhline(w_min, color="#95A5A6", ls=":", label=f"$W_{{min}} = {w_min}$")
    for cp in change_points:
        if min(times) <= cp <= max(times):
            ax2.axvline(cp, color="#7F8C8D", ls=":", lw=1.2)

    ax2.set_title("Adaptive Quantum Observation Window Size", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Time Step $t$", fontsize=11)
    ax2.set_ylabel("Window Size $W$", fontsize=10)
    ax2.grid(True)
    ax2.legend(loc="upper right", fontsize=8.5)

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------- #
# 5. Baseline Benchmark Comparison Bar Charts
# --------------------------------------------------------------------------- #
def plot_benchmark_bars(
    aggregated: Sequence[AggregatedModelMetrics],
    save_path: Path | None = None,
) -> plt.Figure:
    """Bar chart comparing MSE, MAE, and Adaptation Delay across models."""
    _apply_theme()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=200)

    names = [a.name for a in aggregated]
    mses = [a.mse_mean for a in aggregated]
    mse_stds = [a.mse_std for a in aggregated]

    colors = ["#2ECC71" if "adaptive" in n.lower() else "#3498DB" for n in names]

    # Panel 1: MSE
    y_pos = np.arange(len(names))
    ax1.barh(y_pos, mses, xerr=mse_stds, color=colors, capsize=4, alpha=0.85, edgecolor="#2C3E50")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=10)
    ax1.invert_yaxis()
    ax1.set_xlabel("Mean Squared Error (Physical Units)", fontsize=10)
    ax1.set_title("Forecast Accuracy (Lower is Better)", fontsize=11, fontweight="bold")
    ax1.grid(True, axis="x")

    # Panel 2: Adaptation Delay
    delays = [a.adaptation_delay_mean if a.adaptation_delay_mean is not None else 0.0 for a in aggregated]
    delay_colors = ["#2ECC71" if "adaptive" in n.lower() else "#E67E22" for n in names]
    ax2.barh(y_pos, delays, color=delay_colors, alpha=0.85, edgecolor="#2C3E50")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])
    ax2.invert_yaxis()
    ax2.set_xlabel("Mean Recovery Delay (Time Steps)", fontsize=10)
    ax2.set_title("Regime Shift Adaptation Speed (Lower is Better)", fontsize=11, fontweight="bold")
    ax2.grid(True, axis="x")

    fig.suptitle("Model Benchmark Comparison", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------- #
# 6. Quantum Circuit Diagram Export
# --------------------------------------------------------------------------- #
def export_circuit_diagram(
    model: PQCModel,
    save_path: Path,
) -> None:
    """Export the parameterized quantum circuit diagram to text and PNG/SVG where possible."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path = save_path.with_suffix(".txt")
    txt_path.write_text(str(model.circuit.draw(output="text")), encoding="utf-8")

    try:
        fig = model.circuit.draw(output="mpl")
        fig.savefig(save_path, bbox_inches="tight", dpi=200)
        plt.close(fig)
    except Exception:
        # Fallback to text drawing if matplotlib latex/circuit drawer dependencies missing
        pass


# --------------------------------------------------------------------------- #
# 7. Quantum Entanglement Entropy Dynamics
# --------------------------------------------------------------------------- #
def plot_entanglement_dynamics(
    model: PQCModel,
    params: np.ndarray,
    angles_series: np.ndarray,
    times: np.ndarray,
    change_points: Sequence[int] = (),
    save_path: Path | None = None,
) -> plt.Figure | None:
    """Calculate and plot subsystem von Neumann entanglement entropy across test steps."""
    _apply_theme()
    theta, _, _ = model.split(params)

    # Subsystem cut: partition qubits into two halves
    n_qubits = model.n_qubits
    if n_qubits < 2:
        return None

    subsystem_b = list(range(n_qubits // 2, n_qubits))

    entropies = []
    # Sample every k steps if series is long
    step_skip = max(1, len(angles_series) // 100)
    sampled_indices = range(0, len(angles_series), step_skip)
    sampled_times = times[sampled_indices]

    for idx in sampled_indices:
        ang = angles_series[idx : idx + 1]
        try:
            # Generate statevector for this input
            sv = model.statevector(ang, theta)
            rho_a = partial_trace(sv, subsystem_b)
            # Von Neumann entropy: S(rho) = -Tr(rho log2 rho)
            eigs = np.linalg.eigvalsh(rho_a.data)
            eigs = eigs[eigs > 1e-12]
            s = float(-np.sum(eigs * np.log2(eigs)))
            entropies.append(s)
        except Exception:
            entropies.append(float("nan"))

    fig, ax = plt.subplots(figsize=(10, 4), dpi=200)
    ax.plot(sampled_times, entropies, color="#16A085", lw=1.8, label="Bipartite von Neumann Entropy $S(\\rho_A)$")

    for cp in change_points:
        if min(sampled_times) <= cp <= max(sampled_times):
            ax.axvline(cp, color="#7F8C8D", ls=":", lw=1.2)

    ax.set_title("Quantum Circuit Entanglement Entropy Dynamics Across Stream", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Time Step $t$", fontsize=11)
    ax.set_ylabel("Entropy (bits)", fontsize=11)
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig
