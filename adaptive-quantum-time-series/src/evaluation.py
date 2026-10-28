"""Module I - Evaluation metrics, regime adaptation analysis, and reporting.

Metrics are calculated both in original physical units (via SeriesScaler inverse
transform) and normalised units.

Key measurements:
1. Standard accuracy: MSE, RMSE, MAE, MAPE (guarded against zero-crossings);
2. Regime-shift dynamics:
   - Adaptation delay: time steps after a change point until error drops below
     ``adaptation_tolerance``;
   - Pre- vs post-shift error over ``shift_window``;
3. Computational efficiency: runtime, optimizer iterations, circuit evaluations;
4. Multi-seed aggregation: mean +/- std error bars;
5. Serialization to JSON and Markdown summary tables.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.config import EvaluationConfig
from src.preprocessing import SeriesScaler
from src.trainer import OnlineResult


@dataclass(frozen=True)
class RegimeShiftMetric:
    """Metrics evaluated around one specific regime change point."""

    change_point_time: int
    regime_from: str
    regime_to: str
    pre_shift_mae: float
    post_shift_mae: float
    adaptation_delay_steps: int | None
    recovered: bool


@dataclass(frozen=True)
class ModelMetrics:
    """Consolidated performance profile for one model on one seed."""

    name: str
    seed: int
    n_samples: int
    mse: float
    rmse: float
    mae: float
    mape: float
    norm_mse: float
    norm_mae: float
    mean_window: float
    min_window: int
    max_window: int
    total_predict_time_s: float
    total_update_time_s: float
    n_updates: int
    n_failed_updates: int
    n_optimizer_iterations: int
    regime_shifts: tuple[RegimeShiftMetric, ...]
    mean_adaptation_delay: float | None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def compute_metrics(
    result: OnlineResult,
    scaler: SeriesScaler,
    eval_cfg: EvaluationConfig,
    change_points: tuple[int, ...] = (),
    regime_labels: tuple[str, ...] = (),
    seed: int = 42,
) -> ModelMetrics:
    """Compute comprehensive evaluation metrics for an OnlineResult.

    Converts predictions and targets back to physical signal units for honest
    real-world evaluation.
    """
    records = result.records
    if not records:
        raise ValueError(f"No records found in result '{result.name}'")

    y_true_norm = np.array([r.y_true for r in records], dtype=float)
    y_pred_norm = np.array([r.y_pred for r in records], dtype=float)
    times = np.array([r.time_index for r in records], dtype=int)
    windows = np.array([r.effective_window for r in records], dtype=int)

    # Scale back to original signal units
    y_true_orig = scaler.inverse_transform(y_true_norm)
    y_pred_orig = scaler.inverse_transform(y_pred_norm)

    # Standard metrics in original units
    diff_orig = y_true_orig - y_pred_orig
    mse_val = float(np.mean(diff_orig**2))
    rmse_val = float(np.sqrt(mse_val))
    mae_val = float(np.mean(np.abs(diff_orig)))

    # MAPE with threshold guard against near-zero division
    valid_mask = np.abs(y_true_orig) >= eval_cfg.mape_min_abs_target
    if np.any(valid_mask):
        mape_val = float(np.mean(np.abs(diff_orig[valid_mask] / y_true_orig[valid_mask])) * 100.0)
    else:
        mape_val = float("nan")

    # Metrics in normalised units
    diff_norm = y_true_norm - y_pred_norm
    norm_mse = float(np.mean(diff_norm**2))
    norm_mae = float(np.mean(np.abs(diff_norm)))

    # Window metrics
    mean_w = float(np.mean(windows)) if windows.size > 0 else 0.0
    min_w = int(np.min(windows)) if windows.size > 0 else 0
    max_w = int(np.max(windows)) if windows.size > 0 else 0

    # Regime change metrics
    shift_metrics: list[RegimeShiftMetric] = []
    delays: list[int] = []

    err_orig = np.abs(diff_orig)
    tol = eval_cfg.adaptation_tolerance
    shift_win = eval_cfg.shift_window

    for cp_idx, cp in enumerate(change_points):
        # find matching index in streaming times
        match_indices = np.where(times >= cp)[0]
        if match_indices.size == 0:
            continue
        cp_stream_idx = match_indices[0]

        # pre-shift error
        pre_start = max(0, cp_stream_idx - shift_win)
        pre_errs = err_orig[pre_start:cp_stream_idx]
        pre_mae = float(np.mean(pre_errs)) if pre_errs.size > 0 else float("nan")

        # post-shift error
        post_end = min(len(records), cp_stream_idx + shift_win)
        post_errs = err_orig[cp_stream_idx:post_end]
        post_mae = float(np.mean(post_errs)) if post_errs.size > 0 else float("nan")

        # adaptation delay: steps after cp until err <= tolerance
        delay: int | None = None
        recovered = False
        subsequent = err_orig[cp_stream_idx:]
        for step_offset, e in enumerate(subsequent):
            if e <= tol:
                delay = step_offset
                recovered = True
                delays.append(step_offset)
                break

        lbl_from = regime_labels[cp_idx] if cp_idx < len(regime_labels) else f"regime_{cp_idx}"
        lbl_to = regime_labels[cp_idx + 1] if cp_idx + 1 < len(regime_labels) else f"regime_{cp_idx + 1}"

        shift_metrics.append(
            RegimeShiftMetric(
                change_point_time=int(cp),
                regime_from=lbl_from,
                regime_to=lbl_to,
                pre_shift_mae=pre_mae,
                post_shift_mae=post_mae,
                adaptation_delay_steps=delay,
                recovered=recovered,
            )
        )

    mean_delay = float(np.mean(delays)) if delays else None

    return ModelMetrics(
        name=result.name,
        seed=seed,
        n_samples=len(records),
        mse=mse_val,
        rmse=rmse_val,
        mae=mae_val,
        mape=mape_val,
        norm_mse=norm_mse,
        norm_mae=norm_mae,
        mean_window=mean_w,
        min_window=min_w,
        max_window=max_w,
        total_predict_time_s=result.predict_time_s,
        total_update_time_s=result.update_time_s,
        n_updates=result.n_updates,
        n_failed_updates=result.n_failed_updates,
        n_optimizer_iterations=result.n_optimizer_iterations,
        regime_shifts=tuple(shift_metrics),
        mean_adaptation_delay=mean_delay,
    )


# --------------------------------------------------------------------------- #
# Multi-seed Aggregation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AggregatedModelMetrics:
    """Mean and standard deviation across multiple random seeds."""

    name: str
    n_seeds: int
    mse_mean: float
    mse_std: float
    rmse_mean: float
    rmse_std: float
    mae_mean: float
    mae_std: float
    mape_mean: float
    mape_std: float
    adaptation_delay_mean: float | None
    total_time_s_mean: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def aggregate_across_seeds(metrics_list: list[ModelMetrics]) -> list[AggregatedModelMetrics]:
    """Group metrics by model name and compute mean and standard deviation."""
    by_model: dict[str, list[ModelMetrics]] = {}
    for m in metrics_list:
        by_model.setdefault(m.name, []).append(m)

    aggregated: list[AggregatedModelMetrics] = []
    for name, group in by_model.items():
        n = len(group)
        mses = [m.mse for m in group]
        rmses = [m.rmse for m in group]
        maes = [m.mae for m in group]
        mapes = [m.mape for m in group if np.isfinite(m.mape)]
        times = [m.total_predict_time_s + m.total_update_time_s for m in group]
        delays = [m.mean_adaptation_delay for m in group if m.mean_adaptation_delay is not None]

        aggregated.append(
            AggregatedModelMetrics(
                name=name,
                n_seeds=n,
                mse_mean=float(np.mean(mses)),
                mse_std=float(np.std(mses, ddof=1)) if n > 1 else 0.0,
                rmse_mean=float(np.mean(rmses)),
                rmse_std=float(np.std(rmses, ddof=1)) if n > 1 else 0.0,
                mae_mean=float(np.mean(maes)),
                mae_std=float(np.std(maes, ddof=1)) if n > 1 else 0.0,
                mape_mean=float(np.mean(mapes)) if mapes else float("nan"),
                mape_std=float(np.std(mapes, ddof=1)) if len(mapes) > 1 else 0.0,
                adaptation_delay_mean=float(np.mean(delays)) if delays else None,
                total_time_s_mean=float(np.mean(times)),
            )
        )

    # Sort so adaptive PQC comes first, followed by others
    def sort_key(agg: AggregatedModelMetrics) -> tuple[int, float]:
        if "adaptive" in agg.name.lower():
            return (0, agg.mse_mean)
        return (1, agg.mse_mean)

    aggregated.sort(key=sort_key)
    return aggregated


# --------------------------------------------------------------------------- #
# Serialization and Reporting
# --------------------------------------------------------------------------- #
def save_evaluation_report(
    metrics_list: list[ModelMetrics],
    aggregated: list[AggregatedModelMetrics],
    output_dir: Path,
    experiment_name: str = "adaptive_pqc",
) -> tuple[Path, Path]:
    """Generate both a JSON summary and a readable Markdown evaluation report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    md_path = output_dir / "report.md"

    # 1. JSON Summary
    payload = {
        "experiment": experiment_name,
        "models": [m.to_dict() for m in metrics_list],
        "aggregated": [a.to_dict() for a in aggregated],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # 2. Markdown Report
    lines = [
        f"# Evaluation Report: {experiment_name}",
        "",
        "## 1. Executive Summary & Benchmark Comparison",
        "",
        "| Model | MSE | RMSE | MAE | MAPE (%) | Mean Adaptation Delay (steps) | Total Time (s) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for agg in aggregated:
        delay_str = f"{agg.adaptation_delay_mean:.1f}" if agg.adaptation_delay_mean is not None else "N/A"
        mape_str = f"{agg.mape_mean:.2f}%" if np.isfinite(agg.mape_mean) else "N/A"
        lines.append(
            f"| **{agg.name}** | {agg.mse_mean:.4f} ± {agg.mse_std:.4f} | "
            f"{agg.rmse_mean:.4f} ± {agg.rmse_std:.4f} | "
            f"{agg.mae_mean:.4f} ± {agg.mae_std:.4f} | "
            f"{mape_str} | {delay_str} | {agg.total_time_s_mean:.2f}s |"
        )

    lines.extend([
        "",
        "## 2. Regime Shift Analysis",
        "",
        "This table evaluates how quickly models adapt following known non-stationary regime transitions (frequency jumps, amplitude spikes, variance shifts):",
        "",
    ])

    # Show first seed regime shift table
    first_seed = metrics_list[0].seed if metrics_list else 42
    seed_metrics = [m for m in metrics_list if m.seed == first_seed]

    lines.append(f"### Seed {first_seed} Regime Shift Details")
    lines.append("")
    lines.append("| Model | Transition | Pre-Shift MAE | Post-Shift MAE | Delay Steps | Recovered |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")

    for m in seed_metrics:
        for rs in m.regime_shifts:
            delay_text = str(rs.adaptation_delay_steps) if rs.adaptation_delay_steps is not None else "Timeout"
            lines.append(
                f"| {m.name} | {rs.regime_from} -> {rs.regime_to} | {rs.pre_shift_mae:.4f} | {rs.post_shift_mae:.4f} | {delay_text} | {rs.recovered} |"
            )

    lines.extend([
        "",
        "## 3. Methodological Discussion & Limitations",
        "",
        "- **Simulation Scope**: Evaluated on statevector simulation (Qiskit Aer / Statevector). Does not claim quantum computational supremacy over classical hardware.",
        "- **Parameter-Shift Rule**: Exact analytical quantum gradients were estimated across the parameterized circuit ansatz.",
        "- **Dynamic Windowing**: Shortening the window size after a regime boundary prevents stale observations from corrupting the gradient updates, while expanding the window in stationary phases reduces variance.",
        "",
    ])

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return json_path, md_path
