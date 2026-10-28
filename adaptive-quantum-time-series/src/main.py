"""Main orchestrator and CLI for Adaptive Quantum Time-Series Learning.

Usage:
    # Run rapid smoke test
    python -m src.main --mode smoke

    # Run single-seed adaptive PQC training and visualization
    python -m src.main --mode train --config config/config.yaml

    # Run multi-seed benchmark across all models (Adaptive PQC, Fixed-W, MLP, LSTM)
    python -m src.main --mode benchmark --config config/config.yaml

    # Generate and plot synthetic non-stationary dataset only
    python -m src.main --mode generate --config config/config.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.adaptive_window import AdaptiveWindowController, FixedWindowPolicy
from src.classical_baselines import (
    TORCH_AVAILABLE,
    train_offline_lstm,
    train_offline_mlp,
)
from src.config import Config, load_config
from src.data_generator import SyntheticSeries, generate_synthetic_series
from src.evaluation import (
    AggregatedModelMetrics,
    ModelMetrics,
    aggregate_across_seeds,
    compute_metrics,
    save_evaluation_report,
)
from src.preprocessing import (
    AngleMapper,
    FeatureReducer,
    SeriesScaler,
    clean_series,
    make_lagged_dataset,
)
from src.quantum_circuit import PQCModel
from src.trainer import (
    OnlineResult,
    PQCOnlineLearner,
    run_online,
    train_offline_pqc,
)
from src.visualization import (
    export_circuit_diagram,
    plot_adaptive_window_dynamics,
    plot_benchmark_bars,
    plot_entanglement_dynamics,
    plot_loss_curves,
    plot_predictions_comparison,
    plot_time_series_regimes,
)


def run_pipeline_for_seed(
    cfg: Config,
    seed: int,
    output_dir: Path,
    include_baselines: bool = True,
) -> tuple[list[OnlineResult], list[ModelMetrics], SeriesScaler, SyntheticSeries, PQCModel, np.ndarray]:
    """Run data generation, feature prep, pre-training and online streaming for one seed."""
    print(f"\n=======================================================")
    print(f" Starting Pipeline Run for Seed: {seed}")
    print(f"=======================================================")

    # 1. Generate Synthetic Data
    series = generate_synthetic_series(cfg.data, seed=seed)
    cleaned = clean_series(series.values, missing=cfg.data.missing)

    # 2. Chronological Splits & Scaling
    splits = series.splits(cfg.data.train_fraction, cfg.data.val_fraction)
    train_raw = cleaned[: splits.train_end]

    scaler = SeriesScaler.fit(train_raw)
    z_all = scaler.transform(cleaned)

    # 3. Supervised Lag Dataset
    X_lags, y_lags, target_times = make_lagged_dataset(z_all, cfg.features.n_lags)

    # Sample-level split masks
    train_mask = target_times < splits.train_end
    val_mask = (target_times >= splits.train_end) & (target_times < splits.val_end)
    test_mask = target_times >= splits.val_end

    X_train, y_train = X_lags[train_mask], y_lags[train_mask]
    X_val, y_val = X_lags[val_mask], y_lags[val_mask]
    test_indices = np.where(test_mask)[0]

    print(f"Dataset: Total Samples = {len(cleaned)}, Lag Pairs = {len(X_lags)}")
    print(f"Splits: Train = {len(X_train)}, Val = {len(X_val)}, Test = {len(test_indices)}")

    # 4. Feature Reduction and Angle Mapping
    reducer = FeatureReducer.fit(X_train, cfg.features.reduction, cfg.quantum.n_qubits)
    X_tr_red = reducer.transform(X_train)
    X_val_red = reducer.transform(X_val)
    X_all_red = reducer.transform(X_lags)

    mapper = AngleMapper(
        angle_min=cfg.features.angle_min,
        angle_max=cfg.features.angle_max,
        headroom=cfg.features.headroom,
    )
    angles_tr = mapper.transform(X_tr_red)
    angles_val = mapper.transform(X_val_red)
    angles_all = mapper.transform(X_all_red)

    # 5. Quantum Circuit Model
    pqc = PQCModel(cfg.quantum)
    print(f"Quantum Model: {pqc.n_qubits} qubits, {cfg.quantum.ansatz_layers} ansatz layers, {pqc.n_params} parameters")

    # 6. Offline Pre-Training of PQC
    print("\n--- [1/3] Offline PQC Pre-Training ---")
    t0 = time.perf_counter()
    offline_res = train_offline_pqc(
        pqc,
        cfg.training,
        angles_tr,
        y_train,
        angles_val,
        y_val,
        seed=seed,
    )
    print(f"PQC Pre-training complete in {time.perf_counter() - t0:.2f}s | Best Val Loss = {offline_res.best_val_loss:.6f}")

    # Save offline loss curve
    plots_dir = output_dir / "plots"
    plot_loss_curves(offline_res, save_path=plots_dir / f"pqc_loss_curve_seed{seed}.png")

    # 7. Online Adaptation across Models
    print("\n--- [2/3] Online Streaming Adaptation ---")
    online_results: list[OnlineResult] = []

    # Model A: Adaptive-Window PQC
    print("Streaming Model: Adaptive-Window PQC...")
    adaptive_learner = PQCOnlineLearner(
        model=pqc,
        params=offline_res.params,
        opt_cfg=cfg.online.optimizer,
        steps_per_update=cfg.online.steps_per_update,
        seed=seed,
    )
    adaptive_policy = AdaptiveWindowController(cfg.adaptive_window)
    res_adaptive = run_online(
        name="Adaptive-Window PQC",
        learner=adaptive_learner,
        policy=adaptive_policy,
        features=angles_all,
        targets=y_lags,
        target_time=target_times,
        stream_idx=test_indices,
        update_every=cfg.online.update_every,
    )
    online_results.append(res_adaptive)

    # Model B: Fixed-Window PQCs
    if include_baselines:
        for w in cfg.baselines.fixed_windows:
            print(f"Streaming Model: Fixed-Window PQC (W={w})...")
            learner_fixed = PQCOnlineLearner(
                model=pqc,
                params=offline_res.params,
                opt_cfg=cfg.online.optimizer,
                steps_per_update=cfg.online.steps_per_update,
                seed=seed,
            )
            policy_fixed = FixedWindowPolicy(w)
            res_fixed = run_online(
                name=f"Fixed-Window PQC (W={w})",
                learner=learner_fixed,
                policy=policy_fixed,
                features=angles_all,
                targets=y_lags,
                target_time=target_times,
                stream_idx=test_indices,
                update_every=cfg.online.update_every,
            )
            online_results.append(res_fixed)

        # Model C: Offline-Only Frozen PQC (Zero online adaptation)
        if cfg.baselines.include_offline_only_pqc:
            print("Streaming Model: Frozen PQC (No online update)...")
            learner_frozen = PQCOnlineLearner(
                model=pqc,
                params=offline_res.params,
                opt_cfg=cfg.online.optimizer,
                steps_per_update=0,
                seed=seed,
            )
            res_frozen = run_online(
                name="Frozen PQC (No Adaptation)",
                learner=learner_frozen,
                policy=None,
                features=angles_all,
                targets=y_lags,
                target_time=target_times,
                stream_idx=test_indices,
            )
            online_results.append(res_frozen)

        # Model D: Classical MLP Baseline
        if cfg.baselines.mlp.enabled:
            print("Streaming Model: Classical MLP Regressor...")
            mlp_learner = train_offline_mlp(cfg.baselines.mlp, X_train, y_train, seed=seed)
            mlp_policy = FixedWindowPolicy(cfg.baselines.mlp.online_window)
            res_mlp = run_online(
                name="Classical MLP",
                learner=mlp_learner,
                policy=mlp_policy,
                features=X_lags,
                targets=y_lags,
                target_time=target_times,
                stream_idx=test_indices,
                update_every=cfg.online.update_every,
            )
            online_results.append(res_mlp)

        # Model E: Classical LSTM Baseline
        if cfg.baselines.lstm.enabled and TORCH_AVAILABLE:
            print("Streaming Model: Classical PyTorch LSTM...")
            lstm_learner = train_offline_lstm(cfg.baselines.lstm, X_train, y_train, seed=seed)
            lstm_policy = FixedWindowPolicy(cfg.baselines.lstm.online_window)
            res_lstm = run_online(
                name="Classical LSTM",
                learner=lstm_learner,
                policy=lstm_policy,
                features=X_lags,
                targets=y_lags,
                target_time=target_times,
                stream_idx=test_indices,
                update_every=cfg.online.update_every,
            )
            online_results.append(res_lstm)

    # 8. Compute Metrics for all models on this seed
    print("\n--- [3/3] Evaluating Seed Results ---")
    metrics_list: list[ModelMetrics] = []
    for res in online_results:
        m = compute_metrics(
            result=res,
            scaler=scaler,
            eval_cfg=cfg.evaluation,
            change_points=series.change_points,
            regime_labels=series.regime_labels,
            seed=seed,
        )
        metrics_list.append(m)
        delay_str = f"{m.mean_adaptation_delay:.1f} steps" if m.mean_adaptation_delay is not None else "N/A"
        print(f"-> {m.name:30s} | MSE: {m.mse:.4f} | MAE: {m.mae:.4f} | Adaptation Delay: {delay_str}")

    return online_results, metrics_list, scaler, series, pqc, offline_res.params


def run_benchmark_mode(cfg: Config) -> None:
    """Execute complete benchmarking across models and random seeds."""
    out_dir = cfg.output_root
    plots_dir = out_dir / "plots"
    reports_dir = out_dir / "reports"

    all_metrics: list[ModelMetrics] = []
    primary_results: list[OnlineResult] = []
    primary_scaler: SeriesScaler | None = None
    primary_series: SyntheticSeries | None = None
    primary_pqc: PQCModel | None = None
    primary_params: np.ndarray | None = None

    seeds = cfg.benchmark.seeds
    print(f"Starting Multi-Seed Benchmark across {len(seeds)} seeds: {seeds}")

    for idx, s in enumerate(seeds):
        results, metrics, scaler, series, pqc, params = run_pipeline_for_seed(
            cfg=cfg,
            seed=s,
            output_dir=out_dir,
            include_baselines=True,
        )
        all_metrics.extend(metrics)
        if idx == 0:
            primary_results = results
            primary_scaler = scaler
            primary_series = series
            primary_pqc = pqc
            primary_params = params

    # Aggregated metrics across seeds
    aggregated = aggregate_across_seeds(all_metrics)

    # Save Reports
    json_path, md_path = save_evaluation_report(
        metrics_list=all_metrics,
        aggregated=aggregated,
        output_dir=reports_dir,
        experiment_name=cfg.experiment.name,
    )
    print(f"\nSaved Reports:\n- JSON: {json_path}\n- Markdown: {md_path}")

    # Generate benchmark visualizations from primary seed & aggregations
    assert primary_series is not None
    assert primary_scaler is not None
    assert primary_pqc is not None
    assert primary_params is not None

    print("\nGenerating publication figures...")
    plot_time_series_regimes(primary_series, save_path=plots_dir / "01_time_series_regimes.png")
    plot_predictions_comparison(
        primary_results,
        scaler=primary_scaler,
        change_points=primary_series.change_points,
        save_path=plots_dir / "02_predictions_comparison.png",
    )

    adaptive_res = next(r for r in primary_results if "adaptive" in r.name.lower())
    plot_adaptive_window_dynamics(
        result=adaptive_res,
        tau_low=cfg.adaptive_window.tau_low,
        tau_high=cfg.adaptive_window.tau_high,
        w_min=cfg.adaptive_window.w_min,
        w_max=cfg.adaptive_window.w_max,
        change_points=primary_series.change_points,
        save_path=plots_dir / "03_adaptive_window_dynamics.png",
    )

    plot_benchmark_bars(aggregated, save_path=plots_dir / "04_benchmark_model_comparison.png")
    export_circuit_diagram(primary_pqc, save_path=plots_dir / "05_quantum_circuit_diagram.png")

    # Quantum Entanglement Entropy Plot
    X_lags, _, times = make_lagged_dataset(primary_scaler.transform(primary_series.values), cfg.features.n_lags)
    reducer = FeatureReducer.fit(X_lags, cfg.features.reduction, cfg.quantum.n_qubits)
    mapper = AngleMapper(cfg.features.angle_min, cfg.features.angle_max, cfg.features.headroom)
    angles_all = mapper.transform(reducer.transform(X_lags))

    plot_entanglement_dynamics(
        model=primary_pqc,
        params=primary_params,
        angles_series=angles_all,
        times=times,
        change_points=primary_series.change_points,
        save_path=plots_dir / "06_quantum_entanglement_entropy.png",
    )

    print(f"All benchmark figures successfully generated in '{plots_dir}'.")


def run_train_mode(cfg: Config) -> None:
    """Run single-seed training and visualization for Adaptive PQC."""
    out_dir = cfg.output_root
    plots_dir = out_dir / "plots"
    reports_dir = out_dir / "reports"

    results, metrics, scaler, series, pqc, params = run_pipeline_for_seed(
        cfg=cfg,
        seed=cfg.experiment.seed,
        output_dir=out_dir,
        include_baselines=True,
    )
    aggregated = aggregate_across_seeds(metrics)

    json_path, md_path = save_evaluation_report(
        metrics_list=metrics,
        aggregated=aggregated,
        output_dir=reports_dir,
        experiment_name=cfg.experiment.name,
    )
    print(f"\nSaved Reports:\n- JSON: {json_path}\n- Markdown: {md_path}")

    plot_time_series_regimes(series, save_path=plots_dir / "time_series_regimes.png")
    plot_predictions_comparison(
        results,
        scaler=scaler,
        change_points=series.change_points,
        save_path=plots_dir / "predictions_comparison.png",
    )
    adaptive_res = next(r for r in results if "adaptive" in r.name.lower())
    plot_adaptive_window_dynamics(
        result=adaptive_res,
        tau_low=cfg.adaptive_window.tau_low,
        tau_high=cfg.adaptive_window.tau_high,
        w_min=cfg.adaptive_window.w_min,
        w_max=cfg.adaptive_window.w_max,
        change_points=series.change_points,
        save_path=plots_dir / "adaptive_window_dynamics.png",
    )
    export_circuit_diagram(pqc, save_path=plots_dir / "quantum_circuit_diagram.png")
    print(f"Training completed. Outputs saved to '{out_dir}'.")


def run_generate_mode(cfg: Config) -> None:
    """Generate synthetic data and plot time series."""
    series = generate_synthetic_series(cfg.data, seed=cfg.experiment.seed)
    data_dir = cfg.data_root / "generated"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "synthetic_series.csv"
    series.save_csv(csv_path)
    print(f"Saved synthetic series to {csv_path}")

    plots_dir = cfg.output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig_path = plots_dir / "time_series_regimes.png"
    plot_time_series_regimes(series, save_path=fig_path)
    print(f"Saved time series plot to {fig_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive Quantum Time-Series Learning using PQC")
    parser.add_argument(
        "--mode",
        choices=["smoke", "train", "benchmark", "generate"],
        default="smoke",
        help="Execution mode (default: smoke)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    # Load appropriate config
    if args.mode == "smoke":
        cfg_path = Path(args.config) if args.config else Path(__file__).resolve().parents[1] / "config" / "smoke.yaml"
    else:
        cfg_path = Path(args.config) if args.config else Path(__file__).resolve().parents[1] / "config" / "config.yaml"

    print(f"Loading configuration from: {cfg_path}")
    cfg = load_config(cfg_path)

    if args.mode in {"smoke", "benchmark"}:
        run_benchmark_mode(cfg)
    elif args.mode == "train":
        run_train_mode(cfg)
    elif args.mode == "generate":
        run_generate_mode(cfg)


if __name__ == "__main__":
    main()
