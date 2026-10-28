"""Unit tests for Module I: Evaluation metrics, adaptation delay, and reporting."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.config import EvaluationConfig
from src.evaluation import (
    aggregate_across_seeds,
    compute_metrics,
    save_evaluation_report,
)
from src.preprocessing import SeriesScaler
from src.trainer import OnlineResult, StepRecord


def _make_dummy_result(name: str = "TestModel", n_steps: int = 20) -> OnlineResult:
    records = []
    for i in range(n_steps):
        records.append(
            StepRecord(
                step=i,
                sample_index=i,
                time_index=100 + i,
                y_true=float(np.sin(i * 0.2)),
                y_pred=float(np.sin(i * 0.2) + 0.05),
                abs_error=0.05,
                window=10,
                effective_window=10,
                window_start_time=100 + max(0, i - 10),
                window_end_time=100 + i,
                update_loss=0.0025,
                update_success=True,
                predict_time_s=0.001,
                update_time_s=0.005,
            )
        )
    return OnlineResult(
        name=name,
        records=tuple(records),
        window_events=(),
        n_updates=n_steps,
        n_failed_updates=0,
        n_optimizer_iterations=n_steps * 2,
        predict_time_s=0.02,
        update_time_s=0.1,
    )


def test_compute_metrics_accuracy_and_delays() -> None:
    res = _make_dummy_result("Dummy")
    scaler = SeriesScaler.fit(np.array([-2.0, 2.0]))
    eval_cfg = EvaluationConfig(adaptation_tolerance=0.1, shift_window=5)

    metrics = compute_metrics(
        result=res,
        scaler=scaler,
        eval_cfg=eval_cfg,
        change_points=(105,),
        regime_labels=("regime1", "regime2"),
        seed=42,
    )

    assert metrics.name == "Dummy"
    assert metrics.n_samples == 20
    assert metrics.mse > 0.0
    assert metrics.rmse > 0.0
    assert metrics.mae > 0.0
    assert len(metrics.regime_shifts) == 1
    assert metrics.regime_shifts[0].recovered


def test_aggregate_across_seeds_and_reporting() -> None:
    scaler = SeriesScaler.fit(np.array([-1.0, 1.0]))
    eval_cfg = EvaluationConfig()

    m1 = compute_metrics(_make_dummy_result("ModelA"), scaler, eval_cfg, seed=1)
    m2 = compute_metrics(_make_dummy_result("ModelA"), scaler, eval_cfg, seed=2)
    m3 = compute_metrics(_make_dummy_result("ModelB"), scaler, eval_cfg, seed=1)

    aggregated = aggregate_across_seeds([m1, m2, m3])
    assert len(aggregated) == 2

    # Save reports check
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        json_p, md_p = save_evaluation_report([m1, m2, m3], aggregated, out_dir)
        assert json_p.exists()
        assert md_p.exists()
        content = md_p.read_text(encoding="utf-8")
        assert "Executive Summary" in content
        assert "ModelA" in content
        assert "ModelB" in content
