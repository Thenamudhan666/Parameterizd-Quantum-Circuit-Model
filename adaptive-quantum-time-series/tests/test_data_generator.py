"""Unit tests for Module B: synthetic non-stationary time series generator."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.config import DataConfig, RegimeConfig
from src.data_generator import SyntheticSeries, generate_synthetic_series


def test_generator_output_shape_and_types() -> None:
    cfg = DataConfig(
        length=200,
        regimes=(
            RegimeConfig(start=0, amplitude=1.0, period=20.0, noise_std=0.05, label="stable"),
            RegimeConfig(start=100, amplitude=1.5, period=10.0, noise_std=0.1, label="shift"),
        ),
    )
    series = generate_synthetic_series(cfg, seed=42)

    assert isinstance(series, SyntheticSeries)
    assert len(series) == 200
    assert series.time.shape == (200,)
    assert series.values.shape == (200,)
    assert series.clean.shape == (200,)
    assert series.change_points == (100,)
    assert series.regime_labels == ("stable", "shift")


def test_generator_reproducibility() -> None:
    cfg = DataConfig(length=150)
    s1 = generate_synthetic_series(cfg, seed=123)
    s2 = generate_synthetic_series(cfg, seed=123)
    s3 = generate_synthetic_series(cfg, seed=999)

    np.testing.assert_allclose(s1.values, s2.values)
    assert not np.allclose(s1.values, s3.values)


def test_chronological_splits_no_leakage() -> None:
    cfg = DataConfig(length=200, train_fraction=0.6, val_fraction=0.2)
    series = generate_synthetic_series(cfg, seed=42)
    splits = series.splits(cfg.train_fraction, cfg.val_fraction)

    assert splits.train_end == 120
    assert splits.val_end == 160
    assert splits.length == 200

    assert splits.split_of(0) == "train"
    assert splits.split_of(119) == "train"
    assert splits.split_of(120) == "val"
    assert splits.split_of(159) == "val"
    assert splits.split_of(160) == "test"
    assert splits.split_of(199) == "test"

    with pytest.raises(IndexError):
        splits.split_of(200)


def test_series_serialization() -> None:
    cfg = DataConfig(length=100)
    series = generate_synthetic_series(cfg, seed=42)

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "test.csv"
        series.save_csv(csv_path)
        assert csv_path.exists()
        assert csv_path.stat().st_size > 0
