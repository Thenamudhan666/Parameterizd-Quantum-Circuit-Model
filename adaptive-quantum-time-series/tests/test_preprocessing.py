"""Unit tests for Module C: preprocessing, scaling, lag generation, and angle mapping."""

import numpy as np
import pytest

from src.preprocessing import (
    AngleMapper,
    FeatureReducer,
    SeriesScaler,
    clean_series,
    make_lagged_dataset,
)


def test_clean_series() -> None:
    raw = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    with pytest.raises(ValueError):
        clean_series(raw, missing="raise")

    cleaned = clean_series(raw, missing="interpolate")
    assert np.all(np.isfinite(cleaned))
    assert np.isclose(cleaned[2], 3.0)


def test_series_scaler_affine_map() -> None:
    train_vals = np.array([10.0, 20.0, 30.0])
    scaler = SeriesScaler.fit(train_vals)

    # 10 maps to -1, 30 maps to +1, 20 maps to 0
    z = scaler.transform(train_vals)
    np.testing.assert_allclose(z, [-1.0, 0.0, 1.0], atol=1e-7)

    # Exact invertibility
    recovered = scaler.inverse_transform(z)
    np.testing.assert_allclose(recovered, train_vals, atol=1e-7)


def test_lagged_dataset_temporal_causality() -> None:
    series = np.arange(10, dtype=float)
    n_lags = 3
    X, y, target_times = make_lagged_dataset(series, n_lags=n_lags)

    assert len(X) == 10 - 3  # 7 samples
    assert len(y) == 7
    assert len(target_times) == 7

    # Verify sample 0: lags are [0, 1, 2], target is 3, time is 3
    np.testing.assert_allclose(X[0], [0.0, 1.0, 2.0])
    assert y[0] == 3.0
    assert target_times[0] == 3

    # Check strictly increasing times
    assert np.all(np.diff(target_times) == 1)

    # Ensure no future target is inside lagged features
    for i, t in enumerate(target_times):
        assert np.all(X[i] < t)


def test_angle_mapper_bounds() -> None:
    mapper = AngleMapper(angle_min=0.0, angle_max=np.pi, headroom=1.5)
    z_vals = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    angles = mapper.transform(z_vals)

    assert np.all(angles >= 0.0)
    assert np.all(angles <= np.pi)
    # Monotonicity check
    assert np.all(np.diff(angles) >= 0.0)


def test_feature_reducer_identity_and_pca() -> None:
    X = np.random.RandomState(42).randn(50, 4)

    # 1. Identity when n_lags <= n_qubits
    red_id = FeatureReducer.fit(X, strategy="identity", n_qubits=4)
    out_id = red_id.transform(X)
    assert out_id.shape == (50, 4)
    np.testing.assert_allclose(out_id, X)

    # 2. PCA when n_lags > n_qubits
    red_pca = FeatureReducer.fit(X, strategy="pca", n_qubits=2)
    out_pca = red_pca.transform(X)
    assert out_pca.shape == (50, 2)
