"""Module C (classical part) - cleaning, scaling and lag-feature construction.

Pipeline (all statistics are fitted on the *training* split only):

1. ``clean_series``      - reject or interpolate missing/non-finite samples.
2. ``SeriesScaler``      - affine map of the training range onto [-1, 1]
                           ("normalised units", z).  Test values outside the
                           training range are *not* clipped here, so the
                           original scale can always be recovered exactly.
3. ``make_lagged_dataset`` - supervised pairs  x_t = (z_{t-L}, ..., z_{t-1}),
                           y_t = z_t.  The feature vector of target t only
                           contains strictly earlier samples (no leakage).
4. ``FeatureReducer``    - maps L lags onto the quantum feature dimension
                           (identity, PCA fitted on training features, or
                           pass-through for data re-uploading).
5. ``AngleMapper``       - maps z onto rotation angles in [angle_min, angle_max]
                           using ``headroom`` so that values up to
                           ``headroom`` x the training range stay distinct;
                           larger values are clipped (and counted).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import FeatureConfig


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def clean_series(values: np.ndarray, missing: str = "raise", min_length: int = 3) -> np.ndarray:
    """Validate a 1-D series and optionally fill missing values.

    Args:
        values: Raw observations.
        missing: ``"raise"`` to reject NaN/inf, ``"interpolate"`` to fill them
            by linear interpolation in time (edges use the nearest valid value).
        min_length: Minimum number of observations required.

    Returns:
        A new float array (the input is never modified).

    Raises:
        ValueError: On wrong shape, too few observations, unknown ``missing``
            strategy, or missing values when ``missing="raise"``.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"series must be one-dimensional, got shape {arr.shape}")
    if arr.size < min_length:
        raise ValueError(f"series has {arr.size} observations; at least {min_length} required")
    if missing not in {"raise", "interpolate"}:
        raise ValueError("missing must be 'raise' or 'interpolate'")
    bad = ~np.isfinite(arr)
    if not bad.any():
        return arr.copy()
    if missing == "raise":
        idx = np.flatnonzero(bad)[:5].tolist()
        raise ValueError(f"series contains {int(bad.sum())} missing/non-finite values (e.g. at {idx})")
    good = np.flatnonzero(~bad)
    if good.size < 2:
        raise ValueError("cannot interpolate: fewer than two valid observations")
    out = arr.copy()
    out[bad] = np.interp(np.flatnonzero(bad), good, arr[good])
    return out


# --------------------------------------------------------------------------- #
# Scaling
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SeriesScaler:
    """Affine scaler mapping the training range [lo, hi] onto [-1, 1]."""

    lo: float
    hi: float

    @classmethod
    def fit(cls, train_values: np.ndarray) -> "SeriesScaler":
        arr = np.asarray(train_values, dtype=float)
        if arr.size < 2 or not np.all(np.isfinite(arr)):
            raise ValueError("scaler needs >= 2 finite training values")
        lo, hi = float(arr.min()), float(arr.max())
        if hi - lo < 1e-12:
            raise ValueError("training values are constant; cannot scale")
        return cls(lo=lo, hi=hi)

    @property
    def half_range(self) -> float:
        return 0.5 * (self.hi - self.lo)

    @property
    def center(self) -> float:
        return 0.5 * (self.hi + self.lo)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.center) / self.half_range

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, dtype=float) * self.half_range + self.center

    def scale_error(self, z_error: np.ndarray) -> np.ndarray:
        """Convert an error magnitude from normalised to original units."""
        return np.asarray(z_error, dtype=float) * self.half_range


# --------------------------------------------------------------------------- #
# Lag features
# --------------------------------------------------------------------------- #
def make_lagged_dataset(z: np.ndarray, n_lags: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build one-step-ahead supervised pairs from a scaled series.

    Returns:
        X: (N, n_lags) with X[i] = z[t_i - n_lags : t_i] (oldest lag first),
        y: (N,) with y[i] = z[t_i],
        target_time: (N,) the time index t_i of each target.
    """
    arr = np.asarray(z, dtype=float)
    if n_lags < 1:
        raise ValueError("n_lags must be >= 1")
    if arr.size <= n_lags:
        raise ValueError(f"series of length {arr.size} too short for {n_lags} lags")
    target_time = np.arange(n_lags, arr.size)
    X = np.stack([arr[t - n_lags : t] for t in target_time])
    y = arr[target_time].copy()
    return X, y, target_time


# --------------------------------------------------------------------------- #
# Feature reduction / mapping onto qubits
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FeatureReducer:
    """Maps L lag features onto the quantum feature dimension.

    Strategies:
        * ``identity``  - L <= n_qubits; the circuit repeats features
                          cyclically over qubits when L < n_qubits.
        * ``pca``       - L > n_qubits; project centred lags onto the top
                          n_qubits principal components of the *training*
                          features, then divide by the training max-abs score
                          so components live on the same [-1, 1] scale.
        * ``reupload``  - L > n_qubits; pass all lags through and let the
                          circuit encode them in successive re-uploading blocks.
    """

    strategy: str
    n_out: int
    mean: np.ndarray | None = None
    components: np.ndarray | None = None
    scale: np.ndarray | None = None

    @classmethod
    def fit(cls, X_train: np.ndarray, n_qubits: int, reduction: str) -> "FeatureReducer":
        X = np.asarray(X_train, dtype=float)
        n_lags = X.shape[1]
        if n_lags <= n_qubits:
            return cls(strategy="identity", n_out=n_lags)
        if reduction == "reupload":
            return cls(strategy="reupload", n_out=n_lags)
        if reduction != "pca":
            raise ValueError(f"unknown reduction strategy '{reduction}'")
        if X.shape[0] <= n_qubits:
            raise ValueError("PCA reduction needs more training samples than qubits")
        mean = X.mean(axis=0)
        _, _, vt = np.linalg.svd(X - mean, full_matrices=False)
        components = vt[:n_qubits]
        scores = (X - mean) @ components.T
        scale = np.maximum(np.abs(scores).max(axis=0), 1e-12)
        return cls(strategy="pca", n_out=n_qubits, mean=mean, components=components, scale=scale)

    def transform(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        if self.strategy != "pca":
            return arr.copy()
        assert self.mean is not None and self.components is not None and self.scale is not None
        return ((arr - self.mean) @ self.components.T) / self.scale


@dataclass(frozen=True)
class AngleMapper:
    """Maps normalised features onto rotation angles with clipping headroom."""

    angle_min: float
    angle_max: float
    headroom: float

    @classmethod
    def from_config(cls, cfg: FeatureConfig) -> "AngleMapper":
        return cls(angle_min=cfg.angle_min, angle_max=cfg.angle_max, headroom=cfg.headroom)

    def transform(self, features: np.ndarray) -> np.ndarray:
        u = np.clip(np.asarray(features, dtype=float) / self.headroom, -1.0, 1.0)
        return self.angle_min + (u + 1.0) * 0.5 * (self.angle_max - self.angle_min)

    def clip_fraction(self, features: np.ndarray) -> float:
        arr = np.asarray(features, dtype=float)
        return float(np.mean(np.abs(arr) > self.headroom)) if arr.size else 0.0


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SupervisedData:
    """Leakage-free supervised dataset shared by every model."""

    X_lags: np.ndarray       # (N, L) normalised lags, used by classical baselines
    X_angles: np.ndarray     # (N, d) rotation angles, used by the PQC
    y: np.ndarray            # (N,) normalised targets
    target_time: np.ndarray  # (N,) time index of each target
    train_idx: np.ndarray    # sample indices whose target lies in the train split
    val_idx: np.ndarray
    test_idx: np.ndarray
    angle_clip_fraction: float

    def __len__(self) -> int:
        return int(self.y.shape[0])


@dataclass(frozen=True)
class Preprocessor:
    """Fitted preprocessing state (scaler, reducer, angle mapper)."""

    scaler: SeriesScaler
    reducer: FeatureReducer
    angle_mapper: AngleMapper
    n_lags: int

    @classmethod
    def fit(cls, values: np.ndarray, train_end: int, cfg: FeatureConfig, n_qubits: int) -> "Preprocessor":
        """Fit all statistics on observations strictly before ``train_end``."""
        arr = clean_series(values)
        if not cfg.n_lags < train_end <= arr.size:
            raise ValueError("train_end must exceed n_lags and not exceed the series length")
        scaler = SeriesScaler.fit(arr[:train_end])
        z_train = scaler.transform(arr[:train_end])
        X_train, _, _ = make_lagged_dataset(z_train, cfg.n_lags)
        reducer = FeatureReducer.fit(X_train, n_qubits, cfg.reduction)
        return cls(scaler=scaler, reducer=reducer, angle_mapper=AngleMapper.from_config(cfg), n_lags=cfg.n_lags)

    @property
    def n_quantum_features(self) -> int:
        return self.reducer.n_out

    def build(self, values: np.ndarray, train_end: int, val_end: int) -> SupervisedData:
        """Create supervised arrays and split masks for the full series."""
        arr = clean_series(values)
        z = self.scaler.transform(arr)
        X, y, tt = make_lagged_dataset(z, self.n_lags)
        reduced = self.reducer.transform(X)
        angles = self.angle_mapper.transform(reduced)
        idx = np.arange(len(y))
        return SupervisedData(
            X_lags=X,
            X_angles=angles,
            y=y,
            target_time=tt,
            train_idx=idx[tt < train_end],
            val_idx=idx[(tt >= train_end) & (tt < val_end)],
            test_idx=idx[tt >= val_end],
            angle_clip_fraction=self.angle_mapper.clip_fraction(reduced),
        )
