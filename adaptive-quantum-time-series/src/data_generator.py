"""Module B - synthetic non-stationary time-series generator.

The signal follows

    X(t) = A(t) * sin(Phi(t)) + eta(t),        eta(t) ~ N(0, sigma(t)^2)

where A(t), omega(t) = 2*pi/period(t) and sigma(t) are piecewise constant over
configurable *regimes*.  Two phase conventions are supported:

* ``instantaneous``: Phi(t) = omega(t) * t + phi  (the literal textbook formula;
  produces a phase jump whenever omega changes), and
* ``continuous`` (default): Phi(t) = phi + sum_{s<t} omega(s), which equals
  omega*t + phi inside a constant-frequency regime but keeps the waveform
  continuous across frequency changes so that regime shifts are caused by the
  change in dynamics rather than by an artificial discontinuity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config import DataConfig


@dataclass(frozen=True)
class SplitIndices:
    """Chronological split boundaries on the *time* axis.

    train = [0, train_end), validation = [train_end, val_end),
    test = [val_end, length).
    """

    train_end: int
    val_end: int
    length: int

    def split_of(self, time_index: int) -> str:
        """Return 'train', 'val' or 'test' for a time index."""
        if not 0 <= time_index < self.length:
            raise IndexError(f"time index {time_index} outside [0, {self.length})")
        if time_index < self.train_end:
            return "train"
        if time_index < self.val_end:
            return "val"
        return "test"


@dataclass(frozen=True)
class SyntheticSeries:
    """A generated series together with its ground-truth regime annotations."""

    time: np.ndarray
    values: np.ndarray
    clean: np.ndarray
    amplitude: np.ndarray
    omega: np.ndarray
    noise_std: np.ndarray
    regime_id: np.ndarray
    change_points: tuple[int, ...]
    regime_labels: tuple[str, ...]
    seed: int

    def __len__(self) -> int:
        return int(self.values.shape[0])

    def splits(self, train_fraction: float, val_fraction: float) -> SplitIndices:
        """Compute chronological split boundaries for this series."""
        return split_indices(len(self), train_fraction, val_fraction)

    def save_csv(self, path: Path) -> None:
        """Export series to CSV file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        header = "time,value,clean,amplitude,omega,noise_std,regime_id\n"
        with p.open("w", encoding="utf-8", newline="") as fh:
            fh.write(header)
            for i in range(len(self)):
                fh.write(
                    f"{self.time[i]},{self.values[i]:.10f},{self.clean[i]:.10f},"
                    f"{self.amplitude[i]:.6f},{self.omega[i]:.10f},"
                    f"{self.noise_std[i]:.6f},{self.regime_id[i]}\n"
                )


def _piecewise(cfg: DataConfig, attr: str) -> np.ndarray:
    """Expand a per-regime attribute into a per-time-step array."""
    out = np.empty(cfg.length, dtype=float)
    starts = [r.start for r in cfg.regimes] + [cfg.length]
    for regime, lo, hi in zip(cfg.regimes, starts[:-1], starts[1:]):
        out[lo:hi] = getattr(regime, attr)
    return out


def generate_series(cfg: DataConfig, seed: int) -> SyntheticSeries:
    """Generate a reproducible non-stationary sinusoid with regime shifts.

    Args:
        cfg: Data configuration (length, regimes, phase convention).
        seed: Seed for the Gaussian noise generator.

    Returns:
        SyntheticSeries with noisy values, clean signal and regime metadata.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative")
    rng = np.random.default_rng(seed)
    t = np.arange(cfg.length, dtype=int)
    amplitude = _piecewise(cfg, "amplitude")
    omega = 2.0 * np.pi / _piecewise(cfg, "period")
    sigma = _piecewise(cfg, "noise_std")

    if cfg.phase_mode == "continuous":
        phase = cfg.phase + np.concatenate(([0.0], np.cumsum(omega[:-1])))
    else:
        phase = omega * t + cfg.phase

    clean = amplitude * np.sin(phase)
    noise = rng.normal(0.0, 1.0, size=cfg.length) * sigma
    values = clean + noise

    regime_id = np.zeros(cfg.length, dtype=int)
    for k, regime in enumerate(cfg.regimes):
        regime_id[regime.start :] = k

    labels = tuple(r.label or f"regime_{k}" for k, r in enumerate(cfg.regimes))
    change_points = tuple(int(r.start) for r in cfg.regimes[1:])
    return SyntheticSeries(
        time=t,
        values=values,
        clean=clean,
        amplitude=amplitude,
        omega=omega,
        noise_std=sigma,
        regime_id=regime_id,
        change_points=change_points,
        regime_labels=labels,
        seed=seed,
    )


def split_indices(length: int, train_fraction: float, val_fraction: float) -> SplitIndices:
    """Compute chronological (leakage-free) split boundaries."""
    if length < 3:
        raise ValueError("series must contain at least 3 observations to split")
    train_end = int(length * train_fraction)
    val_end = int(length * (train_fraction + val_fraction))
    if not 0 < train_end < val_end < length:
        raise ValueError(
            f"invalid split for length {length}: train_end={train_end}, val_end={val_end}"
        )
    return SplitIndices(train_end=train_end, val_end=val_end, length=length)


def save_series(series: SyntheticSeries, splits: SplitIndices, directory: Path) -> dict[str, Path]:
    """Persist the series as CSV + NPZ and its metadata as JSON.

    Returns:
        Mapping from artefact name to the written path.
    """
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"series_seed{series.seed}"
    csv_path = directory / f"{stem}.csv"
    npz_path = directory / f"{stem}.npz"
    meta_path = directory / f"{stem}_meta.json"

    split_names = np.array([splits.split_of(int(i)) for i in series.time])
    header = "time,value,clean,amplitude,omega,noise_std,regime_id,split"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(header + "\n")
        for i in range(len(series)):
            fh.write(
                f"{series.time[i]},{series.values[i]:.10f},{series.clean[i]:.10f},"
                f"{series.amplitude[i]:.6f},{series.omega[i]:.10f},"
                f"{series.noise_std[i]:.6f},{series.regime_id[i]},{split_names[i]}\n"
            )
    np.savez(
        npz_path,
        time=series.time,
        values=series.values,
        clean=series.clean,
        amplitude=series.amplitude,
        omega=series.omega,
        noise_std=series.noise_std,
        regime_id=series.regime_id,
    )
    meta = {
        "seed": series.seed,
        "length": len(series),
        "change_points": list(series.change_points),
        "regime_labels": list(series.regime_labels),
        "splits": {
            "train": [0, splits.train_end],
            "val": [splits.train_end, splits.val_end],
            "test": [splits.val_end, splits.length],
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"csv": csv_path, "npz": npz_path, "meta": meta_path}


def load_series(npz_path: Path, meta_path: Path) -> SyntheticSeries:
    """Load a series previously written by :func:`save_series`."""
    if not npz_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"missing dataset files: {npz_path} / {meta_path}")
    arrays = np.load(npz_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return SyntheticSeries(
        time=arrays["time"],
        values=arrays["values"],
        clean=arrays["clean"],
        amplitude=arrays["amplitude"],
        omega=arrays["omega"],
        noise_std=arrays["noise_std"],
        regime_id=arrays["regime_id"],
        change_points=tuple(meta["change_points"]),
        regime_labels=tuple(meta["regime_labels"]),
        seed=int(meta["seed"]),
    )


# Alias for backward and test compatibility
generate_synthetic_series = generate_series
