"""Central, validated experiment configuration.

All tunable values of the project live in a YAML file (``config/config.yaml``)
and are parsed into immutable, type-checked dataclasses defined here.  Every
module receives the dataclass it needs instead of reading raw dictionaries, so
invalid settings fail loudly at start-up with a message that names the
offending key.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config" / "config.yaml"
SMOKE_CONFIG_PATH: Path = PROJECT_ROOT / "config" / "smoke.yaml"


class ConfigError(ValueError):
    """Raised when a configuration file or value is invalid."""


# --------------------------------------------------------------------------- #
# Section dataclasses
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExperimentConfig:
    """Global experiment metadata and output locations."""

    name: str = "adaptive_pqc"
    seed: int = 42
    output_dir: str = "outputs"
    data_dir: str = "data"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must be a non-empty string")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


@dataclass(frozen=True)
class RegimeConfig:
    """One stationary segment of the synthetic signal.

    ``period`` is expressed in time steps; the angular frequency is
    ``omega = 2*pi/period``.
    """

    start: int
    amplitude: float
    period: float
    noise_std: float
    label: str = ""

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("regime start must be >= 0")
        if self.amplitude <= 0:
            raise ValueError("regime amplitude must be > 0")
        if self.period <= 2.0:
            raise ValueError("regime period must be > 2 time steps (Nyquist limit)")
        if self.noise_std < 0:
            raise ValueError("regime noise_std must be >= 0")


@dataclass(frozen=True)
class DataConfig:
    """Synthetic data generation and chronological splitting."""

    length: int = 500
    regimes: tuple[RegimeConfig, ...] = (
        RegimeConfig(start=0, amplitude=1.0, period=20.0, noise_std=0.05, label="stable"),
    )
    phase: float = 0.0
    phase_mode: str = "continuous"
    train_fraction: float = 0.35
    val_fraction: float = 0.15
    missing: str = "raise"

    def __post_init__(self) -> None:
        if self.missing not in {"raise", "interpolate"}:
            raise ValueError("missing must be 'raise' or 'interpolate'")
        if self.length < 30:
            raise ValueError("length must be >= 30 time steps")
        if not self.regimes:
            raise ValueError("at least one regime is required")
        starts = [r.start for r in self.regimes]
        if starts[0] != 0:
            raise ValueError("the first regime must start at index 0")
        if any(b <= a for a, b in zip(starts, starts[1:])):
            raise ValueError("regime start indices must be strictly increasing")
        if starts[-1] >= self.length:
            raise ValueError("every regime must start before the end of the series")
        if self.phase_mode not in {"continuous", "instantaneous"}:
            raise ValueError("phase_mode must be 'continuous' or 'instantaneous'")
        if not 0 < self.train_fraction < 1 or not 0 < self.val_fraction < 1:
            raise ValueError("train_fraction and val_fraction must lie in (0, 1)")
        if self.train_fraction + self.val_fraction >= 0.95:
            raise ValueError("train_fraction + val_fraction must leave >= 5% for testing")


@dataclass(frozen=True)
class FeatureConfig:
    """Lag features and the classical-to-angle mapping."""

    n_lags: int = 4
    reduction: str = "pca"
    headroom: float = 2.0
    angle_min: float = 0.0
    angle_max: float = math.pi

    def __post_init__(self) -> None:
        if self.n_lags < 1:
            raise ValueError("n_lags must be >= 1")
        if self.reduction not in {"pca", "reupload"}:
            raise ValueError("reduction must be 'pca' or 'reupload'")
        if self.headroom < 1.0:
            raise ValueError("headroom must be >= 1.0")
        if not self.angle_max > self.angle_min:
            raise ValueError("angle_max must be greater than angle_min")
        if self.angle_max - self.angle_min > 2 * math.pi + 1e-9:
            raise ValueError("angle range wider than 2*pi aliases rotation angles")


@dataclass(frozen=True)
class QuantumConfig:
    """Parameterized-quantum-circuit architecture and simulator settings."""

    n_qubits: int = 4
    encoding: str = "ry_rz"
    encoding_entanglement: str = "linear"
    entangler: str = "cx"
    ansatz_layers: int = 2
    rotation_gates: tuple[str, ...] = ("ry", "rz")
    ansatz_entanglement: str = "circular"
    readout_qubits: tuple[int, ...] = ()
    backend: str = "aer"
    max_batch_size: int = 8192

    def __post_init__(self) -> None:
        if not 1 <= self.n_qubits <= 12:
            raise ValueError("n_qubits must be between 1 and 12 for CPU simulation")
        if self.encoding not in {"ry", "ry_rz", "zz"}:
            raise ValueError("encoding must be one of 'ry', 'ry_rz', 'zz'")
        for name, value in (
            ("encoding_entanglement", self.encoding_entanglement),
            ("ansatz_entanglement", self.ansatz_entanglement),
        ):
            if value not in {"none", "linear", "circular"}:
                raise ValueError(f"{name} must be 'none', 'linear' or 'circular'")
        if self.entangler not in {"cx", "cz"}:
            raise ValueError("entangler must be 'cx' or 'cz'")
        if self.ansatz_layers < 1:
            raise ValueError("ansatz_layers must be >= 1")
        if not self.rotation_gates or any(
            g not in {"rx", "ry", "rz"} for g in self.rotation_gates
        ):
            raise ValueError("rotation_gates must be a non-empty subset of rx/ry/rz")
        if any(not 0 <= q < self.n_qubits for q in self.readout_qubits):
            raise ValueError("readout_qubits must be valid qubit indices")
        if len(set(self.readout_qubits)) != len(self.readout_qubits):
            raise ValueError("readout_qubits must not contain duplicates")
        if self.backend not in {"aer", "statevector"}:
            raise ValueError("backend must be 'aer' or 'statevector'")
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")

    @property
    def resolved_readout_qubits(self) -> tuple[int, ...]:
        """Readout qubits; an empty tuple in the YAML means *all* qubits."""
        return self.readout_qubits or tuple(range(self.n_qubits))


@dataclass(frozen=True)
class OptimizerConfig:
    """Hyper-parameters of a single optimizer (Adam, SPSA or COBYLA)."""

    name: str = "adam"
    learning_rate: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    grad_clip: float = 5.0
    l2_reg: float = 0.0
    spsa_a: float = 0.1
    spsa_c: float = 0.1
    spsa_alpha: float = 0.602
    spsa_gamma: float = 0.101
    spsa_stability: float = 10.0
    cobyla_rhobeg: float = 0.3

    def __post_init__(self) -> None:
        if self.name not in {"adam", "spsa", "cobyla"}:
            raise ValueError("optimizer name must be 'adam', 'spsa' or 'cobyla'")
        positive = {
            "learning_rate": self.learning_rate,
            "epsilon": self.epsilon,
            "grad_clip": self.grad_clip,
            "spsa_a": self.spsa_a,
            "spsa_c": self.spsa_c,
            "cobyla_rhobeg": self.cobyla_rhobeg,
        }
        for key, value in positive.items():
            if not value > 0:
                raise ValueError(f"{key} must be > 0")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("beta1 and beta2 must lie in [0, 1)")
        if self.l2_reg < 0 or self.spsa_stability < 0:
            raise ValueError("l2_reg and spsa_stability must be >= 0")


@dataclass(frozen=True)
class TrainingConfig:
    """Offline (batch) pre-training of the PQC on the training split."""

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    iterations: int = 60
    eval_every: int = 1
    init_scale: float = 0.1
    early_stopping_patience: int = 0

    def __post_init__(self) -> None:
        if self.iterations < 1 or self.eval_every < 1:
            raise ValueError("iterations and eval_every must be >= 1")
        if self.init_scale < 0:
            raise ValueError("init_scale must be >= 0")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be >= 0 (0 disables it)")


@dataclass(frozen=True)
class OnlineConfig:
    """Online (streaming, test-then-train) adaptation of the PQC."""

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    steps_per_update: int = 2
    update_every: int = 1
    checkpoint_every: int = 0

    def __post_init__(self) -> None:
        if self.steps_per_update < 1 or self.update_every < 1:
            raise ValueError("steps_per_update and update_every must be >= 1")
        if self.checkpoint_every < 0:
            raise ValueError("checkpoint_every must be >= 0 (0 disables it)")


@dataclass(frozen=True)
class AdaptiveWindowConfig:
    """Parameters of the adaptive quantum windowing controller."""

    w_base: int = 30
    w_min: int = 8
    w_max: int = 60
    tau_high: float = 0.25
    tau_low: float = 0.08
    gamma_down: float = 0.5
    gamma_up: float = 1.25
    error_smoothing: float = 1.0
    cooldown: int = 2
    patience_low: int = 3

    def __post_init__(self) -> None:
        if not 1 <= self.w_min <= self.w_base <= self.w_max:
            raise ValueError("window bounds must satisfy 1 <= w_min <= w_base <= w_max")
        if not 0 <= self.tau_low < self.tau_high:
            raise ValueError("thresholds must satisfy 0 <= tau_low < tau_high")
        if not 0 < self.gamma_down < 1:
            raise ValueError("gamma_down must lie in (0, 1) so the window shrinks")
        if not self.gamma_up > 1:
            raise ValueError("gamma_up must be > 1 so the window expands")
        if not 0 < self.error_smoothing <= 1:
            raise ValueError("error_smoothing (EMA alpha) must lie in (0, 1]")
        if self.cooldown < 0 or self.patience_low < 1:
            raise ValueError("cooldown must be >= 0 and patience_low >= 1")


@dataclass(frozen=True)
class MLPConfig:
    """Classical multilayer-perceptron baseline (scikit-learn)."""

    enabled: bool = True
    hidden_layers: tuple[int, ...] = (16,)
    learning_rate: float = 0.01
    epochs: int = 200
    alpha: float = 1e-4
    online_window: int = 30
    online_epochs: int = 2

    def __post_init__(self) -> None:
        if not self.hidden_layers or any(h < 1 for h in self.hidden_layers):
            raise ValueError("hidden_layers must contain positive integers")
        if self.learning_rate <= 0 or self.alpha < 0:
            raise ValueError("learning_rate must be > 0 and alpha >= 0")
        if self.epochs < 1 or self.online_window < 1 or self.online_epochs < 1:
            raise ValueError("epochs, online_window and online_epochs must be >= 1")


@dataclass(frozen=True)
class LSTMConfig:
    """Classical LSTM baseline (PyTorch)."""

    enabled: bool = True
    hidden_size: int = 16
    learning_rate: float = 0.01
    epochs: int = 200
    online_window: int = 30
    online_epochs: int = 2

    def __post_init__(self) -> None:
        if self.hidden_size < 1 or self.learning_rate <= 0:
            raise ValueError("hidden_size must be >= 1 and learning_rate > 0")
        if self.epochs < 1 or self.online_window < 1 or self.online_epochs < 1:
            raise ValueError("epochs, online_window and online_epochs must be >= 1")


@dataclass(frozen=True)
class BaselineConfig:
    """Which comparison models to run."""

    fixed_windows: tuple[int, ...] = (10, 30, 50)
    include_offline_only_pqc: bool = True
    mlp: MLPConfig = field(default_factory=MLPConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)

    def __post_init__(self) -> None:
        if any(w < 1 for w in self.fixed_windows):
            raise ValueError("fixed_windows must contain positive integers")
        if len(set(self.fixed_windows)) != len(self.fixed_windows):
            raise ValueError("fixed_windows must not contain duplicates")


@dataclass(frozen=True)
class EvaluationConfig:
    """Metric definitions (all tolerances are in original signal units)."""

    mape_min_abs_target: float = 0.1
    adaptation_tolerance: float = 0.2
    rolling_window: int = 5
    shift_window: int = 15

    def __post_init__(self) -> None:
        if self.mape_min_abs_target <= 0 or self.adaptation_tolerance <= 0:
            raise ValueError("mape_min_abs_target and adaptation_tolerance must be > 0")
        if self.rolling_window < 1 or self.shift_window < 1:
            raise ValueError("rolling_window and shift_window must be >= 1")


@dataclass(frozen=True)
class BenchmarkConfig:
    """Seeds used for multi-seed benchmarking."""

    seeds: tuple[int, ...] = (0, 1, 2)

    def __post_init__(self) -> None:
        if not self.seeds or any(s < 0 for s in self.seeds):
            raise ValueError("seeds must be a non-empty list of non-negative integers")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must not contain duplicates")


@dataclass(frozen=True)
class Config:
    """Root configuration object."""

    data: DataConfig
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    online: OnlineConfig = field(default_factory=OnlineConfig)
    adaptive_window: AdaptiveWindowConfig = field(default_factory=AdaptiveWindowConfig)
    baselines: BaselineConfig = field(default_factory=BaselineConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    def __post_init__(self) -> None:
        train_len = int(self.data.length * self.data.train_fraction)
        if train_len - self.features.n_lags < 10:
            raise ValueError(
                "training split is too short: need at least 10 supervised samples "
                f"(train length {train_len}, n_lags {self.features.n_lags})"
            )
        if (
            self.features.n_lags > self.quantum.n_qubits
            and self.features.reduction == "pca"
            and self.quantum.n_qubits > train_len - self.features.n_lags
        ):
            raise ValueError("not enough training samples to fit PCA feature reduction")

    @property
    def output_root(self) -> Path:
        """Absolute output directory (relative paths resolve to the project root)."""
        return _resolve_path(self.experiment.output_dir)

    @property
    def data_root(self) -> Path:
        """Absolute data directory (relative paths resolve to the project root)."""
        return _resolve_path(self.experiment.data_dir)


# --------------------------------------------------------------------------- #
# Generic dict -> dataclass conversion with type checking
# --------------------------------------------------------------------------- #
def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _convert(hint: Any, value: Any, where: str) -> Any:
    """Convert a raw YAML value to the annotated type, raising ConfigError."""
    if is_dataclass(hint):
        return _from_dict(hint, value, where)
    origin = get_origin(hint)
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{where}: expected a list, got {type(value).__name__}")
        (item_hint, _ellipsis) = get_args(hint)
        return tuple(_convert(item_hint, v, f"{where}[{i}]") for i, v in enumerate(value))
    if hint is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: expected true/false, got {value!r}")
        return value
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: expected an integer, got {value!r}")
        return value
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where}: expected a number, got {value!r}")
        if not math.isfinite(float(value)):
            raise ConfigError(f"{where}: value must be finite, got {value!r}")
        return float(value)
    if hint is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where}: expected a string, got {value!r}")
        return value
    raise ConfigError(f"{where}: unsupported annotation {hint!r}")  # pragma: no cover


def _from_dict(cls: type, data: Any, where: str) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {unknown}; allowed: {sorted(known)}")
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            if f.default is MISSING and f.default_factory is MISSING:
                raise ConfigError(f"{where}.{f.name}: required key is missing")
            continue
        kwargs[f.name] = _convert(hints[f.name], data[f.name], f"{where}.{f.name}")
    try:
        return cls(**kwargs)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{where}: {exc}") from exc


def config_from_dict(raw: dict[str, Any]) -> Config:
    """Build and validate a :class:`Config` from a plain dictionary."""
    return _from_dict(Config, raw, "config")


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate a YAML configuration file.

    Args:
        path: Path to the YAML file. Defaults to ``config/config.yaml``.

    Raises:
        ConfigError: If the file is missing, unparsable or invalid.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.is_absolute() and not cfg_path.exists():
        cfg_path = PROJECT_ROOT / cfg_path
    if not cfg_path.is_file():
        raise ConfigError(f"configuration file not found: {cfg_path}")
    try:
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse YAML in {cfg_path}: {exc}") from exc
    if raw is None:
        raise ConfigError(f"configuration file is empty: {cfg_path}")
    return config_from_dict(raw)


def config_to_dict(cfg: Config) -> dict[str, Any]:
    """Serialise a configuration to JSON/YAML-friendly primitives."""

    def _plain(obj: Any) -> Any:
        if isinstance(obj, tuple):
            return [_plain(v) for v in obj]
        if isinstance(obj, dict):
            return {k: _plain(v) for k, v in obj.items()}
        return obj

    return _plain(dataclasses.asdict(cfg))


def with_overrides(cfg: Config, **section_updates: dict[str, Any]) -> Config:
    """Return a new validated Config with selected section fields replaced.

    Example:
        ``with_overrides(cfg, experiment={"seed": 3})``
    """
    raw = config_to_dict(cfg)
    for section, updates in section_updates.items():
        if section not in raw:
            raise ConfigError(f"unknown configuration section '{section}'")
        if not isinstance(updates, dict):
            raise ConfigError(f"overrides for '{section}' must be a mapping")
        raw[section] = {**raw[section], **updates}
    return config_from_dict(raw)
