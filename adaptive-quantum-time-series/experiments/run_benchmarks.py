"""Experiment runner for reproducible multi-seed benchmarking.

Runs the full comparison across:
1. Adaptive-Window PQC
2. Fixed-Window PQC (W=10, W=30, W=50)
3. Frozen PQC (No online adaptation)
4. Classical MLP Regressor
5. Classical PyTorch LSTM

Usage:
    python experiments/run_benchmarks.py --config config/config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config
from src.main import run_benchmark_mode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Quantum Time-Series Benchmarks")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to YAML configuration file (default: config/config.yaml)",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).resolve().parents[1] / cfg_path

    print(f"Executing Benchmarks with configuration: {cfg_path}")
    cfg = load_config(cfg_path)
    run_benchmark_mode(cfg)


if __name__ == "__main__":
    main()
