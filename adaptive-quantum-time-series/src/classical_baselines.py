"""Module H - Classical baselines (MLP and LSTM) implementing OnlineLearner.

Both models:
1. Are pre-trained offline on the training split using identical lagged features;
2. Implement the :class:`src.trainer.OnlineLearner` protocol so they plug directly
   into :func:`src.trainer.run_online` for fair, controlled comparison;
3. Support warm-started online updates on historical observation windows;
4. Track exact parameter counts and provide checkpoint state arrays.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from sklearn.neural_network import MLPRegressor

from src.config import LSTMConfig, MLPConfig
from src.trainer import OnlineLearner, UpdateInfo

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# --------------------------------------------------------------------------- #
# MLP Baseline (Scikit-Learn)
# --------------------------------------------------------------------------- #
class MLPOnlineLearner:
    """Multi-Layer Perceptron regressor baseline conforming to OnlineLearner."""

    def __init__(
        self,
        model: MLPRegressor,
        input_dim: int,
        online_epochs: int = 2,
    ) -> None:
        self.model = model
        self.input_dim = input_dim
        self.online_epochs = max(1, online_epochs)

    @property
    def n_parameters(self) -> int:
        """Count total trainable weights and biases in the MLP."""
        if not hasattr(self.model, "coefs_") or not self.model.coefs_:
            # Estimate from architecture before first fit
            return 0
        total = sum(c.size for c in self.model.coefs_) + sum(b.size for b in self.model.intercepts_)
        return int(total)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_in = np.asarray(X, dtype=float)
        if X_in.ndim == 1:
            X_in = X_in.reshape(1, -1)
        preds = self.model.predict(X_in)
        return np.asarray(preds, dtype=float).ravel()

    def update(self, X: np.ndarray, y: np.ndarray) -> UpdateInfo:
        """Warm-started partial fit for online_epochs over the window."""
        X_in = np.asarray(X, dtype=float)
        y_in = np.asarray(y, dtype=float).ravel()
        if X_in.ndim == 1:
            X_in = X_in.reshape(1, -1)

        loss = float("nan")
        try:
            for _ in range(self.online_epochs):
                self.model.partial_fit(X_in, y_in)
            loss = float(np.mean((self.model.predict(X_in) - y_in) ** 2))
            success = np.isfinite(loss)
            msg = "ok" if success else "non-finite loss"
        except Exception as exc:
            return UpdateInfo(loss=float("nan"), n_iterations=0, success=False, message=str(exc))

        return UpdateInfo(
            loss=loss,
            n_iterations=self.online_epochs,
            success=success,
            message=msg,
        )

    def checkpoint_arrays(self) -> dict[str, np.ndarray]:
        """Save weights and biases for inspection/checkpoints."""
        arrays: dict[str, np.ndarray] = {}
        for i, (w, b) in enumerate(zip(self.model.coefs_, self.model.intercepts_)):
            arrays[f"layer_{i}_weight"] = np.asarray(w, dtype=float)
            arrays[f"layer_{i}_bias"] = np.asarray(b, dtype=float)
        return arrays


def train_offline_mlp(
    cfg: MLPConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> MLPOnlineLearner:
    """Train MLP baseline on the offline training split."""
    X_tr = np.asarray(X_train, dtype=float)
    y_tr = np.asarray(y_train, dtype=float).ravel()

    mlp = MLPRegressor(
        hidden_layer_sizes=cfg.hidden_layers,
        activation="relu",
        solver="adam",
        learning_rate_init=cfg.learning_rate,
        max_iter=cfg.epochs,
        alpha=cfg.alpha,
        random_state=seed,
        warm_start=True,
        early_stopping=False,
    )
    mlp.fit(X_tr, y_tr)
    return MLPOnlineLearner(
        model=mlp,
        input_dim=X_tr.shape[1],
        online_epochs=cfg.online_epochs,
    )


# --------------------------------------------------------------------------- #
# PyTorch LSTM Baseline
# --------------------------------------------------------------------------- #
if TORCH_AVAILABLE:

    class _LSTMNetwork(nn.Module):
        def __init__(self, input_dim: int, hidden_size: int) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
            )
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: (batch, seq_len, 1)
            out, _ = self.lstm(x)
            # take last sequence step representation
            last_step = out[:, -1, :]
            return self.head(last_step).squeeze(-1)

    class LSTMOnlineLearner:
        """PyTorch LSTM regressor baseline conforming to OnlineLearner."""

        def __init__(
            self,
            net: _LSTMNetwork,
            optimizer: optim.Optimizer,
            input_dim: int,
            online_epochs: int = 2,
        ) -> None:
            self.net = net
            self.optimizer = optimizer
            self.input_dim = input_dim
            self.online_epochs = max(1, online_epochs)
            self.criterion = nn.MSELoss()

        @property
        def n_parameters(self) -> int:
            return sum(p.numel() for p in self.net.parameters() if p.requires_grad)

        def predict(self, X: np.ndarray) -> np.ndarray:
            self.net.eval()
            X_in = np.asarray(X, dtype=np.float32)
            if X_in.ndim == 1:
                X_in = X_in.reshape(1, -1)
            # shape: (batch, seq_len, 1)
            t_X = torch.from_numpy(X_in).unsqueeze(-1)
            with torch.no_grad():
                preds = self.net(t_X).cpu().numpy().ravel()
            return np.asarray(preds, dtype=float)

        def update(self, X: np.ndarray, y: np.ndarray) -> UpdateInfo:
            self.net.train()
            X_in = np.asarray(X, dtype=np.float32)
            y_in = np.asarray(y, dtype=np.float32).ravel()
            if X_in.ndim == 1:
                X_in = X_in.reshape(1, -1)

            t_X = torch.from_numpy(X_in).unsqueeze(-1)
            t_y = torch.from_numpy(y_in)

            loss_val = float("nan")
            try:
                for _ in range(self.online_epochs):
                    self.optimizer.zero_grad()
                    pred = self.net(t_X)
                    loss = self.criterion(pred, t_y)
                    loss.backward()
                    self.optimizer.step()
                loss_val = float(loss.item())
                success = np.isfinite(loss_val)
                msg = "ok" if success else "non-finite loss"
            except Exception as exc:
                return UpdateInfo(loss=float("nan"), n_iterations=0, success=False, message=str(exc))

            return UpdateInfo(
                loss=loss_val,
                n_iterations=self.online_epochs,
                success=success,
                message=msg,
            )

        def checkpoint_arrays(self) -> dict[str, np.ndarray]:
            arrays: dict[str, np.ndarray] = {}
            for name, param in self.net.named_parameters():
                arrays[name] = param.detach().cpu().numpy()
            return arrays

    def train_offline_lstm(
        cfg: LSTMConfig,
        X_train: np.ndarray,
        y_train: np.ndarray,
        seed: int,
    ) -> LSTMOnlineLearner:
        """Train LSTM baseline on the offline training split."""
        torch.manual_seed(seed)
        np.random.seed(seed)

        X_tr = np.asarray(X_train, dtype=np.float32)
        y_tr = np.asarray(y_train, dtype=np.float32).ravel()
        input_dim = X_tr.shape[1]

        net = _LSTMNetwork(input_dim=input_dim, hidden_size=cfg.hidden_size)
        optimizer = optim.Adam(net.parameters(), lr=cfg.learning_rate)
        criterion = nn.MSELoss()

        t_X = torch.from_numpy(X_tr).unsqueeze(-1)
        t_y = torch.from_numpy(y_tr)

        net.train()
        for _ in range(cfg.epochs):
            optimizer.zero_grad()
            pred = net(t_X)
            loss = criterion(pred, t_y)
            loss.backward()
            optimizer.step()

        return LSTMOnlineLearner(
            net=net,
            optimizer=optimizer,
            input_dim=input_dim,
            online_epochs=cfg.online_epochs,
        )

else:

    class LSTMOnlineLearner:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyTorch is required for the LSTM baseline.")

    def train_offline_lstm(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-redef]
        raise ImportError("PyTorch is required for the LSTM baseline.")
