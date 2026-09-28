"""FT-Transformer (Gorishniy et al. 2021; complexity rank 8), a transformer for tables.

Every feature becomes a token: a numeric value ``x`` is embedded as ``x * w + b`` (one
learned ``w, b`` per feature) and a category by its own embedding table. A learned [CLS]
token is prepended, the sequence passes through pre-norm transformer blocks, and the [CLS]
output gives the CS logit. Implemented here in plain torch (CPU) with a scikit-learn
interface, so it runs in the same harness as every other model.

Preprocessing: ordinal-encoded categoricals (unknown -> its own "unseen" index), standardised
numerics and the missing-value indicator columns (0/1, treated as numerics). Training: AdamW
on binary cross-entropy, early stopping on a 10% stratified validation split of the fit rows.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import numpy.typing as npt
import optuna
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from robson_ml.feature_sets import FeatureSpec
from robson_ml.models.base import ModelSpec, make_preprocessor, pipeline, register, split_params

if TYPE_CHECKING:  # torch is imported lazily (see _ft_network)
    import torch

CATEGORICAL_PREFIX = "cat__"
MAX_EPOCHS = 60
PATIENCE = 6
BATCH_SIZE = 512
VALIDATION_SHARE = 0.1
N_THREADS = min(4, os.cpu_count() or 1)
WIDTHS = (16, 32)  # small token widths: CPU-only training on a slow machine


class FTTransformerClassifier(ClassifierMixin, BaseEstimator):
    """A scikit-learn classifier around :class:`_FTTransformer` (binary outcome)."""

    def __init__(
        self,
        d_token: int = 16,
        n_blocks: int = 1,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        seed: int = 0,
    ) -> None:
        self.d_token = d_token
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed

    def _tensors(self, x: Any) -> tuple[torch.Tensor, torch.Tensor]:
        import torch

        frame = pd.DataFrame(x)
        x_num = frame[self.numeric_].to_numpy(dtype=np.float32)
        # Ordinal codes are 0..k-1 with -1 for an unseen level: shift so unseen is 0.
        codes = frame[self.categorical_].to_numpy(dtype=np.float64) + 1
        x_cat = np.clip(np.nan_to_num(codes, nan=0), 0, np.asarray(self.cardinalities_) - 1)
        return torch.from_numpy(x_num), torch.from_numpy(x_cat.astype(np.int64))

    def fit(self, x: Any, y: npt.ArrayLike) -> Self:
        """Train with early stopping on a stratified 10% validation split of the rows."""
        import torch
        from torch import nn

        from robson_ml.models._ft_network import FTNetwork

        torch.manual_seed(self.seed)
        torch.set_num_threads(N_THREADS)
        frame = pd.DataFrame(x)
        y_arr = np.asarray(y).astype(np.float32)
        self.classes_ = np.array([0, 1])
        self.categorical_ = [c for c in frame.columns if str(c).startswith(CATEGORICAL_PREFIX)]
        self.numeric_ = [c for c in frame.columns if c not in self.categorical_]
        self.cardinalities_ = [
            int(np.nanmax(frame[c].to_numpy(dtype=np.float64), initial=0)) + 2
            for c in self.categorical_
        ]
        x_num, x_cat = self._tensors(frame)
        target = torch.from_numpy(y_arr)
        train, val = train_test_split(
            np.arange(len(y_arr)),
            test_size=VALIDATION_SHARE,
            stratify=y_arr,
            random_state=self.seed,
        )
        self.model_ = FTNetwork(
            len(self.numeric_), self.cardinalities_, self.d_token, self.n_blocks, self.dropout
        )
        optimiser = torch.optim.AdamW(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = nn.BCEWithLogitsLoss()
        generator = torch.Generator().manual_seed(self.seed)
        best, best_state, waited = np.inf, None, 0
        for _ in range(MAX_EPOCHS):
            self.model_.train()
            order = torch.from_numpy(train)[torch.randperm(len(train), generator=generator)]
            for start in range(0, len(order), BATCH_SIZE):
                batch = order[start : start + BATCH_SIZE]
                optimiser.zero_grad()
                loss = loss_fn(self.model_(x_num[batch], x_cat[batch]), target[batch])
                loss.backward()
                optimiser.step()
            self.model_.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(self.model_(x_num[val], x_cat[val]), target[val]))
            if val_loss < best - 1e-4:
                best, waited = val_loss, 0
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
            else:
                waited += 1
                if waited >= PATIENCE:
                    break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.eval()
        return self

    def predict_proba(self, x: Any) -> npt.NDArray[np.float64]:
        """Probabilities of (no CS, CS)."""
        import torch

        x_num, x_cat = self._tensors(x)
        with torch.no_grad():
            p = torch.sigmoid(self.model_(x_num, x_cat)).numpy().astype(np.float64)
        return np.column_stack([1 - p, p])

    def predict(self, x: Any) -> npt.NDArray[Any]:
        """The more probable class."""
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


def search_space(trial: optuna.Trial) -> dict[str, Any]:
    """Token width, depth, dropout, learning rate and weight decay."""
    return {
        "d_token": trial.suggest_categorical("d_token", list(WIDTHS)),
        "n_blocks": trial.suggest_int("n_blocks", 1, 2),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "lr": trial.suggest_float("lr", 3e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
    }


def build(params: dict[str, Any], fs: FeatureSpec) -> Pipeline:
    """Ordinal categoricals and standardised numerics into an FT-Transformer."""
    hyper, strategy, seed = split_params(params)
    prep = make_preprocessor(fs, strategy, seed, encoding="ordinal", scale=True)
    clf = FTTransformerClassifier(
        d_token=int(hyper["d_token"]),
        n_blocks=int(hyper["n_blocks"]),
        dropout=float(hyper["dropout"]),
        lr=float(hyper["lr"]),
        weight_decay=float(hyper["weight_decay"]),
        seed=seed,
    )
    return pipeline(prep, clf)


SPEC = register(
    ModelSpec(
        name="ft_transformer",
        family="neural",
        complexity_rank=8,
        handles_nan=False,
        native_categorical=False,
        needs_scaling=True,
        build=build,
        search_space=search_space,
    )
)
