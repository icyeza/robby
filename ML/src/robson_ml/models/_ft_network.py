"""The FT-Transformer network (torch). Imported only when an FT-Transformer is fitted or used,
so that importing the model zoo does not load torch (it needs a lot of memory just to load)."""

from __future__ import annotations

import torch
from torch import nn

HEADS = {16: 2, 32: 4}  # attention heads per token width


class FTNetwork(nn.Module):
    """Feature tokenizer + pre-norm transformer blocks + [CLS] head, one logit per row."""

    def __init__(
        self, n_numeric: int, cardinalities: list[int], d: int, n_blocks: int, dropout: float
    ) -> None:
        super().__init__()
        self.num_weight = nn.Parameter(torch.randn(n_numeric, d) * d**-0.5)
        self.num_bias = nn.Parameter(torch.zeros(n_numeric, d))
        self.cat_embeddings = nn.ModuleList(nn.Embedding(c, d) for c in cardinalities)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * d**-0.5)
        block = nn.TransformerEncoderLayer(
            d,
            HEADS[d],
            dim_feedforward=2 * d,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, n_blocks, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        tokens = [x_num.unsqueeze(-1) * self.num_weight + self.num_bias]
        if self.cat_embeddings:
            tokens.append(
                torch.stack([e(x_cat[:, i]) for i, e in enumerate(self.cat_embeddings)], dim=1)
            )
        seq = torch.cat([self.cls.expand(x_num.shape[0], -1, -1), *tokens], dim=1)
        return self.head(self.encoder(seq)[:, 0]).squeeze(-1)
