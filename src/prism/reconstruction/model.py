from __future__ import annotations

import torch
import torch.nn as nn


def _mlp_block(input_dim: int, output_dim: int, dropout: float) -> list[nn.Module]:
    return [
        nn.Linear(input_dim, output_dim),
        nn.GELU(),
        nn.LayerNorm(output_dim),
        nn.Dropout(dropout),
    ]


class PrismReconstructionModel(nn.Module):
    def __init__(
        self,
        baseline_dim: int,
        key_dim: int,
        hidden_dim: int = 512,
        depth: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.baseline_dim = baseline_dim
        self.key_dim = key_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.dropout = dropout

        baseline_layers: list[nn.Module] = []
        in_dim = baseline_dim
        for _ in range(max(1, depth)):
            baseline_layers.extend(_mlp_block(in_dim, hidden_dim, dropout))
            in_dim = hidden_dim
        self.baseline_encoder = nn.Sequential(*baseline_layers)

        key_layers: list[nn.Module] = []
        in_dim = key_dim
        for _ in range(max(1, depth)):
            key_layers.extend(_mlp_block(in_dim, hidden_dim, dropout))
            in_dim = hidden_dim
        self.key_encoder = nn.Sequential(*key_layers)

        decoder_layers: list[nn.Module] = []
        in_dim = hidden_dim * 2
        for _ in range(max(1, depth)):
            decoder_layers.extend(_mlp_block(in_dim, hidden_dim, dropout))
            in_dim = hidden_dim
        decoder_layers.append(nn.Linear(in_dim, baseline_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, baseline_x: torch.Tensor, key_x: torch.Tensor) -> torch.Tensor:
        baseline_repr = self.baseline_encoder(baseline_x)
        key_repr = self.key_encoder(key_x)
        combined = torch.cat([baseline_repr, key_repr], dim=-1)
        return self.decoder(combined)
