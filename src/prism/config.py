from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from prism.paths import require_relative_path


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}.")
    return payload


def _parse_pairs(raw_pairs: list[list[int]] | list[tuple[int, int]]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for pair in raw_pairs:
        if len(pair) != 2:
            raise ValueError(f"Invalid year pair: {pair}")
        pairs.append((int(pair[0]), int(pair[1])))
    return pairs


@dataclass(frozen=True)
class SelectionConfig:
    initial_panel_size: int = 256
    candidate_pool_size: int = 1024
    max_pairwise_correlation: float = 0.85
    target_panel_size: int = 64
    prune_step: int = 16
    epochs_per_round: int = 6


@dataclass(frozen=True)
class ReconstructionConfig:
    hidden_dim: int = 512
    depth: int = 2
    dropout: float = 0.1
    batch_size: int = 64
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    cosine_weight: float = 1.0
    train_ratio: float = 0.8
    seed: int = 42
    device: str = "cpu"


@dataclass(frozen=True)
class EvaluationConfig:
    age_alpha_grid: tuple[float, ...] = (1e-8, 1e-5, 1e-4, 1e-3, 1e-2, 5e-1)
    disease_years: tuple[int, ...] = (2007, 2012, 2020)
    disease_epochs: int = 10
    disease_hidden_dim: int = 128
    disease_dropout: float = 0.2
    disease_lr: float = 1e-3


@dataclass(frozen=True)
class PaperConfig:
    config_path: Path
    train_pairs: list[tuple[int, int]]
    eval_pairs: list[tuple[int, int]]
    panel_path: Path
    disease_whitelist_path: Path
    display_year_aliases: dict[int, int] = field(default_factory=dict)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @property
    def all_years(self) -> list[int]:
        years = {year for pair in self.train_pairs + self.eval_pairs for year in pair}
        return sorted(years)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PaperConfig":
        config_path = Path(path).resolve()
        payload = _load_yaml(config_path)
        base_dir = config_path.parent
        panel_path = require_relative_path(payload["panel_path"], label="panel_path")
        disease_path = require_relative_path(
            payload["disease_whitelist_path"],
            label="disease_whitelist_path",
        )
        selection = SelectionConfig(**payload.get("selection", {}))
        reconstruction = ReconstructionConfig(**payload.get("reconstruction", {}))
        evaluation = EvaluationConfig(**payload.get("evaluation", {}))
        aliases = {int(k): int(v) for k, v in payload.get("display_year_aliases", {}).items()}
        return cls(
            config_path=config_path,
            train_pairs=_parse_pairs(payload["train_pairs"]),
            eval_pairs=_parse_pairs(payload["eval_pairs"]),
            panel_path=(base_dir / panel_path).resolve(),
            disease_whitelist_path=(base_dir / disease_path).resolve(),
            display_year_aliases=aliases,
            selection=selection,
            reconstruction=reconstruction,
            evaluation=evaluation,
        )
