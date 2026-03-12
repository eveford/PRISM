from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from prism.reconstruction.dataset import ReconstructionDataset, build_pair_arrays, split_indices_by_id
from prism.reconstruction.model import PrismReconstructionModel
from prism.reconstruction.train import apply_mode_to_batch, reconstruction_loss, set_global_seed


@dataclass(frozen=True)
class SparsePruneResult:
    features: list[str]
    history: list[dict[str, object]]


def _train_one_round(
    arrays,
    feature_columns: list[str],
    key_features: list[str],
    *,
    hidden_dim: int,
    depth: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    cosine_weight: float,
    seed: int,
    device: str,
):
    set_global_seed(seed)
    train_indices, _ = split_indices_by_id(arrays.ids, train_ratio=0.8, seed=seed)
    train_dataset = ReconstructionDataset.from_indices(arrays, train_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = PrismReconstructionModel(
        baseline_dim=len(feature_columns),
        key_dim=len(key_features),
        hidden_dim=hidden_dim,
        depth=depth,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in range(max(1, epochs)):
        model.train()
        for baseline, key, target, _ in train_loader:
            baseline = baseline.to(device)
            key = key.to(device)
            target = target.to(device)
            baseline, key = apply_mode_to_batch(baseline, key, mode="prism")
            optimizer.zero_grad(set_to_none=True)
            prediction = model(baseline, key)
            loss, _ = reconstruction_loss(prediction, target, cosine_weight=cosine_weight)
            loss.backward()
            optimizer.step()
    return model, train_loader


def _gradient_importance(
    model: PrismReconstructionModel,
    loader: DataLoader,
    *,
    device: str,
    cosine_weight: float,
) -> np.ndarray:
    model.eval()
    importance = None
    sample_count = 0

    for baseline, key, target, _ in loader:
        baseline = baseline.to(device)
        key = key.to(device).requires_grad_(True)
        target = target.to(device)
        model.zero_grad(set_to_none=True)
        prediction = model(baseline, key)
        loss, _ = reconstruction_loss(prediction, target, cosine_weight=cosine_weight)
        loss.backward()
        gradients = key.grad.detach().abs().sum(dim=0).cpu().numpy()
        if importance is None:
            importance = gradients
        else:
            importance += gradients
        sample_count += baseline.shape[0]

    if importance is None:
        raise ValueError("Could not compute gradient importances from an empty dataset.")
    return importance / max(1, sample_count)


def iterative_sparse_prune(
    tables: dict[int, object],
    feature_columns: list[str],
    pairs: list[tuple[int, int]],
    initial_features: list[str],
    *,
    target_panel_size: int,
    prune_step: int,
    hidden_dim: int,
    depth: int,
    dropout: float,
    epochs_per_round: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    cosine_weight: float,
    seed: int,
    device: str,
) -> SparsePruneResult:
    current_features = list(initial_features)
    history: list[dict[str, object]] = []

    while len(current_features) > target_panel_size:
        arrays = build_pair_arrays(
            tables=tables,
            feature_columns=feature_columns,
            key_features=current_features,
            pairs=pairs,
        )
        model, train_loader = _train_one_round(
            arrays,
            feature_columns,
            current_features,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
            epochs=epochs_per_round,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            cosine_weight=cosine_weight,
            seed=seed,
            device=device,
        )
        importance = _gradient_importance(
            model,
            train_loader,
            device=device,
            cosine_weight=cosine_weight,
        )
        k = min(prune_step, len(current_features) - target_panel_size)
        remove_indices = np.argsort(importance)[:k]
        removed = [current_features[index] for index in remove_indices]
        current_features = [
            feature
            for index, feature in enumerate(current_features)
            if index not in set(remove_indices.tolist())
        ]
        history.append(
            {
                "round": len(history) + 1,
                "removed_features": removed,
                "remaining_feature_count": len(current_features),
            }
        )

    return SparsePruneResult(features=current_features, history=history)


def save_sparse_prune_result(result: SparsePruneResult, output_dir: str | Path, filename: str = "prism_panel_64.json") -> Path:
    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.features, indent=2), encoding="utf-8")
    history_path = output_path.with_suffix(".history.json")
    history_path.write_text(json.dumps(result.history, indent=2), encoding="utf-8")
    return output_path
