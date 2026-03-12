from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from prism.preprocessing.normalize import ProteomeNormalizer
from prism.reconstruction.dataset import PairArrays, ReconstructionDataset, split_indices_by_id
from prism.reconstruction.model import PrismReconstructionModel


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_mode_to_batch(
    baseline: torch.Tensor,
    key: torch.Tensor,
    *,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mode == "baseline_only":
        key = torch.zeros_like(key)
    elif mode == "key_only":
        baseline = torch.zeros_like(baseline)
    elif mode != "prism":
        raise ValueError(f"Unsupported reconstruction mode: {mode}")
    return baseline, key


def reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    cosine_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    mse = F.mse_loss(prediction, target)
    cosine = F.cosine_similarity(prediction, target, dim=1).mean()
    total = mse + float(cosine_weight) * (1.0 - cosine)
    return total, {
        "mse": float(mse.item()),
        "cosine": float(cosine.item()),
        "loss": float(total.item()),
    }


def _epoch(
    model: PrismReconstructionModel,
    loader: DataLoader,
    *,
    device: str,
    cosine_weight: float,
    optimizer: torch.optim.Optimizer | None,
    mode: str,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "mse": 0.0, "cosine": 0.0}
    steps = 0

    for baseline, key, target, _ in loader:
        baseline = baseline.to(device)
        key = key.to(device)
        target = target.to(device)
        baseline, key = apply_mode_to_batch(baseline, key, mode=mode)

        with torch.set_grad_enabled(training):
            prediction = model(baseline, key)
            loss, metrics = reconstruction_loss(prediction, target, cosine_weight=cosine_weight)
            if training and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        for metric_name in totals:
            totals[metric_name] += metrics[metric_name]
        steps += 1

    return {name: value / max(1, steps) for name, value in totals.items()}


def train_reconstruction_model(
    arrays: PairArrays,
    *,
    normalizer: ProteomeNormalizer,
    hidden_dim: int,
    depth: int,
    dropout: float,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    cosine_weight: float,
    train_ratio: float,
    seed: int,
    device: str,
    output_dir: str | Path,
    mode: str = "prism",
) -> tuple[Path, Path]:
    set_global_seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_indices, val_indices = split_indices_by_id(
        arrays.ids,
        train_ratio=train_ratio,
        seed=seed,
    )
    train_dataset = ReconstructionDataset.from_indices(arrays, train_indices)
    val_dataset = ReconstructionDataset.from_indices(arrays, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = PrismReconstructionModel(
        baseline_dim=len(arrays.feature_columns),
        key_dim=len(arrays.key_features),
        hidden_dim=hidden_dim,
        depth=depth,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: list[dict[str, object]] = []
    best_state = None
    best_val_loss = float("inf")

    for epoch in range(1, max(1, epochs) + 1):
        train_metrics = _epoch(
            model,
            train_loader,
            device=device,
            cosine_weight=cosine_weight,
            optimizer=optimizer,
            mode=mode,
        )
        val_metrics = _epoch(
            model,
            val_loader,
            device=device,
            cosine_weight=cosine_weight,
            optimizer=None,
            mode=mode,
        )
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    checkpoint_path = output_path / f"{mode}_checkpoint.pt"
    metrics_path = output_path / f"{mode}_metrics.json"
    checkpoint_payload = {
        "state_dict": best_state if best_state is not None else model.state_dict(),
        "mode": mode,
        "model_config": {
            "baseline_dim": len(arrays.feature_columns),
            "key_dim": len(arrays.key_features),
            "hidden_dim": hidden_dim,
            "depth": depth,
            "dropout": dropout,
        },
        "feature_columns": arrays.feature_columns,
        "key_features": arrays.key_features,
        "normalizer": normalizer.to_metadata(),
        "train_ratio": train_ratio,
        "seed": seed,
    }
    torch.save(checkpoint_payload, checkpoint_path)
    metrics_path.write_text(
        json.dumps(
            {
                "mode": mode,
                "history": history,
                "best_val_loss": best_val_loss,
                "train_sample_count": int(len(train_dataset)),
                "val_sample_count": int(len(val_dataset)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return checkpoint_path, metrics_path
