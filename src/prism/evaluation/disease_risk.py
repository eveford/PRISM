from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from prism.io.loaders import (
    GeneratedSeriesManifest,
    load_generated_series_manifest,
    load_generated_tables,
    seq_columns,
)
from prism.preprocessing.normalize import ProteomeNormalizer, normalize_tables
from prism.reconstruction.dataset import split_indices_by_id
from prism.reconstruction.train import set_global_seed

YEAR_SUFFIX = {
    2007: "07",
    2012: "12",
    2020: "20",
}


@dataclass(frozen=True)
class DiseaseBenchmarkResult:
    summary: dict[str, object]
    summary_path: Path


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, *, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        mid_dim = max(hidden_dim // 2, 16)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, mid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, 3),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def _normalize_observed_tables(
    observed_tables: dict[int, pd.DataFrame],
    *,
    checkpoint_path: str | Path | None,
) -> dict[int, pd.DataFrame]:
    feature_columns = [column for column in observed_tables[min(observed_tables.keys())].columns if str(column).startswith("seq")]
    if checkpoint_path is None:
        normalized_tables, _ = normalize_tables(observed_tables, feature_columns)
        return normalized_tables

    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    normalizer = ProteomeNormalizer.from_metadata(dict(checkpoint["normalizer"]))
    return {year: normalizer.transform(table) for year, table in observed_tables.items()}


def _load_generated_sources(paths: list[str | Path]) -> dict[str, dict[int, pd.DataFrame]]:
    outputs: dict[str, dict[int, pd.DataFrame]] = {}
    for path in paths:
        manifest: GeneratedSeriesManifest = load_generated_series_manifest(path)
        outputs[manifest.series_name] = load_generated_tables(manifest)
    return outputs


def _shared_seq_columns(tables: dict[int, pd.DataFrame], years: tuple[int, ...]) -> list[str]:
    ordered: list[str] | None = None
    for year in years:
        table = tables.get(int(year))
        if table is None:
            continue
        current = seq_columns(table.columns)
        if ordered is None:
            ordered = list(current)
        else:
            current_set = set(current)
            ordered = [feature for feature in ordered if feature in current_set]
    if not ordered:
        raise ValueError("No shared protein columns were found for disease benchmarking.")
    return ordered


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except Exception:
        return float("nan")


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    def recall_for(label: int) -> float:
        mask = y_true == label
        if mask.sum() == 0:
            return float("nan")
        return float(np.mean(y_pred[mask] == label))

    positive_mask = np.isin(y_true, [1, 2])
    if positive_mask.sum() == 0:
        risk_recall = float("nan")
    else:
        risk_recall = float(np.mean(np.isin(y_pred[positive_mask], [1, 2])))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": _safe_auc(y_true, y_prob),
        "recall_1": recall_for(1),
        "recall_2": recall_for(2),
        "risk_recall": risk_recall,
    }


def _build_disease_frame(
    source_tables: dict[int, pd.DataFrame],
    labels: pd.DataFrame,
    *,
    disease: str,
    years: tuple[int, ...],
    feature_columns: list[str],
) -> pd.DataFrame:
    label_frame = labels.copy()
    label_frame["ID"] = label_frame["ID"].astype(str)

    frames: list[pd.DataFrame] = []
    for year in years:
        label_column = f"{disease}_{YEAR_SUFFIX[int(year)]}"
        if label_column not in label_frame.columns or int(year) not in source_tables:
            continue

        proteome = source_tables[int(year)][["ID"] + feature_columns].copy()
        proteome["ID"] = proteome["ID"].astype(str)
        proteome[feature_columns] = proteome[feature_columns].apply(pd.to_numeric, errors="coerce")
        merged = proteome.merge(label_frame[["ID", label_column]], on="ID", how="inner")
        merged = merged.rename(columns={label_column: "label"})
        merged["label"] = pd.to_numeric(merged["label"], errors="coerce")
        merged["year"] = int(year)
        merged = merged.dropna(subset=["label"])
        merged = merged[merged["label"].isin([0, 1, 2])]
        merged = merged.dropna(subset=feature_columns)
        frames.append(merged.reset_index(drop=True))

    if not frames:
        raise ValueError(f"No aligned data found for disease '{disease}'.")
    return pd.concat(frames, axis=0, ignore_index=True)


def _evaluate_model(
    model: nn.Module,
    dataset: TensorDataset,
    *,
    years: np.ndarray,
    device: str,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    model.eval()
    losses: list[float] = []
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            losses.append(float(loss_fn(logits, yb).item()))
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels.append(yb.cpu().numpy())

    y_prob = np.concatenate(probs, axis=0)
    y_true = np.concatenate(labels, axis=0)
    y_pred = y_prob.argmax(axis=1)

    overall = _classification_metrics(y_true, y_pred, y_prob)
    overall["loss"] = float(np.mean(losses)) if losses else float("nan")
    overall["sample_count"] = int(len(y_true))

    per_year: dict[str, dict[str, float]] = {}
    for year in sorted(set(int(item) for item in years.tolist())):
        mask = years == year
        if not np.any(mask):
            continue
        per_year[str(year)] = _classification_metrics(
            y_true[mask],
            y_pred[mask],
            y_prob[mask],
        )
        per_year[str(year)]["sample_count"] = int(mask.sum())

    return overall, per_year


def _train_for_disease(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    hidden_dim: int,
    dropout: float,
    epochs: int,
    lr: float,
    seed: int,
    device: str,
) -> dict[str, object]:
    ids = frame["ID"].to_numpy()
    years = frame["year"].to_numpy(dtype=np.int64)
    train_indices, val_indices = split_indices_by_id(ids, train_ratio=0.8, seed=seed)
    if len(train_indices) < 8 or len(val_indices) < 2:
        raise ValueError("Disease benchmarking requires at least 8 train rows and 2 validation rows.")

    train_x = torch.from_numpy(frame.iloc[train_indices][feature_columns].to_numpy(dtype=np.float32))
    train_y = torch.from_numpy(frame.iloc[train_indices]["label"].to_numpy(dtype=np.int64))
    val_x = torch.from_numpy(frame.iloc[val_indices][feature_columns].to_numpy(dtype=np.float32))
    val_y = torch.from_numpy(frame.iloc[val_indices]["label"].to_numpy(dtype=np.int64))

    train_dataset = TensorDataset(train_x, train_y)
    val_dataset = TensorDataset(val_x, val_y)
    train_years = years[train_indices]
    val_years = years[val_indices]

    set_global_seed(seed)
    model = MLPClassifier(
        len(feature_columns),
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_dataset, batch_size=min(128, len(train_dataset)), shuffle=True)

    for _ in range(max(1, epochs)):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

    train_metrics, train_by_year = _evaluate_model(
        model,
        train_dataset,
        years=train_years,
        device=device,
    )
    val_metrics, val_by_year = _evaluate_model(
        model,
        val_dataset,
        years=val_years,
        device=device,
    )
    return {
        "train": train_metrics,
        "train_by_year": train_by_year,
        "val": val_metrics,
        "val_by_year": val_by_year,
    }


def benchmark_disease_models(
    observed_tables: dict[int, pd.DataFrame],
    *,
    clinical_labels: pd.DataFrame,
    diseases: list[str],
    years: tuple[int, ...],
    epochs: int,
    hidden_dim: int,
    dropout: float,
    lr: float,
    seed: int,
    device: str,
    output_dir: str | Path,
    generated_manifest_paths: list[str | Path] | None = None,
    checkpoint_path: str | Path | None = None,
) -> DiseaseBenchmarkResult:
    normalized_observed = _normalize_observed_tables(
        observed_tables,
        checkpoint_path=checkpoint_path,
    )
    sources: dict[str, dict[int, pd.DataFrame]] = {"observed": normalized_observed}
    sources.update(_load_generated_sources(generated_manifest_paths or []))

    summary: dict[str, object] = {"years": [int(year) for year in years], "sources": {}}
    for source_name, tables in sources.items():
        feature_columns = _shared_seq_columns(tables, years)
        disease_summary: dict[str, object] = {}
        for disease in diseases:
            frame = _build_disease_frame(
                tables,
                clinical_labels,
                disease=disease,
                years=years,
                feature_columns=feature_columns,
            )
            disease_summary[disease] = _train_for_disease(
                frame,
                feature_columns=feature_columns,
                hidden_dim=hidden_dim,
                dropout=dropout,
                epochs=epochs,
                lr=lr,
                seed=seed,
                device=device,
            )
        summary["sources"][source_name] = disease_summary

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / "disease_benchmark.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return DiseaseBenchmarkResult(summary=summary, summary_path=summary_path)
