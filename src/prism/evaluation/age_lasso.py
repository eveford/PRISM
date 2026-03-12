from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch

from prism.io.loaders import (
    GeneratedSeriesManifest,
    load_generated_series_manifest,
    load_generated_tables,
    seq_columns,
)
from prism.preprocessing.normalize import ProteomeNormalizer, normalize_tables
from prism.reconstruction.dataset import split_indices_by_id


@dataclass(frozen=True)
class AgeBenchmarkResult:
    summary: dict[str, object]
    summary_path: Path


def _safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return float("nan")
    if np.allclose(y_true, y_true[0]) or np.allclose(y_pred, y_pred[0]):
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


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
        raise ValueError("No shared protein columns were found for age benchmarking.")
    return ordered


def _prepare_year_frame(
    table: pd.DataFrame,
    *,
    year: int,
    feature_columns: list[str],
) -> pd.DataFrame:
    required = {"ID", "age"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Year {year} table is missing required columns: {sorted(missing)}")

    frame = table[["ID", "age"] + feature_columns].copy()
    frame["ID"] = frame["ID"].astype(str)
    frame["age"] = pd.to_numeric(frame["age"], errors="coerce")
    frame[feature_columns] = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=["age"])
    frame = frame.dropna(subset=feature_columns)
    frame = frame.drop_duplicates(subset="ID", keep="first")
    return frame.reset_index(drop=True)


def _fit_lasso_for_year(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    alpha_grid: tuple[float, ...],
    seed: int,
) -> dict[str, float]:
    ids = frame["ID"].to_numpy()
    train_indices, val_indices = split_indices_by_id(ids, train_ratio=0.8, seed=seed)
    if len(train_indices) < 5 or len(val_indices) < 2:
        raise ValueError("Age benchmarking requires at least 5 train samples and 2 validation samples.")

    x_train = frame.iloc[train_indices][feature_columns].to_numpy(dtype=np.float32)
    y_train = frame.iloc[train_indices]["age"].to_numpy(dtype=np.float32)
    x_val = frame.iloc[val_indices][feature_columns].to_numpy(dtype=np.float32)
    y_val = frame.iloc[val_indices]["age"].to_numpy(dtype=np.float32)

    cv = min(5, len(train_indices))
    model = LassoCV(
        alphas=list(alpha_grid),
        cv=cv,
        random_state=seed,
        max_iter=5000,
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)

    return {
        "alpha": float(model.alpha_),
        "n_train": int(len(train_indices)),
        "n_val": int(len(val_indices)),
        "mae": float(mean_absolute_error(y_val, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_val, y_pred))),
        "pearson_r": _safe_pearson(y_val, y_pred),
        "r2": float(r2_score(y_val, y_pred)),
    }


def benchmark_age_models(
    observed_tables: dict[int, pd.DataFrame],
    *,
    years: tuple[int, ...],
    alpha_grid: tuple[float, ...],
    output_dir: str | Path,
    seed: int,
    generated_manifest_paths: list[str | Path] | None = None,
    checkpoint_path: str | Path | None = None,
) -> AgeBenchmarkResult:
    normalized_observed = _normalize_observed_tables(
        observed_tables,
        checkpoint_path=checkpoint_path,
    )
    sources: dict[str, dict[int, pd.DataFrame]] = {"observed": normalized_observed}
    sources.update(_load_generated_sources(generated_manifest_paths or []))

    summary: dict[str, object] = {"years": [int(year) for year in years], "sources": {}}
    for source_name, tables in sources.items():
        feature_columns = _shared_seq_columns(tables, years)
        year_summary: dict[str, dict[str, float]] = {}
        for year in years:
            if int(year) not in tables:
                continue
            frame = _prepare_year_frame(
                tables[int(year)],
                year=int(year),
                feature_columns=feature_columns,
            )
            year_summary[str(int(year))] = _fit_lasso_for_year(
                frame,
                feature_columns=feature_columns,
                alpha_grid=alpha_grid,
                seed=seed,
            )
        summary["sources"][source_name] = year_summary

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / "age_benchmark.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return AgeBenchmarkResult(summary=summary, summary_path=summary_path)
