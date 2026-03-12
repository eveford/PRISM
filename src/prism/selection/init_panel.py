from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PanelInitializationResult:
    features: list[str]
    summary: dict[str, object]


def _pooled_matrix(tables: dict[int, pd.DataFrame], feature_columns: list[str]) -> np.ndarray:
    pooled = pd.concat([table[feature_columns] for table in tables.values()], axis=0, ignore_index=True)
    return pooled.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)


def initialize_panel(
    tables: dict[int, pd.DataFrame],
    feature_columns: list[str],
    initial_panel_size: int,
    candidate_pool_size: int,
    max_pairwise_correlation: float,
) -> PanelInitializationResult:
    """
    Manuscript-consistent approximation:
    - rank proteins by pooled variance,
    - build a reduced candidate pool,
    - greedily select variance-dominant proteins with low redundancy.
    """
    matrix = _pooled_matrix(tables, feature_columns)
    variances = np.nanvar(matrix, axis=0)
    ranked_indices = np.argsort(-variances)
    pool_size = min(int(candidate_pool_size), len(ranked_indices))
    candidate_indices = ranked_indices[:pool_size]
    candidate_features = [feature_columns[idx] for idx in candidate_indices]
    candidate_matrix = np.nan_to_num(matrix[:, candidate_indices], nan=0.0)

    corr = np.corrcoef(candidate_matrix, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    abs_corr = np.abs(corr)

    selected_local: list[int] = []
    for local_index in range(len(candidate_features)):
        if len(selected_local) >= initial_panel_size:
            break
        if not selected_local:
            selected_local.append(local_index)
            continue
        max_corr = float(abs_corr[local_index, selected_local].max())
        if max_corr <= max_pairwise_correlation:
            selected_local.append(local_index)

    if len(selected_local) < initial_panel_size:
        remaining = [idx for idx in range(len(candidate_features)) if idx not in selected_local]
        redundancy = []
        for local_index in remaining:
            mean_abs_corr = float(abs_corr[local_index, selected_local].mean()) if selected_local else 0.0
            redundancy.append((mean_abs_corr, local_index))
        redundancy.sort(key=lambda item: item[0])
        for _, local_index in redundancy:
            if len(selected_local) >= initial_panel_size:
                break
            selected_local.append(local_index)

    selected_features = [candidate_features[idx] for idx in selected_local[:initial_panel_size]]
    summary = {
        "method_note": "manuscript-consistent approximation",
        "initial_panel_size": len(selected_features),
        "candidate_pool_size": pool_size,
        "max_pairwise_correlation": max_pairwise_correlation,
    }
    return PanelInitializationResult(features=selected_features, summary=summary)


def save_panel(result: PanelInitializationResult, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.features, indent=2), encoding="utf-8")
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(result.summary, indent=2), encoding="utf-8")
    return output_path
