from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from prism.io.loaders import align_pair_tables
from prism.preprocessing.normalize import ProteomeNormalizer, normalize_tables
from prism.reconstruction.infer import load_checkpoint, reconstruct_pairs, save_reconstructed_tables


def _baseline_carry_forward(
    normalized_tables: dict[int, pd.DataFrame],
    *,
    pairs: list[tuple[int, int]],
    feature_columns: list[str],
) -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}
    for baseline_year, target_year in pairs:
        baseline_df, target_df = align_pair_tables(
            baseline_df=normalized_tables[baseline_year],
            target_df=normalized_tables[target_year],
            feature_columns=feature_columns,
        )
        carry = baseline_df[["ID", "age", "Year"] + feature_columns].copy()
        carry["age"] = pd.to_numeric(target_df["age"], errors="coerce").tolist()
        carry["Year"] = pd.to_numeric(target_df["Year"], errors="coerce").astype(int).tolist()
        outputs[f"{baseline_year}_{target_year}"] = carry
    return outputs


def _pair_metrics(prediction_df: pd.DataFrame, target_df: pd.DataFrame, feature_columns: list[str]) -> dict[str, float]:
    prediction = prediction_df[feature_columns].to_numpy(dtype=np.float32)
    target = target_df[feature_columns].to_numpy(dtype=np.float32)
    diff = prediction - target
    mse = np.mean(diff ** 2, axis=1)
    cosine = np.sum(prediction * target, axis=1) / (
        np.linalg.norm(prediction, axis=1) * np.linalg.norm(target, axis=1) + 1e-8
    )
    per_feature_corr = []
    for feature_index in range(len(feature_columns)):
        corr = np.corrcoef(prediction[:, feature_index], target[:, feature_index])[0, 1]
        per_feature_corr.append(corr)
    return {
        "pair_mse_mean": float(np.nanmean(mse)),
        "pair_cosine_mean": float(np.nanmean(cosine)),
        "feature_pearson_mean": float(np.nanmean(per_feature_corr)),
        "feature_pearson_median": float(np.nanmedian(per_feature_corr)),
    }


def _normalize_with_checkpoint(
    tables: dict[int, pd.DataFrame],
    checkpoint_paths: dict[str, str | Path | None],
    feature_columns: list[str],
) -> dict[int, pd.DataFrame]:
    checkpoint_path = next(
        (path for path in checkpoint_paths.values() if path is not None),
        None,
    )
    if checkpoint_path is None:
        normalized_tables, _ = normalize_tables(tables, feature_columns)
        return normalized_tables

    payload = load_checkpoint(checkpoint_path, device="cpu")
    normalizer = ProteomeNormalizer.from_metadata(dict(payload["normalizer"]))
    return {year: normalizer.transform(table) for year, table in tables.items()}


def benchmark_reconstruction_modes(
    tables: dict[int, pd.DataFrame],
    *,
    pairs: list[tuple[int, int]],
    checkpoint_paths: dict[str, str | Path | None],
    output_dir: str | Path,
) -> dict[str, object]:
    feature_columns = [column for column in tables[min(tables.keys())].columns if str(column).startswith("seq")]
    normalized_tables = _normalize_with_checkpoint(tables, checkpoint_paths, feature_columns)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {"pairs": {}, "generated_manifests": {}}

    for mode_name, checkpoint_path in checkpoint_paths.items():
        if checkpoint_path is None and mode_name != "baseline_only":
            continue

        if mode_name == "baseline_only" and checkpoint_path is None:
            by_pair = _baseline_carry_forward(
                normalized_tables,
                pairs=pairs,
                feature_columns=feature_columns,
            )
        else:
            by_pair, _ = reconstruct_pairs(
                tables=tables,
                pairs=pairs,
                checkpoint_path=checkpoint_path,
                mode=mode_name,
            )

        mode_output_dir = output_path / mode_name
        manifest_path = save_reconstructed_tables(
            by_pair,
            output_dir=mode_output_dir,
            series_name=mode_name,
        )
        summary["generated_manifests"][mode_name] = str(manifest_path.relative_to(output_path))

        for baseline_year, target_year in pairs:
            pair_label = f"{baseline_year}_{target_year}"
            _, target_df = align_pair_tables(
                baseline_df=normalized_tables[baseline_year],
                target_df=normalized_tables[target_year],
                feature_columns=feature_columns,
            )
            pair_summary = _pair_metrics(by_pair[pair_label], target_df, feature_columns)
            summary["pairs"].setdefault(pair_label, {})[mode_name] = pair_summary

    summary_path = output_path / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
