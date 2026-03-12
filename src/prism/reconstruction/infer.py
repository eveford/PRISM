from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from prism.io.loaders import align_pair_tables
from prism.preprocessing.normalize import ProteomeNormalizer
from prism.reconstruction.model import PrismReconstructionModel
from prism.reconstruction.train import apply_mode_to_batch


def load_checkpoint(checkpoint_path: str | Path, *, device: str) -> dict[str, object]:
    payload = torch.load(Path(checkpoint_path), map_location=device)
    return payload


def _build_model_from_checkpoint(payload: dict[str, object], *, device: str) -> PrismReconstructionModel:
    model_config = dict(payload["model_config"])
    model = PrismReconstructionModel(
        baseline_dim=int(model_config["baseline_dim"]),
        key_dim=int(model_config["key_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        depth=int(model_config["depth"]),
        dropout=float(model_config["dropout"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def reconstruct_pairs(
    tables: dict[int, pd.DataFrame],
    pairs: list[tuple[int, int]],
    *,
    checkpoint_path: str | Path,
    mode: str | None = None,
    device: str = "cpu",
) -> tuple[dict[str, pd.DataFrame], dict[int, pd.DataFrame]]:
    payload = load_checkpoint(checkpoint_path, device=device)
    checkpoint_mode = str(payload["mode"])
    active_mode = mode or checkpoint_mode
    feature_columns = [str(item) for item in payload["feature_columns"]]
    key_features = [str(item) for item in payload["key_features"]]
    normalizer = ProteomeNormalizer.from_metadata(dict(payload["normalizer"]))
    normalized_tables = {year: normalizer.transform(table) for year, table in tables.items()}
    model = _build_model_from_checkpoint(payload, device=device)

    by_pair: dict[str, pd.DataFrame] = {}
    by_year: dict[int, pd.DataFrame] = {}

    for baseline_year, target_year in pairs:
        baseline_df, target_df = align_pair_tables(
            baseline_df=normalized_tables[baseline_year],
            target_df=normalized_tables[target_year],
            feature_columns=feature_columns,
        )
        baseline_matrix = torch.from_numpy(
            baseline_df[feature_columns].to_numpy(dtype=np.float32)
        ).to(device)
        key_matrix = torch.from_numpy(
            target_df[key_features].to_numpy(dtype=np.float32)
        ).to(device)
        baseline_matrix, key_matrix = apply_mode_to_batch(
            baseline_matrix,
            key_matrix,
            mode=active_mode,
        )

        with torch.no_grad():
            prediction = model(baseline_matrix, key_matrix).cpu().numpy()

        reconstructed = pd.DataFrame(prediction, columns=feature_columns)
        reconstructed.insert(0, "Year", target_df["Year"].astype(int).tolist())
        reconstructed.insert(0, "age", pd.to_numeric(target_df["age"], errors="coerce").tolist())
        reconstructed.insert(0, "ID", target_df["ID"].astype(str).tolist())

        pair_label = f"{baseline_year}_{target_year}"
        by_pair[pair_label] = reconstructed
        by_year[int(target_year)] = reconstructed

    return by_pair, by_year


def save_reconstructed_tables(
    by_pair: dict[str, pd.DataFrame],
    *,
    output_dir: str | Path,
    series_name: str,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    pair_files: dict[str, str] = {}
    year_files: dict[int, str] = {}

    for pair_label, df in by_pair.items():
        file_name = f"{series_name}_{pair_label}.csv"
        df.to_csv(output_path / file_name, index=False)
        pair_files[pair_label] = file_name
        _, target_year = pair_label.split("_")
        year_files[int(target_year)] = file_name

    manifest_path = output_path / f"{series_name}_generated_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "series_name": series_name,
                "pair_files": pair_files,
                "year_files": {str(year): file_name for year, file_name in year_files.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path
