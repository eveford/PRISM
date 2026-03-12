from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from prism.io.loaders import align_pair_tables


@dataclass(frozen=True)
class PairArrays:
    baseline: np.ndarray
    key: np.ndarray
    target: np.ndarray
    ids: np.ndarray
    pair_labels: np.ndarray
    target_years: np.ndarray
    feature_columns: list[str]
    key_features: list[str]


def _split_from_id(person_id: str, *, seed: int, train_ratio: float) -> str:
    token = f"{person_id}-{seed}".encode("utf-8")
    hash_value = hashlib.md5(token).hexdigest()
    scalar = int(hash_value[:8], 16) / 0xFFFFFFFF
    return "train" if scalar < train_ratio else "val"


def split_indices_by_id(ids: np.ndarray, *, train_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    train_indices: list[int] = []
    val_indices: list[int] = []
    for index, person_id in enumerate(ids):
        split_name = _split_from_id(str(person_id), seed=seed, train_ratio=train_ratio)
        if split_name == "train":
            train_indices.append(index)
        else:
            val_indices.append(index)
    return np.asarray(train_indices, dtype=np.int64), np.asarray(val_indices, dtype=np.int64)


def build_pair_arrays(
    tables: dict[int, pd.DataFrame],
    feature_columns: list[str],
    key_features: list[str],
    pairs: list[tuple[int, int]],
) -> PairArrays:
    baseline_parts: list[np.ndarray] = []
    key_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    ids: list[str] = []
    pair_labels: list[str] = []
    target_years: list[int] = []

    missing_keys = [feature for feature in key_features if feature not in feature_columns]
    if missing_keys:
        raise ValueError(f"Key features are not present in the feature space: {missing_keys[:5]}")

    for baseline_year, target_year in pairs:
        baseline_df, target_df = align_pair_tables(
            baseline_df=tables[baseline_year],
            target_df=tables[target_year],
            feature_columns=feature_columns,
        )
        baseline_parts.append(
            baseline_df[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        )
        key_parts.append(
            target_df[key_features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        )
        target_parts.append(
            target_df[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        )
        ids.extend(target_df["ID"].astype(str).tolist())
        pair_labels.extend([f"{baseline_year}_{target_year}"] * len(target_df))
        target_years.extend([int(target_year)] * len(target_df))

    if not baseline_parts:
        raise ValueError("No pair arrays could be built from the requested data.")

    return PairArrays(
        baseline=np.concatenate(baseline_parts, axis=0),
        key=np.concatenate(key_parts, axis=0),
        target=np.concatenate(target_parts, axis=0),
        ids=np.asarray(ids),
        pair_labels=np.asarray(pair_labels),
        target_years=np.asarray(target_years),
        feature_columns=list(feature_columns),
        key_features=list(key_features),
    )


class ReconstructionDataset(Dataset):
    def __init__(self, arrays: PairArrays) -> None:
        self.arrays = arrays

    @classmethod
    def from_indices(cls, arrays: PairArrays, indices: np.ndarray) -> "ReconstructionDataset":
        subset = PairArrays(
            baseline=arrays.baseline[indices],
            key=arrays.key[indices],
            target=arrays.target[indices],
            ids=arrays.ids[indices],
            pair_labels=arrays.pair_labels[indices],
            target_years=arrays.target_years[indices],
            feature_columns=arrays.feature_columns,
            key_features=arrays.key_features,
        )
        return cls(subset)

    def __len__(self) -> int:
        return self.arrays.baseline.shape[0]

    def __getitem__(self, index: int):
        metadata = {
            "id": str(self.arrays.ids[index]),
            "pair": str(self.arrays.pair_labels[index]),
            "target_year": int(self.arrays.target_years[index]),
        }
        return (
            torch.from_numpy(self.arrays.baseline[index]),
            torch.from_numpy(self.arrays.key[index]),
            torch.from_numpy(self.arrays.target[index]),
            metadata,
        )
