from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProteomeNormalizer:
    feature_columns: list[str]
    shifts: dict[str, float]
    means: dict[str, float]
    stds: dict[str, float]
    clip_low: float = -3.0
    clip_high: float = 3.0

    def transform(self, table: pd.DataFrame) -> pd.DataFrame:
        missing = [feature for feature in self.feature_columns if feature not in table.columns]
        if missing:
            raise ValueError(f"Table is missing normalized feature columns: {missing[:5]}")

        out = table.copy()
        matrix = out[self.feature_columns].apply(pd.to_numeric, errors="coerce")
        for feature in self.feature_columns:
            shift = self.shifts[feature]
            matrix[feature] = np.log1p(matrix[feature] + shift)
            matrix[feature] = (matrix[feature] - self.means[feature]) / self.stds[feature]
        matrix = matrix.clip(self.clip_low, self.clip_high)
        out[self.feature_columns] = matrix
        return out

    def to_metadata(self) -> dict[str, object]:
        return {
            "feature_columns": list(self.feature_columns),
            "shifts": dict(self.shifts),
            "means": dict(self.means),
            "stds": dict(self.stds),
            "clip_low": self.clip_low,
            "clip_high": self.clip_high,
        }

    @classmethod
    def from_metadata(cls, payload: dict[str, object]) -> "ProteomeNormalizer":
        return cls(
            feature_columns=[str(item) for item in payload["feature_columns"]],
            shifts={str(key): float(value) for key, value in dict(payload["shifts"]).items()},
            means={str(key): float(value) for key, value in dict(payload["means"]).items()},
            stds={str(key): float(value) for key, value in dict(payload["stds"]).items()},
            clip_low=float(payload.get("clip_low", -3.0)),
            clip_high=float(payload.get("clip_high", 3.0)),
        )


def fit_proteome_normalizer(
    tables: dict[int, pd.DataFrame],
    feature_columns: list[str],
    clip_low: float = -3.0,
    clip_high: float = 3.0,
) -> ProteomeNormalizer:
    pooled = pd.concat([table[feature_columns] for table in tables.values()], axis=0, ignore_index=True)
    pooled = pooled.apply(pd.to_numeric, errors="coerce")

    shifts: dict[str, float] = {}
    means: dict[str, float] = {}
    stds: dict[str, float] = {}

    for feature in feature_columns:
        series = pooled[feature]
        feature_min = float(series.min(skipna=True))
        shift = max(0.0, -feature_min)
        transformed = np.log1p(series + shift)
        mean = float(transformed.mean(skipna=True))
        std = float(transformed.std(skipna=True, ddof=0))
        shifts[feature] = shift
        means[feature] = mean
        stds[feature] = std if std > 0 else 1.0

    return ProteomeNormalizer(
        feature_columns=feature_columns,
        shifts=shifts,
        means=means,
        stds=stds,
        clip_low=clip_low,
        clip_high=clip_high,
    )


def normalize_tables(
    tables: dict[int, pd.DataFrame],
    feature_columns: list[str],
    clip_low: float = -3.0,
    clip_high: float = 3.0,
) -> tuple[dict[int, pd.DataFrame], ProteomeNormalizer]:
    normalizer = fit_proteome_normalizer(
        tables=tables,
        feature_columns=feature_columns,
        clip_low=clip_low,
        clip_high=clip_high,
    )
    return {year: normalizer.transform(table) for year, table in tables.items()}, normalizer
