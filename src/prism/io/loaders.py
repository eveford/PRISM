from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from prism.data_contract import observed_proteome_key
from prism.io.manifest import DataManifest


def seq_columns(columns: Iterable[str]) -> list[str]:
    return [column for column in columns if str(column).startswith("seq")]


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_feature_list(path: str | Path) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}.")
    return [str(item) for item in payload]


def load_disease_whitelist(path: str | Path) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}.")
    return [str(item) for item in payload]


def _validate_proteome_table(df: pd.DataFrame, logical_name: str) -> None:
    required_columns = {"ID", "age"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"{logical_name} is missing required columns: {sorted(missing_columns)}")
    if not seq_columns(df.columns):
        raise ValueError(f"{logical_name} has no protein columns starting with 'seq'.")


def load_visit_table(manifest: DataManifest, year: int) -> pd.DataFrame:
    """
    REQUIRED PRIVATE DATA FILE: observed_proteome_<YEAR>
    """
    logical_name = observed_proteome_key(year)
    path = manifest.resolve(logical_name)
    df = read_table(path)
    _validate_proteome_table(df, logical_name)
    if "Year" not in df.columns:
        df = df.copy()
        df["Year"] = int(year)
    return df


def load_observed_tables(manifest: DataManifest, years: Iterable[int]) -> dict[int, pd.DataFrame]:
    return {int(year): load_visit_table(manifest, int(year)) for year in years}


def load_clinical_labels(manifest: DataManifest) -> pd.DataFrame:
    """
    REQUIRED PRIVATE DATA FILE: clinical_label_table
    """
    logical_name = "clinical_label_table"
    df = read_table(manifest.resolve(logical_name))
    if "ID" not in df.columns:
        raise ValueError(f"{logical_name} must contain an 'ID' column.")
    return df


def load_annotation_table(manifest: DataManifest) -> pd.DataFrame:
    """
    REQUIRED PRIVATE DATA FILE: protein_annotation_table
    """
    logical_name = "protein_annotation_table"
    df = read_table(manifest.resolve(logical_name))
    if "Protein" not in df.columns:
        raise ValueError(f"{logical_name} must contain a 'Protein' column.")
    return df


def align_feature_columns(tables: dict[int, pd.DataFrame]) -> list[str]:
    ordered: list[str] | None = None
    for df in tables.values():
        current = seq_columns(df.columns)
        if ordered is None:
            ordered = list(current)
        else:
            current_set = set(current)
            ordered = [feature for feature in ordered if feature in current_set]
    if not ordered:
        raise ValueError("No overlapping protein columns were found across the requested tables.")
    return ordered


def align_pair_tables(
    baseline_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = baseline_df.copy()
    right = target_df.copy()
    left["ID"] = left["ID"].astype(str)
    right["ID"] = right["ID"].astype(str)
    common_ids = sorted(set(left["ID"]).intersection(set(right["ID"])))
    if not common_ids:
        raise ValueError("No overlapping IDs were found between the requested time points.")
    left = left[left["ID"].isin(common_ids)].sort_values("ID")
    right = right[right["ID"].isin(common_ids)].sort_values("ID")
    columns = ["ID", "age", "Year"] + feature_columns
    return left[columns].reset_index(drop=True), right[columns].reset_index(drop=True)


@dataclass(frozen=True)
class GeneratedSeriesManifest:
    series_name: str
    pair_files: dict[str, Path]
    year_files: dict[int, Path]


def load_generated_series_manifest(path: str | Path) -> GeneratedSeriesManifest:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    pair_files = {
        str(pair): (manifest_path.parent / Path(rel_path)).resolve()
        for pair, rel_path in payload.get("pair_files", {}).items()
    }
    year_files = {
        int(year): (manifest_path.parent / Path(rel_path)).resolve()
        for year, rel_path in payload.get("year_files", {}).items()
    }
    return GeneratedSeriesManifest(
        series_name=str(payload.get("series_name", "generated")),
        pair_files=pair_files,
        year_files=year_files,
    )


def load_generated_tables(manifest: GeneratedSeriesManifest) -> dict[int, pd.DataFrame]:
    tables: dict[int, pd.DataFrame] = {}
    for year, path in manifest.year_files.items():
        df = read_table(path)
        _validate_proteome_table(df, f"generated:{manifest.series_name}:{year}")
        if "Year" not in df.columns:
            df = df.copy()
            df["Year"] = int(year)
        tables[int(year)] = df
    return tables
