from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


FEATURE_COLUMNS = [f"seq.{index}_protein{index}" for index in range(6)]


def make_observed_tables(years: tuple[int, ...], *, n_ids: int = 24) -> dict[int, pd.DataFrame]:
    rng = np.random.default_rng(42)
    ids = np.asarray([str(1000 + index) for index in range(n_ids)])
    baseline_age = 45.0 + np.arange(n_ids) * 0.8
    latent = rng.normal(size=(n_ids, len(FEATURE_COLUMNS)))

    tables: dict[int, pd.DataFrame] = {}
    for year_index, year in enumerate(years):
        age = baseline_age + year_index * 5.0
        matrix = latent.copy()
        matrix[:, 0] = age * 0.08 + rng.normal(scale=0.1, size=n_ids)
        matrix[:, 1] += year_index * 0.4
        matrix[:, 2] += np.linspace(-1.0, 1.0, n_ids) * (0.5 + 0.1 * year_index)
        matrix[:, 3] = 0.7 * matrix[:, 0] + 0.3 * matrix[:, 2] + rng.normal(scale=0.05, size=n_ids)
        matrix[:, 4] += np.sin(np.linspace(0.0, np.pi, n_ids)) * 0.2
        matrix[:, 5] += year_index * 0.2

        frame = pd.DataFrame(matrix, columns=FEATURE_COLUMNS)
        frame.insert(0, "Year", year)
        frame.insert(0, "age", age)
        frame.insert(0, "ID", ids)
        tables[int(year)] = frame
    return tables


def make_clinical_labels(ids: list[str] | np.ndarray) -> pd.DataFrame:
    ids = [str(item) for item in ids]

    def label_for(onset_year: int | None, current_year: int) -> int:
        if onset_year is None:
            return 0
        if current_year >= onset_year:
            return 2
        return 1

    records = []
    for index, person_id in enumerate(ids):
        if index % 4 == 0:
            ami_onset = 2007
        elif index % 4 == 1:
            ami_onset = 2012
        elif index % 4 == 2:
            ami_onset = 2020
        else:
            ami_onset = None

        if index % 5 == 0:
            stroke_onset = 2007
        elif index % 5 in (1, 2):
            stroke_onset = 2012
        elif index % 5 == 3:
            stroke_onset = 2020
        else:
            stroke_onset = None

        records.append(
            {
                "ID": person_id,
                "amiRecdNew_07": label_for(ami_onset, 2007),
                "amiRecdNew_12": label_for(ami_onset, 2012),
                "amiRecdNew_20": label_for(ami_onset, 2020),
                "strokeRecdNew_07": label_for(stroke_onset, 2007),
                "strokeRecdNew_12": label_for(stroke_onset, 2012),
                "strokeRecdNew_20": label_for(stroke_onset, 2020),
            }
        )
    return pd.DataFrame.from_records(records)


def write_generated_manifest(
    root: Path,
    *,
    series_name: str,
    tables: dict[int, pd.DataFrame],
) -> Path:
    output_dir = root / series_name
    output_dir.mkdir(parents=True, exist_ok=True)
    year_files: dict[str, str] = {}
    for year, frame in tables.items():
        file_name = f"{series_name}_{year}.csv"
        frame.to_csv(output_dir / file_name, index=False)
        year_files[str(year)] = file_name

    manifest_path = output_dir / f"{series_name}_generated_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "series_name": series_name,
                "pair_files": {},
                "year_files": year_files,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path


@pytest.fixture
def observed_tables_full() -> dict[int, pd.DataFrame]:
    return make_observed_tables((2002, 2007, 2012, 2020))


@pytest.fixture
def observed_tables_eval() -> dict[int, pd.DataFrame]:
    return make_observed_tables((2007, 2012, 2020))


@pytest.fixture
def clinical_labels(observed_tables_eval: dict[int, pd.DataFrame]) -> pd.DataFrame:
    ids = observed_tables_eval[2007]["ID"].tolist()
    return make_clinical_labels(ids)
