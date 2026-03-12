from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import json
import pandas as pd

from prism.evaluation.age_lasso import benchmark_age_models
from prism.evaluation.disease_risk import benchmark_disease_models
from prism.interpretation.annotations import build_annotation_summary


def _write_generated_manifest(root: Path, *, series_name: str, tables: dict[int, pd.DataFrame]) -> Path:
    output_dir = root / series_name
    output_dir.mkdir(parents=True, exist_ok=True)
    year_files: dict[str, str] = {}
    for year, frame in tables.items():
        file_name = f"{series_name}_{year}.csv"
        frame.to_csv(output_dir / file_name, index=False)
        year_files[str(year)] = file_name
    manifest_path = output_dir / f"{series_name}_generated_manifest.json"
    manifest_path.write_text(
        json.dumps({"series_name": series_name, "pair_files": {}, "year_files": year_files}, indent=2),
        encoding="utf-8",
    )
    return manifest_path


def test_age_benchmark_runs_with_generated_manifest(tmp_path, observed_tables_eval) -> None:
    generated_tables = {}
    for year, frame in observed_tables_eval.items():
        generated = frame.copy()
        seq_columns = [column for column in generated.columns if str(column).startswith("seq")]
        generated.loc[:, seq_columns] = generated.loc[:, seq_columns] * 0.98
        generated_tables[year] = generated

    manifest_path = _write_generated_manifest(tmp_path, series_name="prism", tables=generated_tables)
    result = benchmark_age_models(
        observed_tables_eval,
        years=(2007, 2012, 2020),
        alpha_grid=(1.0e-4, 1.0e-3, 1.0e-2),
        output_dir=tmp_path / "age",
        seed=19,
        generated_manifest_paths=[manifest_path],
    )

    assert result.summary_path.exists()
    assert "observed" in result.summary["sources"]
    assert "prism" in result.summary["sources"]


def test_disease_benchmark_runs_on_mock_data(tmp_path, observed_tables_eval, clinical_labels) -> None:
    result = benchmark_disease_models(
        observed_tables_eval,
        clinical_labels=clinical_labels,
        diseases=["amiRecdNew", "strokeRecdNew"],
        years=(2007, 2012, 2020),
        epochs=2,
        hidden_dim=16,
        dropout=0.1,
        lr=0.01,
        seed=23,
        device="cpu",
        output_dir=tmp_path / "disease",
    )

    observed = result.summary["sources"]["observed"]
    assert "amiRecdNew" in observed
    assert "strokeRecdNew" in observed
    assert "risk_recall" in observed["amiRecdNew"]["val"]


def test_annotation_summary_and_audit() -> None:
    summary = build_annotation_summary(
        annotation_df=pd.DataFrame(
            [
                {"Protein": "seq.0_protein0", "GeneSymbol": "GENE0", "tony_organ": "Liver", "tony_enriched": True, "protein_class": "Secreted"},
                {"Protein": "seq.1_protein1", "GeneSymbol": "GENE1", "tony_organ": "Heart", "tony_enriched": False, "protein_class": "Membrane"},
            ]
        ),
        features=["seq.0_protein0", "seq.1_protein1", "seq.9_missing"],
    )
    assert summary["records"]["seq.9_missing"]["protein_class"] == "Nan"

    repo_root = Path(__file__).resolve().parents[1]
    audit_script = repo_root / "scripts" / "audit_public_repo.py"
    proc = subprocess.run(
        [sys.executable, str(audit_script), "--root", str(repo_root)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
