from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _string_value(value: object) -> str:
    if value is None or pd.isna(value):
        return "Nan"
    text = str(value).strip()
    return text if text else "Nan"


def build_annotation_summary(annotation_df: pd.DataFrame, features: list[str]) -> dict[str, object]:
    """
    REQUIRED PRIVATE DATA FILE: protein_annotation_table
    """
    if "Protein" not in annotation_df.columns:
        raise ValueError("Annotation table must contain a 'Protein' column.")

    lookup = annotation_df.copy()
    lookup["Protein"] = lookup["Protein"].astype(str)
    lookup = lookup.drop_duplicates(subset="Protein", keep="first").set_index("Protein")

    records: dict[str, dict[str, str]] = {}
    for feature in features:
        if feature in lookup.index:
            row = lookup.loc[feature]
            records[feature] = {
                "gene_symbol": _string_value(row.get("GeneSymbol", row.get("gene_symbol", row.get("gene", "Nan")))),
                "organ": _string_value(row.get("tony_organ", row.get("organ", "Nan"))),
                "tony_enriched": _string_value(row.get("tony_enriched", "Nan")),
                "protein_class": _string_value(row.get("protein_class", "Nan")),
            }
        else:
            records[feature] = {
                "gene_symbol": "Nan",
                "organ": "Nan",
                "tony_enriched": "Nan",
                "protein_class": "Nan",
            }

    class_counts: dict[str, int] = {}
    for record in records.values():
        class_name = record["protein_class"]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    return {
        "feature_count": len(features),
        "class_counts": class_counts,
        "records": records,
    }


def save_annotation_summary(summary: dict[str, object], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path
