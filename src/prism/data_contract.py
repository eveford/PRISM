from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RequiredFileSpec:
    logical_name: str
    description: str
    required_columns: tuple[str, ...]
    notes: str = ""


def observed_proteome_key(year: int) -> str:
    return f"observed_proteome_{year}"


REQUIRED_PRIVATE_FILES: tuple[RequiredFileSpec, ...] = (
    RequiredFileSpec(
        logical_name=observed_proteome_key(2002),
        description="Observed proteome table for visit year 2002.",
        required_columns=("ID", "age"),
        notes="The table must also contain protein columns starting with 'seq'.",
    ),
    RequiredFileSpec(
        logical_name=observed_proteome_key(2007),
        description="Observed proteome table for visit year 2007.",
        required_columns=("ID", "age"),
        notes="The table must also contain protein columns starting with 'seq'.",
    ),
    RequiredFileSpec(
        logical_name=observed_proteome_key(2012),
        description="Observed proteome table for visit year 2012.",
        required_columns=("ID", "age"),
        notes="The table must also contain protein columns starting with 'seq'.",
    ),
    RequiredFileSpec(
        logical_name=observed_proteome_key(2020),
        description="Observed proteome table for visit year 2020 (displayed as 2022 in the manuscript).",
        required_columns=("ID", "age"),
        notes="The table must also contain protein columns starting with 'seq'.",
    ),
    RequiredFileSpec(
        logical_name="clinical_label_table",
        description="Clinical label table for disease-risk benchmarking.",
        required_columns=("ID",),
        notes="Expected disease label columns follow the pattern <disease>_<suffix>.",
    ),
    RequiredFileSpec(
        logical_name="protein_annotation_table",
        description="Protein annotation table used for minimal interpretation output.",
        required_columns=("Protein",),
        notes="Expected annotation columns may include tony_organ, tony_enriched, and protein_class.",
    ),
)
