# Private Data Requirements

This public repository does not ship any cohort tables.

The pipeline expects a manifest file with logical names for the private inputs:

- `observed_proteome_2002`
- `observed_proteome_2007`
- `observed_proteome_2012`
- `observed_proteome_2020`
- `clinical_label_table`
- `protein_annotation_table`

Expected minimum schema:

- Observed proteome tables: `ID`, `age`, and protein columns starting with `seq`
- Clinical label table: `ID` and disease columns named as `<disease>_<suffix>`
- Protein annotation table: `Protein`; optional columns include `GeneSymbol`, `tony_organ`, `tony_enriched`, and `protein_class`

Notes:

- The manuscript displays the last visit as `2022`, while the private file suffix stays `2020` because the serial sampling window ran from 2020 to 2023.
- `configs/templates/paper_diseases.example.json` is an example whitelist. Replace it with the exact disease labels used in your private clinical table if the names differ.
- When a script requires a private input that is not distributed here, the code includes an English note in the relevant loader or module.
