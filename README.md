# PRISM

## What this repository contains

This repository is a public implementation of the paper-core PRISM workflow:

- proteome normalization
- initial key-protein candidate selection
- iterative sparse pruning to a 64-protein panel
- PRISM reconstruction training and ablation benchmarking
- age benchmarking with LassoCV
- disease-risk benchmarking with per-disease MLP models
- a minimal protein annotation summary module

Plot-specific and exploratory scripts from the original private workspace are intentionally excluded.

## Private data requirements

All cohort tables are private and must not be added to this repository.

Prepare a local manifest from `configs/templates/data_manifest.example.yaml`. The manifest should point to your private files by logical name only. Tracked config files in this repo must stay relative-path only.

The required logical inputs are documented in `docs/data_requirements.md`.

## Installation

```bash
pip install -e .
```

## Minimal run sequence

1. Create a private manifest YAML from `configs/templates/data_manifest.example.yaml`.
2. Run panel selection:

```bash
python scripts/run_select_panel.py \
  --manifest path/to/data_manifest.yaml \
  --output-dir work/select_panel
```

3. Train the PRISM reconstruction model:

```bash
python scripts/run_train_reconstruction.py \
  --manifest path/to/data_manifest.yaml \
  --output-dir work/reconstruction
```

4. Benchmark reconstruction:

```bash
python scripts/run_benchmark_reconstruction.py \
  --manifest path/to/data_manifest.yaml \
  --checkpoint work/reconstruction/prism_checkpoint.pt \
  --output-dir work/benchmark_reconstruction
```

5. Run age benchmarking:

```bash
python scripts/run_age_benchmark.py \
  --manifest path/to/data_manifest.yaml \
  --checkpoint work/reconstruction/prism_checkpoint.pt \
  --generated-manifest work/benchmark_reconstruction/prism/prism_generated_manifest.json \
  --output-dir work/benchmark_age
```

6. Run disease-risk benchmarking:

```bash
python scripts/run_disease_benchmark.py \
  --manifest path/to/data_manifest.yaml \
  --checkpoint work/reconstruction/prism_checkpoint.pt \
  --generated-manifest work/benchmark_reconstruction/prism/prism_generated_manifest.json \
  --output-dir work/benchmark_disease
```
