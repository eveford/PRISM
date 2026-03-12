from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from prism.config import PaperConfig
from prism.evaluation.disease_risk import benchmark_disease_models
from prism.io.loaders import load_clinical_labels, load_disease_whitelist, load_observed_tables
from prism.io.manifest import DataManifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the disease-risk benchmark with per-disease MLP models.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "paper.yaml")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--generated-manifest", action="append", default=[])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("work/benchmark_disease"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PaperConfig.from_yaml(args.config)
    manifest = DataManifest.from_yaml(args.manifest)
    observed_tables = load_observed_tables(manifest, config.evaluation.disease_years)
    clinical_labels = load_clinical_labels(manifest)
    diseases = load_disease_whitelist(config.disease_whitelist_path)
    result = benchmark_disease_models(
        observed_tables,
        clinical_labels=clinical_labels,
        diseases=diseases,
        years=config.evaluation.disease_years,
        epochs=config.evaluation.disease_epochs,
        hidden_dim=config.evaluation.disease_hidden_dim,
        dropout=config.evaluation.disease_dropout,
        lr=config.evaluation.disease_lr,
        seed=config.reconstruction.seed,
        device=args.device,
        output_dir=args.output_dir,
        generated_manifest_paths=[Path(path) for path in args.generated_manifest],
        checkpoint_path=args.checkpoint,
    )
    print(f"Wrote {result.summary_path}")


if __name__ == "__main__":
    main()
