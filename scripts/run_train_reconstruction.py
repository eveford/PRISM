from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from prism.config import PaperConfig
from prism.io.loaders import align_feature_columns, load_feature_list, load_observed_tables
from prism.io.manifest import DataManifest
from prism.preprocessing.normalize import normalize_tables
from prism.reconstruction.dataset import build_pair_arrays
from prism.reconstruction.train import train_reconstruction_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the PRISM reconstruction model.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "paper.yaml")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--panel", type=Path, default=None)
    parser.add_argument("--mode", choices=["prism", "baseline_only", "key_only"], default="prism")
    parser.add_argument("--output-dir", type=Path, default=Path("work/reconstruction"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PaperConfig.from_yaml(args.config)
    manifest = DataManifest.from_yaml(args.manifest)
    panel_path = args.panel.resolve() if args.panel is not None else config.panel_path

    tables = load_observed_tables(manifest, config.all_years)
    feature_columns = align_feature_columns(tables)
    normalized_tables, normalizer = normalize_tables(tables, feature_columns)
    key_features = load_feature_list(panel_path)
    arrays = build_pair_arrays(
        normalized_tables,
        feature_columns,
        key_features,
        config.train_pairs,
    )
    checkpoint_path, metrics_path = train_reconstruction_model(
        arrays,
        normalizer=normalizer,
        hidden_dim=config.reconstruction.hidden_dim,
        depth=config.reconstruction.depth,
        dropout=config.reconstruction.dropout,
        batch_size=config.reconstruction.batch_size,
        epochs=config.reconstruction.epochs,
        lr=config.reconstruction.lr,
        weight_decay=config.reconstruction.weight_decay,
        cosine_weight=config.reconstruction.cosine_weight,
        train_ratio=config.reconstruction.train_ratio,
        seed=config.reconstruction.seed,
        device=config.reconstruction.device,
        output_dir=args.output_dir,
        mode=args.mode,
    )

    output_dir = args.output_dir.resolve()
    summary_path = output_dir / "training_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "mode": args.mode,
                "panel_path": str(panel_path.name),
                "checkpoint_path": checkpoint_path.name,
                "metrics_path": metrics_path.name,
                "train_pairs": config.train_pairs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
