from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from prism.config import PaperConfig
from prism.io.loaders import align_feature_columns, load_observed_tables
from prism.io.manifest import DataManifest
from prism.preprocessing.normalize import normalize_tables
from prism.selection.init_panel import initialize_panel, save_panel
from prism.selection.sparse_prune import iterative_sparse_prune, save_sparse_prune_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the paper-core PRISM panel selection pipeline.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "paper.yaml")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("work/select_panel"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PaperConfig.from_yaml(args.config)
    manifest = DataManifest.from_yaml(args.manifest)
    tables = load_observed_tables(manifest, config.all_years)
    feature_columns = align_feature_columns(tables)
    normalized_tables, _ = normalize_tables(tables, feature_columns)

    initial_result = initialize_panel(
        normalized_tables,
        feature_columns,
        initial_panel_size=config.selection.initial_panel_size,
        candidate_pool_size=config.selection.candidate_pool_size,
        max_pairwise_correlation=config.selection.max_pairwise_correlation,
    )
    output_dir = args.output_dir.resolve()
    initial_path = save_panel(initial_result, output_dir / "initial_panel.json")

    prune_result = iterative_sparse_prune(
        normalized_tables,
        feature_columns,
        config.train_pairs,
        initial_result.features,
        target_panel_size=config.selection.target_panel_size,
        prune_step=config.selection.prune_step,
        hidden_dim=config.reconstruction.hidden_dim,
        depth=config.reconstruction.depth,
        dropout=config.reconstruction.dropout,
        epochs_per_round=config.selection.epochs_per_round,
        batch_size=config.reconstruction.batch_size,
        lr=config.reconstruction.lr,
        weight_decay=config.reconstruction.weight_decay,
        cosine_weight=config.reconstruction.cosine_weight,
        seed=config.reconstruction.seed,
        device=config.reconstruction.device,
    )
    final_path = save_sparse_prune_result(
        prune_result,
        output_dir,
        filename="prism_panel_64.json",
    )

    summary = {
        "initial_panel_path": str(initial_path.relative_to(output_dir)),
        "final_panel_path": str(final_path.relative_to(output_dir)),
        "initial_panel_size": len(initial_result.features),
        "final_panel_size": len(prune_result.features),
    }
    summary_path = output_dir / "selection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
