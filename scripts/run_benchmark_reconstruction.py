from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from prism.config import PaperConfig
from prism.io.loaders import load_observed_tables
from prism.io.manifest import DataManifest
from prism.reconstruction.benchmark import benchmark_reconstruction_modes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PRISM reconstruction against observed future visits.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "paper.yaml")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("work/benchmark_reconstruction"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PaperConfig.from_yaml(args.config)
    manifest = DataManifest.from_yaml(args.manifest)
    tables = load_observed_tables(manifest, config.all_years)
    summary = benchmark_reconstruction_modes(
        tables,
        pairs=config.eval_pairs,
        checkpoint_paths={
            "baseline_only": None,
            "key_only": args.checkpoint,
            "prism": args.checkpoint,
        },
        output_dir=args.output_dir,
    )
    print(f"Wrote {Path(args.output_dir) / 'benchmark_summary.json'}")
    print(summary["pairs"])


if __name__ == "__main__":
    main()
