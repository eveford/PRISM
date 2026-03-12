from __future__ import annotations

import numpy as np

from prism.preprocessing.normalize import normalize_tables
from prism.reconstruction.dataset import build_pair_arrays
from prism.reconstruction.infer import reconstruct_pairs
from prism.reconstruction.train import train_reconstruction_model


def test_reconstruction_training_and_inference_modes(tmp_path, observed_tables_full) -> None:
    feature_columns = [column for column in observed_tables_full[2002].columns if str(column).startswith("seq")]
    key_features = feature_columns[:3]
    normalized_tables, normalizer = normalize_tables(observed_tables_full, feature_columns)
    arrays = build_pair_arrays(
        normalized_tables,
        feature_columns,
        key_features,
        [(2002, 2007), (2007, 2012)],
    )

    checkpoint_path, _ = train_reconstruction_model(
        arrays,
        normalizer=normalizer,
        hidden_dim=16,
        depth=1,
        dropout=0.0,
        batch_size=8,
        epochs=2,
        lr=0.01,
        weight_decay=0.0,
        cosine_weight=0.2,
        train_ratio=0.8,
        seed=7,
        device="cpu",
        output_dir=tmp_path / "reconstruction",
        mode="prism",
    )

    prism_pair, _ = reconstruct_pairs(
        observed_tables_full,
        [(2012, 2020)],
        checkpoint_path=checkpoint_path,
        mode="prism",
    )
    baseline_pair, _ = reconstruct_pairs(
        observed_tables_full,
        [(2012, 2020)],
        checkpoint_path=checkpoint_path,
        mode="baseline_only",
    )
    key_pair, _ = reconstruct_pairs(
        observed_tables_full,
        [(2012, 2020)],
        checkpoint_path=checkpoint_path,
        mode="key_only",
    )

    prism_df = prism_pair["2012_2020"]
    baseline_df = baseline_pair["2012_2020"]
    key_df = key_pair["2012_2020"]

    assert prism_df.shape[0] == observed_tables_full[2020].shape[0]
    assert prism_df.shape[1] == observed_tables_full[2020].shape[1]
    assert not np.allclose(
        prism_df[feature_columns].to_numpy(dtype=float),
        baseline_df[feature_columns].to_numpy(dtype=float),
    )
    assert not np.allclose(
        prism_df[feature_columns].to_numpy(dtype=float),
        key_df[feature_columns].to_numpy(dtype=float),
    )
