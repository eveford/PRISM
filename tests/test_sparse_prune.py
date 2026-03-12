from __future__ import annotations

from prism.preprocessing.normalize import normalize_tables
from prism.selection.sparse_prune import iterative_sparse_prune


def test_sparse_prune_reduces_feature_count(observed_tables_full) -> None:
    feature_columns = [column for column in observed_tables_full[2002].columns if str(column).startswith("seq")]
    normalized_tables, _ = normalize_tables(observed_tables_full, feature_columns)

    result = iterative_sparse_prune(
        normalized_tables,
        feature_columns,
        [(2002, 2007)],
        feature_columns[:4],
        target_panel_size=2,
        prune_step=1,
        hidden_dim=8,
        depth=1,
        dropout=0.0,
        epochs_per_round=1,
        batch_size=8,
        lr=0.01,
        weight_decay=0.0,
        cosine_weight=0.1,
        seed=11,
        device="cpu",
    )

    assert len(result.features) == 2
    assert len(result.history) == 2
