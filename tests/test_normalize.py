from __future__ import annotations

import numpy as np

from prism.preprocessing.normalize import ProteomeNormalizer, fit_proteome_normalizer


def test_normalizer_clips_and_roundtrips_metadata(observed_tables_full) -> None:
    feature_columns = [column for column in observed_tables_full[2002].columns if str(column).startswith("seq")]
    normalizer = fit_proteome_normalizer(observed_tables_full, feature_columns)
    transformed = normalizer.transform(observed_tables_full[2002])

    matrix = transformed[feature_columns].to_numpy(dtype=float)
    assert np.isfinite(matrix).all()
    assert matrix.min() >= -3.000001
    assert matrix.max() <= 3.000001

    restored = ProteomeNormalizer.from_metadata(normalizer.to_metadata())
    restored_frame = restored.transform(observed_tables_full[2002])
    assert np.allclose(
        transformed[feature_columns].to_numpy(dtype=float),
        restored_frame[feature_columns].to_numpy(dtype=float),
    )
