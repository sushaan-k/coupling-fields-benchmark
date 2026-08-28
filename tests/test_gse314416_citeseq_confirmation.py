import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from experiments import confirm_gse314416_citeseq as confirmation


def test_frozen_panels_are_cognate_and_secondary_contains_primary():
    assert len(confirmation.MARKERS) == 9
    assert len(confirmation.BROAD_MARKERS) == 24
    assert set(confirmation.MARKERS) <= set(confirmation.BROAD_MARKERS)
    assert len(set(confirmation.BROAD_RNA_FEATURES)) == 24
    assert len(set(confirmation.BROAD_ADT_FEATURES)) == 24
    confirmation._feature_reference()


def test_metadata_only_preflight_binds_roles_and_has_no_h5_access():
    payload = json.loads(confirmation.DEFAULT_PREFLIGHT.read_text())
    assert payload["status"] == "PASS"
    assert payload["role_counts"] == {
        "calibration": 12,
        "excluded": 1,
        "held": 77,
        "pilot": 20,
    }
    assert payload["access_audit"] == {
        "adt_numeric_values_read": 0,
        "gex_numeric_values_read": 0,
        "h5_files_opened": 0,
    }
    assert len(payload["markers"]) == 9
    assert len(payload["secondary_broad_markers"]) == 24
    assert len(payload["donors"]) == 110


def test_primary_configuration_grid_is_frozen_and_deduplicates_graph_zero():
    configs = confirmation._primary_configs()
    assert len(configs) == 168
    assert sum(config.graph_penalty == 0.0 for config in configs) == 24
    assert all(
        config.graph_neighbors == 1 for config in configs if config.graph_penalty == 0.0
    )


def test_signed_root_deviance_inversion_preserves_negative_coordinate():
    rows = np.asarray([184, 328])
    columns = np.asarray([256, 256])
    target = -4.024939692792063
    reconstructed = confirmation._classical_table(target, rows, columns, "deviance")
    np.testing.assert_allclose(reconstructed.sum(axis=1), rows)
    np.testing.assert_allclose(reconstructed.sum(axis=0), columns)
    assert confirmation._fractional_deviance(reconstructed) == pytest.approx(
        target, abs=1e-10
    )
    assert not np.allclose(
        reconstructed,
        np.outer(rows, columns) / confirmation.CELL_BUDGET,
    )


def test_adt_tie_break_has_exact_reproducible_margins():
    counts = np.zeros((confirmation.CELL_BUDGET, len(confirmation.MARKERS)), dtype=int)
    cells = [f"cell-{index:03d}" for index in range(confirmation.CELL_BUDGET)]
    first = confirmation._adt_states(counts, cells, "donor")
    second = confirmation._adt_states(counts, cells, "donor")
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(
        first.sum(axis=0),
        np.full(len(confirmation.MARKERS), confirmation.CELL_BUDGET // 2),
    )


def test_held_margin_constructor_does_not_require_adt_values():
    counts = np.zeros((confirmation.CELL_BUDGET, len(confirmation.MARKERS)), dtype=int)
    counts[:100, 0] = 1
    rows, columns = confirmation._held_margins(counts)
    assert rows.shape == (9, 9, 2)
    assert columns.shape == rows.shape
    np.testing.assert_array_equal(rows[0, 0], [412, 100])
    np.testing.assert_array_equal(columns, np.full_like(columns, 256))


def test_common_effect_exact_conditional_baseline_is_finite():
    table = np.asarray([[150, 106], [106, 150]], dtype=int)
    tables = np.broadcast_to(table, (3, 9, 9, 2, 2)).copy()
    fit = confirmation._fit_common_effect(tables)
    assert fit["population_log_odds"].shape == (9, 9)
    assert np.isfinite(fit["population_log_odds"]).all()
    assert fit["gradient_norm"] < 1e-6


def test_10x_reader_uses_requested_feature_and_barcode_order(tmp_path: Path):
    path = tmp_path / "matrix.h5"
    with h5py.File(path, "w") as handle:
        matrix = handle.create_group("matrix")
        matrix.create_dataset("barcodes", data=np.asarray([b"cell-a", b"cell-b"]))
        matrix.create_dataset("data", data=np.asarray([3, 5, 7], dtype=np.int32))
        matrix.create_dataset("indices", data=np.asarray([0, 2, 1], dtype=np.int32))
        matrix.create_dataset("indptr", data=np.asarray([0, 2, 3], dtype=np.int64))
        matrix.create_dataset("shape", data=np.asarray([3, 2], dtype=np.int64))
        features = matrix.create_group("features")
        features.create_dataset("id", data=np.asarray([b"f0", b"f1", b"f2"]))
    observed = confirmation._read_10x_columns(path, ["cell-b", "cell-a"], ("f2", "f0"))
    np.testing.assert_array_equal(observed, [[0, 0], [5, 3]])


def test_source_manifest_binds_public_metadata_files():
    payload = json.loads(confirmation.DEFAULT_SOURCE_MANIFEST.read_text())
    for record in payload["files"].values():
        if "path" not in record:
            continue
        path = confirmation.ROOT / record["path"]
        if path.exists():
            assert path.stat().st_size == record["bytes"]
            assert confirmation._sha256(path) == record["sha256"]


def test_protocol_requires_distinct_public_stage_tags():
    assert (
        len(
            {
                confirmation.PROTOCOL_TAG,
                confirmation.DEVELOPMENT_TAG,
                confirmation.PREDICTION_TAG,
            }
        )
        == 3
    )
    text = confirmation.DEFAULT_PROTOCOL.read_text()
    assert confirmation.PROTOCOL_TAG in text
    assert confirmation.DEVELOPMENT_TAG in text
    assert confirmation.PREDICTION_TAG in text
