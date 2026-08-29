import gzip
import json
from pathlib import Path

import numpy as np

from experiments import confirm_gse158769_citeseq as confirmation
from experiments import confirm_gse314416_citeseq as shared_core


def _synthetic_raw(path: Path, cells: list[str], *, poison_adt: bool = False) -> None:
    adt_labels = (
        "CD3.1",
        "CD4.1",
        "CD5.1",
        "CD8a",
        "CD161",
        "CD127",
        "CD27.1",
        "CD38.1",
        "CD26",
    )
    with gzip.open(path, "wt", newline="") as stream:
        stream.write("feature\t" + "\t".join(cells) + "\n")
        for marker, feature in enumerate(confirmation.RNA_FEATURES):
            values = [str((index + marker) % 3) for index in range(len(cells))]
            stream.write(feature + "\t" + "\t".join(values) + "\n")
        for marker, feature in enumerate(adt_labels):
            if poison_adt:
                values = ["HELD_ADT_POISON"] * len(cells)
            else:
                values = [
                    str((index * (marker + 1)) % 17) for index in range(len(cells))
                ]
            stream.write(feature + "\t" + "\t".join(values) + "\n")


def test_frozen_split_panel_and_configuration_grid():
    assert len(confirmation.MARKERS) == 9
    assert len(set(confirmation.RNA_FEATURES)) == 9
    assert all(confirmation.ADT_ALIASES)
    assert len(set(confirmation.CALIBRATION_BATCHES)) == 16
    assert len(set(confirmation.PILOT_BATCHES)) == 14
    assert len(set(confirmation.HELD_BATCHES)) == 16
    assert set(confirmation.CALIBRATION_BATCHES) | set(
        confirmation.PILOT_BATCHES
    ) | set(confirmation.HELD_BATCHES) == set(range(1, 47))
    assert len(confirmation._primary_configs()) == 144


def test_metadata_preflight_is_count_blind_and_binds_all_donors():
    payload = json.loads(confirmation.DEFAULT_PREFLIGHT.read_text())
    assert payload["status"] == "PASS"
    assert payload["role_counts"] == {"calibration": 85, "pilot": 69, "held": 80}
    assert payload["access_audit"] == {
        "raw_matrix_downloaded": False,
        "raw_matrix_decompressed": False,
        "rna_numeric_values_read": 0,
        "adt_numeric_values_read": 0,
    }
    assert len(payload["donors"]) == 259
    assert sum(row["role"] == "excluded_repeated" for row in payload["donors"]) == 12
    assert (
        sum(row["role"] == "excluded_under_budget" for row in payload["donors"]) == 13
    )


def test_streaming_reducer_preserves_cell_order_and_reads_only_selected_rows(
    tmp_path: Path,
):
    cells = [f"cell-{index:04d}" for index in range(confirmation.CELL_BUDGET)]
    path = tmp_path / "matrix.tsv.gz"
    _synthetic_raw(path, cells)
    selected = {"D": list(reversed(cells))}
    counts, audit = confirmation._stream_panel(
        path, ["D"], selected, frozenset({"rna", "adt"})
    )
    assert counts["D"]["rna"].shape == (512, 9)
    assert counts["D"]["adt"].shape == (512, 9)
    assert counts["D"]["rna"][0, 0] == (511 % 3)
    assert counts["D"]["adt"][0, 0] == (511 % 17)
    assert audit["selected_cell_columns"] == 512
    assert audit["full_tsv_materialized"] is False


def test_held_rna_stream_never_tokenizes_poison_adt_values(tmp_path: Path):
    cells = [f"cell-{index:04d}" for index in range(confirmation.CELL_BUDGET)]
    path = tmp_path / "matrix.tsv.gz"
    _synthetic_raw(path, cells, poison_adt=True)
    counts, audit = confirmation._stream_panel(
        path, ["held"], {"held": cells}, frozenset({"rna"})
    )
    assert counts["held"]["rna"].shape == (512, 9)
    assert audit["adt_feature_rows_tokenized"] == 0
    assert audit["adt_candidate_rows_skipped_without_tokenization"] >= 9


def test_adt_midrank_and_destroyed_link_are_exact_and_reproducible():
    cells = [f"cell-{index:04d}" for index in range(confirmation.CELL_BUDGET)]
    counts = np.zeros((512, 9), dtype=int)
    first = confirmation._adt_states(counts, cells, "D")
    second = confirmation._adt_states(counts, cells, "D")
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first.sum(axis=0), np.full(9, 256))
    destroyed = confirmation._destroyed_adt(first, cells, "D")
    np.testing.assert_array_equal(destroyed.sum(axis=0), first.sum(axis=0))
    assert not np.array_equal(destroyed, first)


def test_shared_core_configuration_is_scoped_and_restored():
    original = shared_core.MARKERS
    with confirmation._configured_core():
        assert shared_core.MARKERS == confirmation.MARKERS
    assert shared_core.MARKERS == original


def test_batch_block_gate_uses_donor_and_batch_requirements():
    donors = [f"D{index:02d}" for index in range(69)]
    batches = {
        donor: confirmation.PILOT_BATCHES[index % 14]
        for index, donor in enumerate(donors)
    }
    losses = {
        "primary": np.full(69, 0.80),
        "best_residual": np.full(69, 1.00),
        "destroyed_link": np.full(69, 1.05),
    }
    gate = confirmation._gate(donors, batches, losses, held=False)
    assert gate["passes"]
    comparison = gate["primary_vs_selected_classical_residual"]
    assert comparison["favorable_donors"] == 69
    assert comparison["favorable_batches"] == 14
    assert comparison["batch_block_bootstrap_95_interval"][1] < 0


def test_exact_held_batch_sign_flip_is_enumerated():
    result = confirmation._exact_batch_sign_flip(np.full(16, -1.0))
    assert result["draws"] == 65536
    assert result["one_sided_p"] < 0.025


def test_source_manifest_leaves_raw_digest_outcome_blind_until_acquisition():
    source = json.loads(confirmation.DEFAULT_MANIFEST.read_text())
    raw = source["files"]["raw_linked_matrix"]
    assert raw["bytes"] == 783271940
    assert raw["sha256"] is None
    access = json.loads(confirmation.DEFAULT_METADATA_ACCESS.read_text())
    assert access["status"] == "METADATA_ONLY"
    assert access["count_matrix_access"]["numeric_values_read"] == 0


def test_stage_tags_are_distinct():
    assert (
        len(
            {
                confirmation.PROTOCOL_TAG,
                confirmation.SOURCE_TAG,
                confirmation.DEVELOPMENT_TAG,
                confirmation.PREDICTION_TAG,
            }
        )
        == 4
    )
