import json
from pathlib import Path

from experiments import preflight_gse252762_celiac as preflight


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = (
    ROOT / "results/development/gse252762_celiac_metadata_preflight_v1.json"
)


def test_frozen_preflight_has_disjoint_complete_roles() -> None:
    payload = json.loads(PREFLIGHT.read_text())
    assert payload["status"] == "PASS"
    assert payload["numeric_matrix_gets"] == 0
    assert payload["cell_budget"] == preflight.CELL_BUDGET
    assert payload["role_counts"] == {
        "calibration": 9,
        "pilot": 7,
        "held": 13,
    }
    samples = payload["samples"]
    assert len(samples) == 29
    assert len({sample["sample_id"] for sample in samples}) == 29
    assert {sample["sample_id"] for sample in samples if sample["role"] == "held"} == set(
        preflight.HELD
    )
    assert {sample["batch"] for sample in samples if sample["role"] == "held"} == {6}
    assert {
        sample["condition"] for sample in samples if sample["role"] == "held"
    } == {"ACD", "GFD", "CONTROL"}
    assert all(len(sample["selected_barcodes"]) == 256 for sample in samples)
    assert all(
        len(sample["selected_columns_1_based"]) == 256 for sample in samples
    )


def test_frozen_marker_and_matrix_axes_are_exact() -> None:
    payload = json.loads(PREFLIGHT.read_text())
    assert [
        (marker["rna"], marker["adt"]) for marker in payload["markers"]
    ] == list(preflight.MARKERS)
    assert all(batch["rna_shape"][0] == 36_601 for batch in payload["batches"])
    assert all(batch["cite_shape"][0] == 204 for batch in payload["batches"])
    for batch in payload["batches"]:
        matrix_files = [
            record for record in batch["files"] if record.get("numeric_outcome")
        ]
        assert len(matrix_files) == 2
        assert all(record["sha256"] is None for record in matrix_files)
