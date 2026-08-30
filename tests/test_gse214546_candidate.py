from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESIGNATION = ROOT / "data/confirmation/gse214546_teaseq/candidate_designation_v1.json"
PROTOCOL = ROOT / "docs/GSE214546_TEASEQ_HELD_DONOR_CONFIRMATION_PROTOCOL_2026-08-30.md"
AMENDMENT = (
    ROOT / "data/confirmation/gse214546_teaseq/pre_access_schema_amendment_v1.json"
)
CLARIFICATION = (
    ROOT
    / "data/confirmation/gse214546_teaseq/pre_access_implementation_clarification_v1.json"
)


def _designation() -> dict:
    return json.loads(DESIGNATION.read_text(encoding="utf-8"))


def test_candidate_is_balanced_and_donor_disjoint() -> None:
    payload = _designation()
    source = [sample for sample in payload["samples"] if sample["role"] == "source"]
    held = [sample for sample in payload["samples"] if sample["role"] == "held"]

    assert len(source) == len(held) == 8
    assert {sample["donor"] for sample in source}.isdisjoint(
        sample["donor"] for sample in held
    )
    for samples in (source, held):
        assert {
            group: sum(s["age_group"] == group for s in samples)
            for group in ("adult", "pediatric")
        } == {
            "adult": 4,
            "pediatric": 4,
        }
        assert {
            batch: sum(s["batch"] == batch for s in samples)
            for batch in ("B065", "B069", "B076")
        } == {
            "B065": 2,
            "B069": 2,
            "B076": 4,
        }

    assert sum(sample["h5_bytes"] for sample in source) == payload["source_h5_bytes"]
    assert sum(sample["h5_bytes"] for sample in held) == payload["held_h5_bytes"]
    assert (
        max(sample["h5_bytes"] for sample in payload["samples"])
        == payload["maximum_single_h5_bytes"]
    )


def test_candidate_marker_axis_is_unique() -> None:
    markers = _designation()["candidate_markers"]
    assert len(markers) == 32
    for key in ("protein", "rna", "adt_barcode"):
        values = [marker[key] for marker in markers]
        assert len(set(values)) == len(values)


def test_protocol_fixes_firewall_comparators_and_gate() -> None:
    text = PROTOCOL.read_text(encoding="utf-8")
    for required in (
        "pooled fixed-interaction Poisson",
        "Age-stratified fixed-interaction Poisson",
        "Destroyed links",
        "20,000-draw paired-donor bootstrap",
        "at least seven of eight donors",
        "GSE214546-DESTROY-v1",
        "No failed or completed campaign may be rerun",
    ):
        assert required in text


def test_pre_access_amendment_uses_only_schema_and_string_axes() -> None:
    payload = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    assert payload["numeric_count_or_sparse_value_dataset_read"] is False
    assert payload["held_h5_requested_or_opened"] is False
    assert payload["inventory_access"]["forbidden_value_datasets_read"] == []
    assert payload["amendment"]["source_held_split_changed"] is True
    assert payload["amendment"]["estimator_or_grid_changed"] is False
    assert payload["amendment"]["maximum_retained_markers"] == 24
    markers = payload["candidate_markers"]
    assert len(markers) == 53
    assert len({marker["protein"] for marker in markers}) == 53
    assert len({marker["rna"] for marker in markers}) == 53

    samples = {sample["gsm"]: sample for sample in _designation()["samples"]}
    source = [samples[gsm] for gsm in payload["amended_split"]["source"]]
    held = [samples[gsm] for gsm in payload["amended_split"]["held"]]
    assert sum(sample["h5_bytes"] for sample in source) == 653304346
    assert sum(sample["h5_bytes"] for sample in held) == 648239062
    for split in (source, held):
        assert {
            group: sum(s["age_group"] == group for s in split)
            for group in ("adult", "pediatric")
        } == {"adult": 4, "pediatric": 4}
        assert {
            batch: sum(s["batch"] == batch for s in split)
            for batch in ("B065", "B069", "B076")
        } == {"B065": 2, "B069": 2, "B076": 4}


def test_pre_access_clarification_fixes_axis_and_comparator_availability() -> None:
    payload = json.loads(CLARIFICATION.read_text(encoding="utf-8"))
    assert payload["numeric_count_or_sparse_value_dataset_read"] is False
    assert payload["held_h5_requested_or_opened"] is False
    assert payload["metadata_eligibility"] == {
        "barcode_column": "barcodes",
        "eligibility_column": "singlet",
        "eligible_literal": "TRUE",
        "comparison": "exact case-sensitive string equality",
    }
    axis = payload["source_axis_algorithm"]
    assert axis["minimum_markers"] == 20
    assert axis["maximum_markers"] == 24
    assert axis["pseudocount"] is None
    assert "strictly positive pooled 2x2 cells" in axis["selection"]
    assert payload["classical_comparators"]["zero_cell_policy"].startswith(
        "unavailable"
    )
