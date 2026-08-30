from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESIGNATION = (
    ROOT / "data/confirmation/gse214546_teaseq/candidate_designation_v1.json"
)
PROTOCOL = ROOT / "docs/GSE214546_TEASEQ_HELD_DONOR_CONFIRMATION_PROTOCOL_2026-08-30.md"


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
        assert {group: sum(s["age_group"] == group for s in samples) for group in ("adult", "pediatric")} == {
            "adult": 4,
            "pediatric": 4,
        }
        assert {batch: sum(s["batch"] == batch for s in samples) for batch in ("B065", "B069", "B076")} == {
            "B065": 2,
            "B069": 2,
            "B076": 4,
        }

    assert sum(sample["h5_bytes"] for sample in source) == payload["source_h5_bytes"]
    assert sum(sample["h5_bytes"] for sample in held) == payload["held_h5_bytes"]
    assert max(sample["h5_bytes"] for sample in payload["samples"]) == payload[
        "maximum_single_h5_bytes"
    ]


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
