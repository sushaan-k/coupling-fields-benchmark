from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = (
    ROOT / "data/confirmation/gse217494_heart/candidate_designation_v1.json"
)
PROTOCOL = (
    ROOT / "docs/GSE217494_CARDIAC_CITESEQ_HELD_DONOR_PROTOCOL_2026-08-30.md"
)


def _candidate() -> dict:
    return json.loads(CANDIDATE.read_text())


def test_frozen_split_replays_from_first_semantic_salt() -> None:
    candidate = _candidate()
    samples = candidate["samples"]
    salt = candidate["split_rule"]["salt"]
    for etiology in ("Donor", "AMI", "ICM", "NICM"):
        stratum = [row for row in samples if row["etiology"] == etiology]
        ranked = sorted(
            stratum,
            key=lambda row: hashlib.sha256(
                f"{salt}|{etiology}|{row['sample']}".encode()
            ).hexdigest(),
        )
        assert {row["sample"] for row in ranked[:2]} == {
            row["sample"] for row in stratum if row["role"] == "held"
        }


def test_candidate_counts_and_public_byte_totals_are_bound() -> None:
    candidate = _candidate()
    samples = candidate["samples"]
    assert len(samples) == 22
    assert sum(row["role"] == "source" for row in samples) == 14
    assert sum(row["role"] == "held" for row in samples) == 8
    assert all(row["metadata_cells"] >= 512 for row in samples)

    def triplet_bytes(role: str) -> int:
        return sum(
            row["barcodes_bytes"] + row["features_bytes"] + row["matrix_bytes"]
            for row in samples
            if row["role"] == role
        )

    assert triplet_bytes("source") == candidate["files"]["source_triplet_bytes"]
    assert triplet_bytes("held") == candidate["files"]["held_triplet_bytes"]
    assert sum(
        row["barcodes_bytes"] + row["features_bytes"] + row["matrix_bytes"]
        for row in samples
    ) == candidate["files"]["all_triplet_bytes"]


def test_protocol_keeps_held_assays_closed_until_fields_are_public() -> None:
    candidate = _candidate()
    protocol = PROTOCOL.read_text()
    normalized = protocol.lower()
    assert candidate["numeric_assay_matrix_entry_accessed_before_designation"] is False
    assert (
        candidate["pre_designation_metadata_access"][
            "held_matrix_market_body_or_header_requested"
        ]
        is False
    )
    assert candidate["published_cell_labels_used_for_selection"] is False
    assert "all eight held matrix bodies remain physically unopened" in normalized
    assert "source-only interaction fields" in protocol
    assert "Pooled fixed-interaction Poisson" in protocol
    assert "Etiology-specific fixed-interaction Poisson" in protocol
    assert "at least 16 ADT counts" in protocol
    assert "zero-variance or nonfinite marker profile is ineligible" in protocol
    assert "at least seven of eight held hearts" in protocol
