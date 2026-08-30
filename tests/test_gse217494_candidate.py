from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "data/confirmation/gse217494_heart/candidate_designation_v1.json"
PROTOCOL = ROOT / "docs/GSE217494_CARDIAC_CITESEQ_HELD_DONOR_PROTOCOL_2026-08-30.md"
HARDENING = ROOT / "data/confirmation/gse217494_heart/pre_access_hardening_v1.json"
COGNATE_AXIS = ROOT / "data/confirmation/gse217494_heart/cognate_axis_v1.tsv"


def _candidate() -> dict:
    return json.loads(CANDIDATE.read_text())


def _hardening() -> dict:
    return json.loads(HARDENING.read_text())


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
    assert (
        sum(
            row["barcodes_bytes"] + row["features_bytes"] + row["matrix_bytes"]
            for row in samples
        )
        == candidate["files"]["all_triplet_bytes"]
    )


def test_pre_access_hardening_binds_candidate_and_barcode_axes() -> None:
    candidate = _candidate()
    hardening = _hardening()
    candidate_by_sample = {row["sample"]: row for row in candidate["samples"]}
    axes_by_sample = {row["sample"]: row for row in hardening["barcode_axes"]}
    features_by_sample = {row["sample"]: row for row in hardening["feature_axes"]}

    assert hardening["candidate_tag"] == "gse217494-heart-v1-candidate"
    assert hardening["candidate_commit"] == ("0001176fa055ee21a55cd51a4d8fcb9e1aa5f468")
    assert (
        hardening["candidate_designation_sha256"]
        == hashlib.sha256(CANDIDATE.read_bytes()).hexdigest()
    )
    assert hardening["numeric_matrix_body_header_or_entry_read"] is False
    assert hardening["held_matrix_body_header_or_entry_requested"] is False
    assert hardening["candidate_artifact_changed"] is False
    assert hardening["rerun_permitted"] is False

    assert set(axes_by_sample) == set(candidate_by_sample)
    assert set(features_by_sample) == set(candidate_by_sample)
    assert len(axes_by_sample) == 22
    assert all(
        len(value) == 64
        for value in {row["gzip_sha256"] for row in axes_by_sample.values()}
    )
    assert all(
        len(value) == 64
        for value in {row["axis_sha256"] for row in axes_by_sample.values()}
    )
    assert len({row["gzip_sha256"] for row in axes_by_sample.values()}) == 22
    assert len({row["axis_sha256"] for row in axes_by_sample.values()}) == 22
    assert all(len(row["gzip_sha256"]) == 64 for row in features_by_sample.values())
    assert len({row["gzip_sha256"] for row in features_by_sample.values()}) == 22
    assert len({row["axis_sha256"] for row in features_by_sample.values()}) == 1
    assert (
        hardening["clarifications"]["feature_axes_decompressed_byte_identical"] is True
    )
    for sample, axis in axes_by_sample.items():
        candidate_row = candidate_by_sample[sample]
        assert axis["role"] == candidate_row["role"]
        assert axis["bytes"] == candidate_row["barcodes_bytes"]
        assert axis["count"] >= candidate["primary_cell_budget_per_donor"]
        feature_axis = features_by_sample[sample]
        assert feature_axis["role"] == candidate_row["role"]
        assert feature_axis["bytes"] == candidate_row["features_bytes"]
        assert feature_axis["count"] == (
            candidate["axis_inventory"]["gene_expression_features"]
            + candidate["axis_inventory"]["antibody_capture_features"]
        )
        assert (
            feature_axis["axis_sha256"]
            == candidate["axis_inventory"]["all_22_decompressed_feature_axes_sha256"]
        )


def test_amended_protocol_matches_pre_access_hardening() -> None:
    hardening = _hardening()
    assert (
        hardening["amended_protocol_sha256"]
        == hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    )
    assert hardening["clarifications"]["mandatory_gate_comparators"] == [
        "pooled_fixed_interaction_poisson",
        "etiology_specific_fixed_interaction_poisson",
        "strongest_remaining_classical_comparator",
        "destroyed_links",
    ]


def test_cognate_axis_is_complete_unique_and_hash_bound() -> None:
    candidate = _candidate()
    hardening = _hardening()
    with COGNATE_AXIS.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    symbols = [row["symbol"] for row in rows]

    assert (
        len(rows) == candidate["axis_inventory"]["unique_exact_rna_adt_symbol_cognates"]
    )
    assert symbols == sorted(set(symbols))
    assert len({int(row["rna_row_1based"]) for row in rows}) == len(rows)
    assert len({int(row["adt_row_1based"]) for row in rows}) == len(rows)
    assert all(1 <= int(row["rna_row_1based"]) <= 33538 for row in rows)
    assert all(33539 <= int(row["adt_row_1based"]) <= 33817 for row in rows)
    assert (
        hardening["cognate_axis_sha256"]
        == hashlib.sha256(COGNATE_AXIS.read_bytes()).hexdigest()
    )
    symbol_bytes = ("\n".join(symbols) + "\n").encode()
    assert (
        hardening["cognate_symbol_axis_sha256"]
        == hashlib.sha256(symbol_bytes).hexdigest()
    )
    assert (
        hardening["cognate_symbol_axis_sha256"]
        == candidate["axis_inventory"]["sorted_exact_cognate_symbol_axis_sha256"]
    )


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
    assert "512 - max_count_frequency >= 16" in protocol
    assert "nonfinite or zero-norm profile is ineligible" in protocol
    assert "mandatory gate comparators" in normalized
    assert "unpenalized poisson likelihood on every training heart" in normalized
    assert "degenerate-margin tables remain in the likelihood" in normalized
    assert "four-column one-hot encoding" in normalized
    assert "every etiology coefficient receives the same ridge penalty" in normalized
    assert "ranked by `(euclidean distance, marker symbol)`" in normalized
    assert "at least seven of eight held hearts" in protocol
