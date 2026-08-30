from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = (
    ROOT / "data/confirmation/gse179221_bmmc/candidate_designation_v1.json"
)
AMENDMENT = (
    ROOT
    / "data/confirmation/gse179221_bmmc/pre_access_implementation_amendment_v1.json"
)


def _load() -> dict:
    return json.loads(CANDIDATE.read_text())


def test_split_is_disjoint_complete_and_reproduces_frozen_hash_rule() -> None:
    record = _load()
    source = record["source_files"]
    held = record["held_files"]
    all_files = source + held

    source_gsms = {item["gsm"] for item in source}
    held_gsms = {item["gsm"] for item in held}
    assert source_gsms.isdisjoint(held_gsms)
    assert len(source_gsms) == 8
    assert len(held_gsms) == 10
    assert len(source_gsms | held_gsms) == 18

    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for item in all_files:
        by_stratum[item["stratum"]].append(item)

    salt = record["split_rule"]["salt"]
    expected_held = set()
    for stratum, items in by_stratum.items():
        ordered = sorted(
            items,
            key=lambda item: (
                hashlib.sha256(
                    f"{salt}\0{stratum}\0{item['gsm']}".encode()
                ).hexdigest(),
                item["gsm"],
            ),
        )
        expected_held.update(
            item["gsm"] for item in ordered[: (len(ordered) + 1) // 2]
        )
    assert held_gsms == expected_held


def test_candidate_binds_all_files_and_preaccess_boundary() -> None:
    record = _load()
    files = record["source_files"] + record["held_files"]

    assert sum(item["bytes"] for item in record["source_files"]) == 240_051_651
    assert sum(item["bytes"] for item in record["held_files"]) == 331_345_364
    assert all(item["filename"].endswith("_raw_feature_bc_matrix.h5") for item in files)
    assert len(record["ordered_cognate_candidates"]) == 9
    assert record["panel_rule"]["analysis_panel"] == (
        "all 81 ordered RNA-to-ADT pairs"
    )

    access = record["pre_design_access_attestation"]
    assert access["repository_and_local_search_found_prior_use"] is False
    assert "any per-donor H5 response body" in access["not_accessed"]
    assert record["metadata_bindings"]["all_donor_tar_forbidden"] is True


def test_preaccess_amendment_closes_mask_and_poisson_ambiguities() -> None:
    record = json.loads(AMENDMENT.read_text())

    assert record["status"] == "FROZEN_BEFORE_ANY_H5_BODY_ACCESS"
    assert record["access_attestation"]["h5_response_bodies_requested"] == 0
    masks = record["fold_specific_comparison_masks"]
    assert "only the other seven source donors" in masks["correction"]
    assert "all eight source donors only" in masks["correction"]
    assert masks["minimum_scored_coordinates_per_donor"] == 64

    graph = record["graph_contract"]
    assert graph["neighbors"] == 2
    assert "mean log1p raw antibody count" in graph["adt_profile"]
    assert "forbidden graph inputs" in graph["fold_rule"]

    poisson = record["estimator_and_comparator_contract"][
        "pooled_saturated_poisson"
    ]
    assert "every training donor" in poisson["fit"]
    assert "binary_table_from_helmert_coordinate" in poisson[
        "recipient_reconstruction"
    ]
    assert "noncentral-hypergeometric" in poisson["forbidden_reconstruction"]


def test_preaccess_amendment_freezes_exact_panel_resolution_and_firewall() -> None:
    record = json.loads(AMENDMENT.read_text())
    feature = record["feature_contract"]
    panel = feature["ordered_cognates_and_exact_adt_aliases"]

    assert feature["gene_expression_type"] == "Gene Expression"
    assert feature["antibody_capture_type"] == "Antibody Capture"
    assert feature["mitochondrial_rule"].endswith("begins with MT-.")
    assert [item["rna"] for item in panel] == [
        "CD3D",
        "NCAM1",
        "CD19",
        "CD14",
        "FCGR3A",
        "MS4A1",
        "CD27",
        "CD38",
        "CD79B",
    ]
    assert panel[1]["aliases"] == ["NCAM", "CD56"]
    assert panel[-1]["aliases"] == ["CD79b (Igβ)", "CD79b"]

    firewall = record["access_firewall"]
    assert "before the first source URL GET" in firewall["source_attempt"]
    assert "makes every held URL unreachable" in firewall["source_failure"]
    assert "may not import or call a joint-table constructor" in firewall[
        "held_margin_stage"
    ]
