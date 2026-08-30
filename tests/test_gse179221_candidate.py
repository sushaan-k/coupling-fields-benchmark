from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = (
    ROOT / "data/confirmation/gse179221_bmmc/candidate_designation_v1.json"
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
