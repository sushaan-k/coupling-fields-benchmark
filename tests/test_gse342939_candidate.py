from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "data/confirmation/gse342939_ra_bcell"
CANDIDATE = FREEZE / "candidate_designation_v1.json"
MANIFEST = FREEZE / "metadata_access_manifest_v1.json"
AMENDMENT = FREEZE / "pre_access_implementation_amendment_v1.json"
PROTOCOL = (
    ROOT
    / "docs/GSE342939_RA_BCELL_CITESEQ_HELD_DONOR_PROTOCOL_2026-08-29.md"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _donors(candidate: dict) -> list[dict]:
    return candidate["source_donors"] + candidate["held_donors"]


def test_metadata_access_manifest_is_exhaustive_and_nonnumeric() -> None:
    manifest = _load(MANIFEST)
    boundary = manifest["access_boundary"]
    records = manifest["records"]

    assert manifest["status"].endswith("BEFORE_ANY_NUMERIC_MATRIX_ACCESS")
    assert len(records) == boundary["network_get_response_bodies"] == 107
    assert sum(record["bytes"] for record in records) == 98_414_933
    assert boundary["network_get_response_bytes"] == 98_414_933
    assert boundary["official_geo_metadata_bodies"] == 3
    assert boundary["nonnumeric_feature_axis_bodies"] == 52
    assert boundary["nonnumeric_barcode_axis_bodies"] == 52
    assert boundary["numeric_matrix_body_gets"] == 0
    assert boundary["matrix_market_headers_or_entries_read"] == 0
    assert boundary["bcr_body_gets"] == 0
    assert boundary["raw_tar_body_gets"] == 0
    assert boundary["assay_values_associations_predictions_or_losses_computed"] == 0

    names = [record["name"] for record in records]
    assert len(names) == len(set(names))
    assert not any(name.endswith("matrix.mtx.gz") for name in names)
    assert not any("filtered_contig" in name for name in names)
    assert "GSE342939_RAW.tar" not in names
    assert all(record["assay_numeric_values_present"] is False for record in records)
    assert all(record["url"].startswith("https://ftp.ncbi.nlm.nih.gov/") for record in records)
    assert all(record["final_url"] == record["url"] for record in records)
    assert all(re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) for record in records)


def test_axis_structure_binds_every_library_and_exact_pairing() -> None:
    manifest = _load(MANIFEST)
    structure = manifest["axis_structure"]
    visits = structure["visit_barcode_intersections"]

    assert structure["gex_feature_axes"] == {
        "files": 26,
        "decoded_lines_each": 36_617,
        "decoded_sha256_each": (
            "60aa83680e0d8b87d562e7c5257d905a848ef7cbb95a4cad230ab8539f6301ad"
        ),
    }
    assert structure["cite_feature_axes"]["files"] == 26
    assert structure["cite_feature_axes"]["decoded_lines_each"] == 63
    assert structure["cite_feature_axes"]["includes_exact_unmapped_sentinel"] is True
    assert structure["all_axis_lines_unique_within_file"] is True
    assert len(visits) == 26
    assert len({visit["visit"] for visit in visits}) == 26
    assert all(visit["gex_unique"] == visit["gex_barcodes"] for visit in visits)
    assert all(visit["cite_unique"] == visit["cite_barcodes"] for visit in visits)
    assert all(visit["gex_all_suffix_minus1"] is True for visit in visits)
    assert min(visit["intersection"] for visit in visits) == 392
    assert max(visit["intersection"] for visit in visits) == 19_683
    assert next(v for v in visits if v["visit"] == "PC2A")["intersection"] == 392

    axis_records = [
        record
        for record in manifest["records"]
        if record["content_class"].startswith("nonnumeric_")
    ]
    assert len(axis_records) == 104
    assert sum(record["bytes"] for record in axis_records) == 98_386_130


def test_candidate_and_amendment_bind_exact_freeze_hashes() -> None:
    candidate = _load(CANDIDATE)
    amendment = _load(AMENDMENT)
    dependencies = amendment["freeze_dependencies"]

    assert candidate["metadata_bindings"]["metadata_access_manifest"]["sha256"] == _sha256(
        MANIFEST
    )
    assert candidate["metadata_bindings"]["metadata_access_manifest"]["bytes"] == (
        MANIFEST.stat().st_size
    )
    assert dependencies["candidate_sha256"] == _sha256(CANDIDATE)
    assert dependencies["metadata_access_manifest_sha256"] == _sha256(MANIFEST)
    assert dependencies["tags_created_by_this_task"] is False
    assert amendment["numeric_matrix_access_before_freeze"] is False


def test_physical_donor_split_is_disjoint_complete_and_reproducible() -> None:
    candidate = _load(CANDIDATE)
    source = candidate["source_donors"]
    held = candidate["held_donors"]
    all_donors = source + held

    source_ids = {record["donor"] for record in source}
    held_ids = {record["donor"] for record in held}
    assert source_ids == {"NN3", "PC3", "PC4", "PC5", "PN1", "PN2", "PN3"}
    assert held_ids == {"NN1", "NN2", "PC1", "PC2", "PN4", "PN5"}
    assert source_ids.isdisjoint(held_ids)
    assert len(source_ids | held_ids) == 13

    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for record in all_donors:
        by_stratum[record["stratum"]].append(record)

    salt = candidate["split_rule"]["salt"]
    expected_held = set()
    for stratum, records in by_stratum.items():
        ordered = sorted(
            records,
            key=lambda record: (
                hashlib.sha256(
                    f"{salt}\0{stratum}\0{record['donor']}".encode()
                ).hexdigest(),
                record["donor"],
            ),
        )
        expected_held.update(record["donor"] for record in ordered[:2])
        for record in records:
            assert record["split_digest_sha256"] == hashlib.sha256(
                f"{salt}\0{stratum}\0{record['donor']}".encode()
            ).hexdigest()
    assert held_ids == expected_held


def test_every_donor_keeps_both_visits_modalities_and_biosample_pairing() -> None:
    candidate = _load(CANDIDATE)
    axis_visits = {
        visit["visit"]: visit
        for visit in _load(MANIFEST)["axis_structure"]["visit_barcode_intersections"]
    }
    seen_gsms: set[str] = set()
    seen_numeric_urls: set[str] = set()

    for donor in _donors(candidate):
        assert [visit["timepoint"] for visit in donor["visits"]] == ["pre", "post"]
        assert [visit["visit"] for visit in donor["visits"]] == [
            donor["donor"] + "A",
            donor["donor"] + "B",
        ]
        for visit in donor["visits"]:
            gex = visit["gex"]
            cite = visit["cite"]
            assert gex["gsm"] != cite["gsm"]
            assert gex["gsm"] == axis_visits[visit["visit"]]["gex_gsm"]
            assert cite["gsm"] == axis_visits[visit["visit"]]["cite_gsm"]
            assert visit["nonnumeric_axis_intersection_barcodes"] == axis_visits[
                visit["visit"]
            ]["intersection"]
            for assay in (gex, cite):
                assert assay["gsm"] not in seen_gsms
                seen_gsms.add(assay["gsm"])
                matrix = assay["matrix"]
                assert matrix["body_accessed_before_freeze"] is False
                assert matrix["url"] not in seen_numeric_urls
                seen_numeric_urls.add(matrix["url"])
                assert matrix["filename"].endswith("_matrix.mtx.gz")

    assert len(seen_gsms) == len(seen_numeric_urls) == 52


def test_numeric_inventory_is_metadata_only_and_stage_partitioned() -> None:
    candidate = _load(CANDIDATE)
    inventory = candidate["numeric_matrix_inventory"]

    assert inventory == {
        "files": 52,
        "source_files": 28,
        "held_files": 24,
        "total_expected_bytes": 1_756_884_246,
        "source_expected_bytes": 814_592_401,
        "held_expected_bytes": 942_291_845,
        "response_bodies_accessed_before_freeze": 0,
    }
    assert candidate["metadata_bindings"]["all_series_raw_tar_forbidden"] is True
    assert candidate["metadata_bindings"]["bcr_files_out_of_scope"] is True
    assert candidate["pre_design_access_attestation"]["numeric_matrix_body_gets"] == 0
    assert candidate["pre_design_access_attestation"][
        "matrix_market_headers_or_entries_read"
    ] == 0


def test_panel_is_exact_one_to_one_and_fixed_before_values() -> None:
    candidate = _load(CANDIDATE)
    panel = candidate["ordered_cognate_panel"]
    rule = candidate["panel_rule"]

    assert len(panel) == rule["markers"] == 45
    assert rule["coordinate_universe"] == "all 2025 ordered RNA-to-ADT pairs"
    assert len({marker["rna"] for marker in panel}) == 45
    assert len({marker["adt"] for marker in panel}) == 45
    assert len({marker["adt_axis_exact"] for marker in panel}) == 45
    for marker in panel:
        assert marker["adt_axis_exact"] == (
            f"{marker['adt']}-{marker['adt_sequence']}"
        )
        assert re.fullmatch(r"[ACGT]{15}", marker["adt_sequence"])
        assert marker["adt_definition_id"].startswith("ADT_")
    assert rule["axis_metadata_cannot_add_replace_reorder_or_drop_markers"] is True
    assert "CD3 because the ADT target does not select one CD3 transcript" in rule[
        "excluded_categories"
    ]


def test_sampling_rule_uses_axis_support_without_assay_outcomes() -> None:
    candidate = _load(CANDIDATE)
    states = candidate["states_and_sampling"]

    assert states["maximum_cells_per_donor_visit"] == 512
    assert states["minimum_cells_per_donor_visit"] == 128
    assert "2*floor(m/2)" in states["selected_cell_count_rule"]
    assert "exact equality" in states["barcode_pairing"]
    assert states["rna_state"] == "raw GEX UMI count greater than zero"
    assert "equal low and high halves" in states["adt_state"]
    assert "within donor-visit" in states["destroyed_link"]


def test_longitudinal_fit_and_masks_are_training_donor_only() -> None:
    amendment = _load(AMENDMENT)
    paired = amendment["paired_visit_contract"]
    masks = amendment["fold_specific_comparison_masks"]
    primary = amendment["estimator_and_comparator_contract"]["primary"]
    graph = amendment["graph_contract"]

    assert paired["visit_code"] == {"pre": -0.5, "post": 0.5}
    assert "both visits" in paired["blocking"]
    assert "physical donors" in paired["bootstrap_and_sign_test"]
    assert "other six source donors" in masks["source_fold_rule"]
    assert "paired counts and association values cannot" in masks["source_fold_rule"]
    assert "all seven source donors" in masks["final_rule"]
    assert masks["minimum_scored_coordinates_per_visit"] == 256
    assert len(masks["training_only_conditions_per_coordinate"]) == 5
    assert primary["family"].startswith("paired longitudinal")
    assert "Mu + q*Delta + B_d + q*C_d" in primary["parameterization"]
    assert "donor effects are not transported" in primary["recipient_coordinates"]
    assert graph["neighbors"] == 2
    assert "Validation and held profiles are forbidden" in graph["fold_rule"]


def test_classical_poisson_is_fixed_interaction_not_nch_hybrid() -> None:
    amendment = _load(AMENDMENT)
    methods = amendment["estimator_and_comparator_contract"]
    poisson = methods["pooled_saturated_poisson"]

    assert "every training donor" in poisson["fit"]
    assert "including fixed-margin-degenerate tables" in poisson["fit"]
    assert "row and column nuisance parameters" in poisson[
        "recipient_reconstruction"
    ]
    assert "odds ratio" in poisson["recipient_reconstruction"]
    assert "Never pass" in poisson["forbidden_reconstruction"]
    assert "noncentral-hypergeometric" in poisson["forbidden_reconstruction"]
    assert "normalized maximum cell error 1e-8" in poisson["certificates"]
    assert methods["primary"]["recipient_reconstruction"].startswith(
        "Exact noncentral-hypergeometric"
    )


def test_gates_use_physical_donors_and_exact_six_of_six_confirmation() -> None:
    candidate = _load(CANDIDATE)
    source = candidate["source_promotion"]
    held = candidate["held_confirmation"]

    assert source["minimum_relative_reduction_vs_residual_poisson_destroyed_and_independence"] == 0.05
    assert source["minimum_favorable_source_donors"] == 6
    assert source["bootstrap_draws"] == held["bootstrap_draws"] == 20_000
    assert source["bootstrap_seed"] == held["bootstrap_seed"] == 20260830
    assert held["minimum_favorable_held_donors"] == 6
    assert held["one_sided_exact_sign_test_p_if_six_of_six"] == 1 / 64
    assert held["maximum_one_sided_exact_sign_test_p"] == 0.025
    assert held["rerun_permitted"] is False
    assert "physical-donor-equal" in held["primary_metric"]


def test_access_firewall_separates_rna_adt_prediction_and_score() -> None:
    amendment = _load(AMENDMENT)
    firewall = amendment["access_firewall"]

    assert firewall["future_cli_stages"] == [
        "claim-source",
        "run-source",
        "claim-held-rna",
        "run-held-rna",
        "claim-held-adt",
        "run-held-adt",
        "predict-held",
        "authorize-score",
        "score-held",
    ]
    assert len(firewall["future_public_tags"]) == 9
    assert "28 source matrix URLs" in firewall["source_allowlist"]
    assert "every held matrix URL remains unreachable" in firewall[
        "source_failure"
    ]
    assert "do not request CITE matrices" in firewall["held_rna_stage"]
    assert "cannot read RNA states" in firewall["held_adt_stage"]
    assert "public held row/column margins" in firewall["prediction_stage"]
    assert "networking disabled" in firewall["score_stage"]
    assert "atomic rename" in firewall["atomic_publication"]
    assert "including the file that fails" in firewall["partial_provenance"]


def test_private_future_state_paths_are_ignored() -> None:
    ignore = (ROOT / ".gitignore").read_text().splitlines()
    expected = {
        "data/confirmation/gse342939_ra_bcell/private_held_selected_barcodes_v1.json",
        "data/confirmation/gse342939_ra_bcell/private_held_rna_states_v1.npz",
        "data/confirmation/gse342939_ra_bcell/private_held_adt_states_v1.npz",
    }
    assert expected <= set(ignore)


def test_protocol_states_freeze_without_claiming_an_outcome() -> None:
    text = PROTOCOL.read_text()
    assert "No Matrix Market header or entry" in text
    assert "six\ndonor-disjoint individuals" in text
    assert "classical\ncontinuous log-linear table" in text
    assert "does not create public tags" in text
    assert "Numeric\nmatrix access remains forbidden" in text
