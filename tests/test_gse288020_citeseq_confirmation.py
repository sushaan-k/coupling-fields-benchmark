from __future__ import annotations

import json
import math
import inspect
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from scipy import sparse

from experiments import confirm_gse288020_citeseq as confirmation


def _synthetic_schema() -> dict[str, list[str]]:
    return {
        "id": [*confirmation.RNA_IDS, *confirmation.ADT_IDS],
        "name": [*confirmation.RNA_SYMBOLS, *confirmation.ADT_IDS],
        "feature_type": [
            *(["Gene Expression"] * len(confirmation.RNA_IDS)),
            *(["Antibody Capture"] * len(confirmation.ADT_IDS)),
        ],
    }


def _write_h5(
    path: Path, *, poison_matrix: bool = False, adt_dense: np.ndarray | None = None
) -> None:
    schema = _synthetic_schema()
    string = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as handle:
        matrix = handle.create_group("matrix")
        features = matrix.create_group("features")
        for key, values in schema.items():
            features.create_dataset(key, data=np.asarray(values, dtype=string))
        if poison_matrix:
            matrix["barcodes"] = h5py.ExternalLink("poison.h5", "/barcodes")
            matrix["data"] = h5py.ExternalLink("poison.h5", "/data")
            matrix["indices"] = h5py.ExternalLink("poison.h5", "/indices")
            matrix["indptr"] = h5py.ExternalLink("poison.h5", "/indptr")
            matrix["shape"] = h5py.ExternalLink("poison.h5", "/shape")
            return
        dense = np.zeros((len(schema["id"]), 4), dtype=np.int32)
        dense[: len(confirmation.RNA_IDS), 1::2] = 1
        if adt_dense is None:
            dense[len(confirmation.RNA_IDS) :, :] = np.arange(1, 5)
        else:
            dense[len(confirmation.RNA_IDS) :, :] = adt_dense
        csc = sparse.csc_matrix(dense)
        matrix.create_dataset("barcodes", data=np.asarray(["A", "B", "C", "D"], dtype=string))
        matrix.create_dataset("data", data=csc.data)
        matrix.create_dataset("indices", data=csc.indices)
        matrix.create_dataset("indptr", data=csc.indptr)
        matrix.create_dataset("shape", data=np.asarray(csc.shape, dtype=np.int64))


def test_frozen_split_is_donor_disjoint_age_balanced_and_disease_shifted():
    designation = confirmation._designation()
    samples = designation["samples"]
    assert len({sample["donor"] for sample in samples}) == 23
    assert sum(sample["role"] == "calibration" for sample in samples) == 7
    assert sum(sample["role"] == "pilot" for sample in samples) == 7
    assert sum(sample["role"] == "held" for sample in samples) == 9
    assert {
        sample["condition"] for sample in samples if sample["role"] != "held"
    } == {"MGUS"}
    assert {sample["condition"] for sample in samples if sample["role"] == "held"} == {
        "MM"
    }
    assert [
        sum(
            sample["role"] == role and sample["immune_age"] == age
            for sample in samples
        )
        for role in ("calibration", "pilot", "held")
        for age in ("Young", "Old")
    ] == [4, 3, 3, 4, 4, 5]


def test_official_geo_parser_reconstructs_labels_and_normalizes_r005_typo(
    tmp_path: Path,
):
    source = json.loads(confirmation.DEFAULT_DESIGNATION.read_text())["samples"]
    lines = []
    for sample in source:
        disease = (
            "Multiple Myeloma"
            if sample["condition"] == "MM"
            else "monoclonal gammopathy of uncertain significance"
        )
        age = f"Immune-Age {sample['immune_age']}"
        if sample["donor"] == "R005":
            age = "mmune-Age  Young"
        lines.extend(
            [
                f"^SAMPLE = {sample['gsm']}",
                f"!Sample_title = {sample['condition']} {sample['donor']} Immune-Age {sample['immune_age']}",
                f"!Sample_characteristics_ch1 = disease state: {disease}",
                f"!Sample_characteristics_ch1 = age group: {age}",
                f"!Sample_description = Library name: {sample['donor']}BM",
                f"!Sample_supplementary_file_1 = ftp://example/{sample['filename']}",
            ]
        )
    path = tmp_path / "samples.soft"
    path.write_text("\n".join(lines) + "\n")
    parsed = confirmation._parse_geo_metadata(path)
    expected = [{key: value for key, value in sample.items() if key != "role"} for sample in source]
    assert parsed == expected
    roles = confirmation._roles_from_official_metadata(parsed)
    assert {donor for donor, role in roles.items() if role == "calibration"} == set(
        confirmation.CALIBRATION_DONORS
    )


def test_primary_panel_has_16_unique_mappings_and_256_ordered_pairs():
    schema = _synthetic_schema()
    resolved = confirmation._resolve_panel(schema)
    assert len(resolved["rna"]) == len(set(resolved["rna"])) == 16
    assert len(resolved["adt"]) == len(set(resolved["adt"])) == 16
    assert len(confirmation.MARKERS) ** 2 == 256
    schema["id"].append(confirmation.ADT_IDS[0])
    schema["name"].append(confirmation.ADT_IDS[0])
    schema["feature_type"].append("Antibody Capture")
    with pytest.raises(PermissionError, match="not unique"):
        confirmation._resolve_panel(schema)


def test_configuration_grid_and_recipient_margins_are_frozen():
    with confirmation._configured_model():
        assert len(confirmation.base._primary_configs()) == 144
    rows, columns = confirmation._margins_from_positive_counts(list(range(16)))
    np.testing.assert_array_equal(rows[:, 0, 1], np.arange(16))
    assert np.all(columns[..., 0] == 256)
    assert np.all(columns[..., 1] == 256)


def test_feature_schema_reader_does_not_open_barcode_or_matrix_datasets(tmp_path: Path):
    path = tmp_path / "schema_only.h5"
    _write_h5(path, poison_matrix=True)
    schema = confirmation._feature_schema(path)
    assert schema == _synthetic_schema()
    assert confirmation._resolve_panel(schema)["rna"] == list(range(16))


def test_held_reducer_returns_only_rna_margins_and_discloses_combined_access(
    monkeypatch, tmp_path: Path
):
    path = tmp_path / "combined.h5"
    _write_h5(path)
    monkeypatch.setattr(confirmation, "CELL_BUDGET", 4)
    monkeypatch.setattr(confirmation, "MINIMUM_DETECTED_GENES", 0)
    monkeypatch.setattr(confirmation, "MAXIMUM_MITOCHONDRIAL_FRACTION", 1.0)
    monkeypatch.setattr(confirmation, "MINIMUM_INFORMATIVE_ENTITIES", 48)
    value = confirmation.held_rna_reducer_payload("D1", path)
    assert value["eligible"]
    assert value["rna_positive_counts"] == [2] * 16
    assert value["informative_marker_identities"] == list(confirmation.MARKERS)
    assert value["informative_marker_support_mask"] == [True] * 16
    assert not ({"adt", "barcodes", "selected_cells", "rna"} & set(value))
    assert value["audit"]["adt_count_elements_co_resident_in_decoded_csc"]
    assert value["audit"]["adt_values_returned_to_parent"] == 0
    assert value["audit"]["adt_values_serialized"] == 0


def test_held_prediction_input_is_invariant_to_adt_magnitude_and_sparsity(
    monkeypatch, tmp_path: Path
):
    first = tmp_path / "first.h5"
    second = tmp_path / "second.h5"
    _write_h5(first, adt_dense=np.ones((16, 4), dtype=np.int32))
    altered = np.zeros((16, 4), dtype=np.int32)
    altered[::2, ::2] = 1_000_000
    _write_h5(second, adt_dense=altered)
    monkeypatch.setattr(confirmation, "CELL_BUDGET", 4)
    monkeypatch.setattr(confirmation, "MINIMUM_DETECTED_GENES", 0)
    monkeypatch.setattr(confirmation, "MAXIMUM_MITOCHONDRIAL_FRACTION", 1.0)
    monkeypatch.setattr(confirmation, "MINIMUM_INFORMATIVE_ENTITIES", 48)
    one = confirmation.held_rna_reducer_payload("D1", first)
    two = confirmation.held_rna_reducer_payload("D1", second)
    assert one == two


def test_development_loader_never_requests_a_held_donor(monkeypatch):
    opened = []

    def fake_read(donor, path, *, return_adt):
        del path
        opened.append(donor)
        return {
            "audit": {},
            "filtered_barcodes": 1,
            "informative_marker_count": 16,
            "informative_marker_identities": list(confirmation.MARKERS),
            "informative_marker_support_mask": [True] * 16,
            "informative_ordered_pairs": 256,
            "qc_eligible_barcodes": 1,
            "selected_axis_sha256": donor,
            "return_adt": return_adt,
        }

    monkeypatch.setattr(confirmation, "_read_donor_h5", fake_read)
    monkeypatch.setattr(
        confirmation,
        "_sample_by_donor",
        lambda: {
            donor: {"filename": f"{donor}.h5"}
            for donor in (
                confirmation.CALIBRATION_DONORS
                + confirmation.PILOT_DONORS
                + confirmation.HELD_DONORS
            )
        },
    )
    confirmation._load_development_values()
    assert opened == list(
        confirmation.CALIBRATION_DONORS + confirmation.PILOT_DONORS
    )
    assert not (set(opened) & set(confirmation.HELD_DONORS))


def test_support_gate_requires_donors_and_both_age_strata():
    values = {
        donor: {
            "eligible": True,
            "informative_marker_count": 16,
            "informative_marker_identities": list(confirmation.MARKERS),
            "informative_marker_support_mask": [True] * 16,
            "informative_ordered_pairs": 256,
            "qc_eligible_barcodes": 512,
        }
        for donor in confirmation.PILOT_DONORS
    }
    assert confirmation._support_gate("pilot", confirmation.PILOT_DONORS, values)[
        "passes"
    ]
    values["R003"]["eligible"] = False
    values["R015"]["eligible"] = False
    assert not confirmation._support_gate(
        "pilot", confirmation.PILOT_DONORS, values
    )["passes"]


def test_primary_gate_uses_five_percent_bootstrap_sign_and_age_checks():
    primary = np.full(7, 0.5)
    comparator = np.full(7, 1.0)
    result = confirmation._comparison(
        list(confirmation.PILOT_DONORS),
        primary,
        comparator,
        gating=True,
        seed_offset=0,
        analysis_role="post_selection_pilot_promotion_diagnostic",
    )
    assert result["passes"]
    assert not result["inference_claimed"]
    assert "not inference" in result["statistical_interpretation"]
    assert result["donor_exact_sign_test"]["one_sided_p"] == 1 / 2**7
    assert set(result["immune_age_mean_differences"]) == {"Young", "Old"}


def test_classical_support_has_no_arbitrary_five_percent_gate():
    primary = np.full(7, 0.99)
    comparator = np.full(7, 1.0)
    result = confirmation._comparison(
        list(confirmation.PILOT_DONORS),
        primary,
        comparator,
        gating=False,
        seed_offset=10,
        analysis_role="post_selection_pilot_promotion_diagnostic",
    )
    assert result["passes"]
    assert set(result["checks"]) == {
        "lower_donor_equal_mean_deviance",
        "donor_bootstrap_upper_95_below_zero",
    }


def test_classical_head_to_head_reports_prespecified_refusals():
    losses = {
        "primary": np.full(7, 0.5),
        "common_effect_stratified_cmle": np.full(7, 1.0),
    }
    result = confirmation._classical_comparisons(
        list(confirmation.PILOT_DONORS),
        losses,
        {"pooled_saturated_poisson_interaction": "structural zero"},
        analysis_role="post_selection_pilot_promotion_diagnostic",
    )
    assert set(result) == set(confirmation.CLASSICAL_METHODS)
    assert result["common_effect_stratified_cmle"]["passes"]
    assert result["pooled_saturated_poisson_interaction"] == {
        "status": "REFUSED",
        "reason": "structural zero",
        "passes": False,
    }


def test_held_statistics_are_labeled_candidate_specific_inference_with_ties():
    primary = np.asarray([0.5] * 6 + [1.0])
    comparator = np.ones(7)
    result = confirmation._comparison(
        list(confirmation.PILOT_DONORS),
        primary,
        comparator,
        gating=True,
        seed_offset=0,
        analysis_role="held_confirmatory_inference",
    )
    assert result["inference_claimed"]
    assert result["retained_donor_count"] == 7
    assert result["donor_exact_sign_test"]["exact_ties"] == 1
    assert result["donor_exact_sign_test"]["one_sided_p"] == 1 / 2**6


def _external_source_bundle(
    calibration: list[str] | None = None,
    pilot: list[str] | None = None,
    *,
    refuse_classical: bool = False,
):
    if calibration is None:
        calibration = list(confirmation.CALIBRATION_DONORS)
    if pilot is None:
        pilot = list(confirmation.PILOT_DONORS)
    donors = calibration + pilot
    table = np.full((16, 16, 2, 2), 128, dtype=np.int64)
    records = {donor: {"tables": table} for donor in donors}
    values = {
        donor: {"informative_marker_support_mask": [True] * 16}
        for donor in donors
    }
    selection = {
        "primary": {"transport_multiplier": 1.0},
        "residual_candidates": [
            {
                "configuration": {
                    "family": "deviance",
                    "transport_multiplier": 0.75,
                },
                "mean_pilot_loss": 1.0,
            }
        ],
    }
    coordinate = np.zeros((16, 16)).tolist()
    source_models = {
        "primary": {
            "population_log_odds": coordinate,
            "fit_certificate": {"gradient_norm": 0.0},
        },
        "destroyed_link": {
            "population_log_odds": coordinate,
            "fit_certificate": {"gradient_norm": 0.0},
        },
    }
    if refuse_classical:
        source_classical = {
            "models": {},
            "refusals": {
                "common_effect_stratified_cmle": "synthetic CMLE refusal",
                "pooled_saturated_poisson_interaction": (
                    "synthetic structural-zero refusal"
                ),
            },
        }
    else:
        source_classical = {
            "models": {
                "common_effect_stratified_cmle": {
                    "population_log_odds": coordinate,
                    "fit_certificate": {
                        "gradient_norm": 0.0,
                        "condition_number": 1.0,
                    },
                    "transport_multiplier": 1.0,
                },
                "pooled_saturated_poisson_interaction": {
                    "population_log_odds": coordinate,
                    "no_structural_zero": True,
                    "transport_multiplier": 1.0,
                },
            },
            "refusals": {},
        }
    return confirmation._source_only_external_study_bundle(
        records,
        values,
        calibration,
        pilot,
        selection,
        source_models,
        source_classical,
    )


def test_external_source_bundle_has_five_methods_and_machine_support_certificate():
    bundle = _external_source_bundle()
    assert bundle["schema"] == "gse288020-independent-study-source-model/1.0"
    expected_methods = {
        "primary",
        "matched_deviance_residual",
        "destroyed_link",
        "common_effect_cmle",
        "pooled_saturated_poisson",
    }
    assert set(bundle["methods"]) == expected_methods
    assert set(bundle["declared_method_order"]) == expected_methods
    assert list(bundle["methods"]) == bundle["declared_method_order"]
    assert all(value["status"] == "VALID" for value in bundle["methods"].values())
    assert bundle["refit_axis"]["retained_mgus_donor_count"] == 14
    assert (
        len(bundle["source_support"]["informative_source_donors_per_ordered_pair"])
        == 256
    )
    assert bundle["numerical_certificate"]["core_passes"]
    assert bundle["numerical_certificate"]["classical_head_to_head_ready"]
    assert bundle["numerical_certificate"]["external_study_ready"]
    assert bundle["numerical_certificate"]["passes"]
    assert not bundle["selection_provenance"]["gse288020_mm_values_used"]
    assert not bundle["selection_provenance"]["gse309593_values_used"]


@pytest.mark.parametrize(
    ("calibration_count", "pilot_count"), ((6, 6), (6, 7), (7, 6))
)
def test_external_source_bundle_marks_attrited_internal_refit_ineligible(
    calibration_count: int, pilot_count: int
):
    bundle = _external_source_bundle(
        list(confirmation.CALIBRATION_DONORS[:calibration_count]),
        list(confirmation.PILOT_DONORS[:pilot_count]),
    )
    certificate = bundle["numerical_certificate"]
    assert bundle["refit_axis"]["retained_mgus_donor_count"] in {12, 13}
    assert not certificate["core_passes"]
    assert certificate["classical_head_to_head_ready"]
    assert not certificate["external_study_ready"]
    assert not certificate["passes"]


def test_external_source_bundle_keeps_five_slots_and_refuses_classical_access():
    bundle = _external_source_bundle(refuse_classical=True)
    assert len(bundle["methods"]) == 5
    assert bundle["methods"]["common_effect_cmle"]["status"] == "REFUSED"
    assert bundle["methods"]["pooled_saturated_poisson"]["status"] == "REFUSED"
    assert set(bundle["method_refusals"]) == {
        "common_effect_cmle",
        "pooled_saturated_poisson",
    }
    certificate = bundle["numerical_certificate"]
    assert certificate["core_passes"]
    assert not certificate["classical_head_to_head_ready"]
    assert not certificate["external_study_ready"]
    assert not certificate["passes"]


def test_pilot_failure_still_exports_source_only_model_and_forbids_gse288_held(
    monkeypatch,
):
    captured = {}
    monkeypatch.setattr(
        confirmation,
        "_read_json",
        lambda path: {"status": "PASS_BEFORE_BARCODE_OR_COUNT_ACCESS"},
    )
    monkeypatch.setattr(confirmation, "_validate_source_bytes", lambda: {})
    monkeypatch.setattr(confirmation, "_load_development_values", lambda: ({}, {}))
    monkeypatch.setattr(
        confirmation,
        "_support_gate",
        lambda role, donors, values: {
            "passes": True,
            "eligible_donors": list(donors),
        },
    )
    monkeypatch.setattr(confirmation, "_records", lambda donors, values: {})
    selection = {"primary": {}, "residual_candidates": []}
    losses = {
        "primary": np.ones(7),
        "best_residual": np.ones(7),
        "destroyed_link": np.ones(7),
    }
    monkeypatch.setattr(
        confirmation.base,
        "_select_on_pilot",
        lambda records, calibration, pilot: (selection, {}, losses),
    )
    classical_selection = {"methods": {}, "refusals": {}}
    monkeypatch.setattr(
        confirmation,
        "_select_classical",
        lambda records, calibration, pilot: (classical_selection, {}),
    )
    monkeypatch.setattr(
        confirmation,
        "_gate",
        lambda donors, losses, analysis_role: {"passes": False},
    )

    def fake_fit(records, donors, selected):
        captured["donors"] = list(donors)
        return {"frozen": True}

    monkeypatch.setattr(confirmation.model_core, "_fit_models", fake_fit)
    monkeypatch.setattr(
        confirmation,
        "_refit_classical",
        lambda records, donors, selected: {"models": {}, "refusals": {}},
    )
    monkeypatch.setattr(
        confirmation,
        "_source_only_external_study_bundle",
        lambda *args: {"schema": "gse288020-independent-study-source-model/1.0"},
    )
    payload = confirmation._development_body("a" * 40)
    assert payload["status"] == "TERMINAL_PILOT_PROMOTION_FAILURE"
    assert payload["source_only_external_study_model"]["schema"].endswith("/1.0")
    assert captured["donors"] == list(
        confirmation.CALIBRATION_DONORS + confirmation.PILOT_DONORS
    )
    assert "no GSE288020 held HDF5" in payload["terminal_rule"]


def test_pooled_poisson_interaction_is_saturated_log_odds_ratio():
    tables = np.asarray([[[[[10, 20], [30, 40]]]], [[[[5, 10], [15, 20]]]]])
    value = confirmation._pooled_loglinear_interaction(tables)
    pooled = tables.sum(axis=0)[0, 0]
    expected = math.log(pooled[0, 0] * pooled[1, 1] / (pooled[0, 1] * pooled[1, 0]))
    assert value[0, 0] == pytest.approx(expected)


def test_schema_access_record_has_real_timestamp_and_zero_count_access():
    access = json.loads(confirmation.DEFAULT_ACCESS.read_text())
    assert access["record_created_at_utc"] != "2026-08-28T00:00:00Z"
    assert set(access["numeric_assay_values_accessed"].values()) == {0}
    assert "/matrix/data" in access["hdf5_paths_not_opened"]
    preflight = json.loads(confirmation.DEFAULT_PREFLIGHT.read_text())
    assert preflight["status"] == "PASS_BEFORE_BARCODE_OR_COUNT_ACCESS"
    assert preflight["access_audit"]["numeric_assay_values_accessed"] == 0
    manifest = json.loads(confirmation.DEFAULT_MANIFEST.read_text())
    assert (
        preflight["source_bindings"]["official_geo_metadata_sha256"]
        == manifest["official_geo_metadata"]["sha256"]
    )
    assert "source_cache" not in confirmation.PROTOCOL_BINDINGS


def test_public_stage_tags_are_distinct():
    assert len(
        {
            confirmation.PROTOCOL_TAG,
            confirmation.DEVELOPMENT_TAG,
            confirmation.PREDICTION_TAG,
        }
    ) == 3


def test_candidate_search_and_state_estimand_are_disclosed_pre_outcome():
    disclosure = confirmation.CANDIDATE_SEARCH_DISCLOSURE
    assert disclosure["sequential_public_candidate_search"]
    assert not disclosure["familywise_adjustment_across_public_candidates"]
    assert disclosure["disclosed_before_gse288020_barcode_or_count_access"]
    assert [
        value["accession"]
        for value in disclosure["immediate_public_terminal_predecessors"]
    ] == ["GSE314416", "GSE158769", "GSE189050"]
    ledger = disclosure["earlier_public_campaign_ledger"]
    assert ledger["path"] in confirmation.PROTOCOL_BINDINGS
    assert confirmation._sha256(confirmation.ROOT / ledger["path"]) == ledger["sha256"]
    preflight = json.loads(confirmation.DEFAULT_PREFLIGHT.read_text())
    assert preflight["candidate_search_disclosure"] == disclosure
    estimand = confirmation._state_estimand()
    assert "conditional RNA-ADT log odds" in estimand["coupling_parameter"]
    assert "512" in estimand["cell_population"]


def test_protocol_binds_complete_transitive_runtime_and_environment():
    required = {
        ".gitignore",
        "experiments/confirm_gse288020_citeseq.py",
        "experiments/reduce_gse288020_held_rna.py",
        "experiments/confirm_gse158769_citeseq.py",
        "experiments/confirm_gse314416_citeseq.py",
        "docs/FINAL_PUBLIC_EVIDENCE_LEDGER.md",
        "mapreg/__init__.py",
        "mapreg/classical_residuals.py",
        "mapreg/coupling_fields.py",
        "mapreg/factorial_coupling.py",
        "mapreg/heterogeneity_adaptive_coupling.py",
        "mapreg/hierarchical_conditional_coupling.py",
        "mapreg/table_prediction.py",
        "requirements.txt",
        "pyproject.toml",
        "results/development/gse288020_runtime_environment_v1.json",
    }
    assert required <= set(confirmation.PROTOCOL_BINDINGS)


def test_runtime_fingerprint_is_exact_and_machine_checked():
    required = json.loads(confirmation.DEFAULT_RUNTIME_SPEC.read_text())[
        "required_runtime"
    ]
    assert required == {
        "python": {"implementation": "CPython", "version": "3.9.6"},
        "packages": {"numpy": "2.0.2", "scipy": "1.13.1", "h5py": "3.14.0"},
        "hdf5": {
            "runtime_version": "1.14.6",
            "runtime_version_tuple": [1, 14, 6],
            "built_against_version_tuple": [1, 14, 6],
            "h5py_api_version": "1.8",
        },
    }
    assert confirmation._require_runtime_environment() == required


@pytest.mark.parametrize("mutated", confirmation.PROTOCOL_BINDINGS)
def test_public_binding_rejects_every_changed_bound_file(
    mutated: str, monkeypatch, tmp_path: Path
):
    paths = confirmation.PROTOCOL_BINDINGS
    tagged = {relative: f"{relative}\n".encode() for relative in paths}
    for relative, value in tagged.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    commit = "a" * 40

    def fake_run(args, **kwargs):
        del kwargs
        if args[1] == "rev-list":
            return SimpleNamespace(stdout=commit + "\n")
        if args[1] == "ls-remote":
            return SimpleNamespace(stdout=f"{commit}\trefs/tags/frozen^{{}}\n")
        if args[1] == "show":
            relative = args[-1].split(":", 1)[1]
            return SimpleNamespace(stdout=tagged[relative])
        raise AssertionError(args)

    monkeypatch.setattr(confirmation, "ROOT", tmp_path)
    monkeypatch.setattr(confirmation.subprocess, "run", fake_run)
    assert confirmation._require_public_tag("frozen", paths) == commit
    (tmp_path / mutated).write_bytes(b"changed\n")
    with pytest.raises(PermissionError, match="differs from public tag"):
        confirmation._require_public_tag("frozen", paths)


def test_claimed_stage_records_started_before_work_and_terminalizes_exception(
    monkeypatch, tmp_path: Path
):
    output = tmp_path / "results/stage.json"
    attempt = tmp_path / "results/attempt.jsonl"
    monkeypatch.setattr(confirmation, "ROOT", tmp_path)
    runtime = confirmation._runtime_environment()

    def fail_after_claim():
        lines = attempt.read_text().splitlines()
        assert json.loads(lines[0])["event"] == "STARTED"
        raise RuntimeError("pre-source failure")

    payload = confirmation._run_claimed_stage(
        "development",
        output,
        output,
        attempt,
        "a" * 40,
        runtime,
        fail_after_claim,
    )
    assert payload["status"] == "TERMINAL_DEVELOPMENT_EXCEPTION"
    assert json.loads(output.read_text()) == payload
    assert [json.loads(line)["event"] for line in attempt.read_text().splitlines()] == [
        "STARTED",
        "FINISHED",
    ]
    with pytest.raises(PermissionError, match="already has"):
        confirmation._run_claimed_stage(
            "development",
            output,
            output,
            attempt,
            "a" * 40,
            runtime,
            fail_after_claim,
        )


def test_claimed_stage_replaces_a_body_written_success_with_one_terminal_exception(
    monkeypatch, tmp_path: Path
):
    output = tmp_path / "results/stage.json"
    attempt = tmp_path / "results/attempt.jsonl"
    monkeypatch.setattr(confirmation, "ROOT", tmp_path)
    runtime = confirmation._runtime_environment()

    def contradictory_body():
        confirmation._write_json_atomic_new(
            output, {"status": "CONTRADICTORY_BODY_SUCCESS"}
        )
        raise RuntimeError("failure after direct success output")

    payload = confirmation._run_claimed_stage(
        "development",
        output,
        output,
        attempt,
        "a" * 40,
        runtime,
        contradictory_body,
    )
    frozen = json.loads(output.read_text())
    ledger = [json.loads(line) for line in attempt.read_text().splitlines()]
    assert payload == frozen
    assert frozen["status"] == "TERMINAL_DEVELOPMENT_EXCEPTION"
    assert "failure after direct success output" in frozen["reason"]
    assert ledger[-1]["status"] == frozen["status"]
    assert ledger[-1]["output_sha256"] == confirmation._sha256(output)
    assert "CONTRADICTORY_BODY_SUCCESS" not in output.read_text()


def test_claimed_stage_is_the_only_assay_stage_output_writer():
    for body in (
        confirmation._development_body,
        confirmation._prediction_body,
        confirmation._score_body,
    ):
        source = inspect.getsource(body)
        assert "_write_json_atomic_new" not in source
        assert "output_path" not in source


def _prior_stage_fixture(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(confirmation, "ROOT", tmp_path)
    runtime = confirmation._runtime_environment()
    authorization = "a" * 40
    result = tmp_path / "results/development.json"
    result.parent.mkdir(parents=True)
    payload = {
        "authorization_commit": authorization,
        "runtime_environment": runtime,
        "stage": "development",
        "status": "PILOT_PROMOTION_PASS",
    }
    result.write_text(json.dumps(payload, sort_keys=True) + "\n")
    output = confirmation._relative(result)
    records = [
        {
            "authorization_commit": authorization,
            "created_at_utc": "2026-08-29T00:00:00Z",
            "event": "STARTED",
            "output": output,
            "runtime_environment": runtime,
            "stage": "development",
        },
        {
            "authorization_commit": authorization,
            "created_at_utc": "2026-08-29T00:00:01Z",
            "event": "FINISHED",
            "output": output,
            "output_sha256": confirmation._sha256(result),
            "runtime_environment": runtime,
            "stage": "development",
            "status": "PILOT_PROMOTION_PASS",
        },
    ]
    attempt = tmp_path / "results/development_attempt.jsonl"
    return attempt, result, authorization, runtime, records, payload


@pytest.mark.parametrize(
    "mismatch",
    (
        "record_count",
        "extra_field",
        "started_event",
        "finished_event",
        "started_authorization",
        "finished_authorization",
        "result_authorization",
        "started_stage",
        "finished_stage",
        "result_stage",
        "started_output_path",
        "finished_output_path",
        "output_hash",
        "finished_status",
        "result_status",
        "started_runtime",
        "finished_runtime",
        "result_runtime",
    ),
)
def test_prior_stage_validator_rejects_every_semantic_mismatch(
    mismatch: str, monkeypatch, tmp_path: Path
):
    attempt, result, authorization, runtime, records, payload = _prior_stage_fixture(
        tmp_path, monkeypatch
    )
    if mismatch == "record_count":
        records.append(dict(records[-1]))
    elif mismatch == "extra_field":
        records[0]["undeclared"] = True
    elif mismatch == "started_event":
        records[0]["event"] = "FINISHED"
    elif mismatch == "finished_event":
        records[1]["event"] = "STARTED"
    elif mismatch == "started_authorization":
        records[0]["authorization_commit"] = "b" * 40
    elif mismatch == "finished_authorization":
        records[1]["authorization_commit"] = "b" * 40
    elif mismatch == "result_authorization":
        payload["authorization_commit"] = "b" * 40
    elif mismatch == "started_stage":
        records[0]["stage"] = "prediction"
    elif mismatch == "finished_stage":
        records[1]["stage"] = "prediction"
    elif mismatch == "result_stage":
        payload["stage"] = "prediction"
    elif mismatch == "started_output_path":
        records[0]["output"] = "results/other.json"
    elif mismatch == "finished_output_path":
        records[1]["output"] = "results/other.json"
    elif mismatch == "output_hash":
        records[1]["output_sha256"] = "0" * 64
    elif mismatch == "finished_status":
        records[1]["status"] = "TERMINAL_PILOT_PROMOTION_FAILURE"
    elif mismatch == "result_status":
        payload["status"] = "TERMINAL_PILOT_PROMOTION_FAILURE"
        records[1]["status"] = payload["status"]
    elif mismatch == "started_runtime":
        records[0]["runtime_environment"] = {}
    elif mismatch == "finished_runtime":
        records[1]["runtime_environment"] = {}
    else:
        payload["runtime_environment"] = {}
    if mismatch.startswith("result_"):
        result.write_text(json.dumps(payload, sort_keys=True) + "\n")
        records[1]["output_sha256"] = confirmation._sha256(result)
    attempt.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    with pytest.raises(PermissionError):
        confirmation._validate_prior_stage(
            "development",
            attempt,
            result,
            authorization,
            runtime,
            "PILOT_PROMOTION_PASS",
        )


def test_prior_stage_validator_accepts_one_exact_started_finished_chain(
    monkeypatch, tmp_path: Path
):
    attempt, result, authorization, runtime, records, payload = _prior_stage_fixture(
        tmp_path, monkeypatch
    )
    attempt.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    assert (
        confirmation._validate_prior_stage(
            "development",
            attempt,
            result,
            authorization,
            runtime,
            "PILOT_PROMOTION_PASS",
        )
        == payload
    )


def test_runtime_mismatch_refuses_before_attempt_or_tag_check(monkeypatch, tmp_path: Path):
    output = tmp_path / "development.json"
    attempt = tmp_path / "attempt.jsonl"
    monkeypatch.setattr(confirmation, "DEFAULT_DEVELOPMENT", output)
    monkeypatch.setattr(confirmation, "DEFAULT_DEVELOPMENT_ATTEMPT", attempt)
    monkeypatch.setattr(
        confirmation,
        "_require_runtime_environment",
        lambda: (_ for _ in ()).throw(PermissionError("runtime mismatch")),
    )
    monkeypatch.setattr(
        confirmation,
        "_require_public_tag",
        lambda *args, **kwargs: pytest.fail("tag check must not run"),
    )
    with pytest.raises(PermissionError, match="runtime mismatch"):
        confirmation.run_development(output)
    assert not output.exists()
    assert not attempt.exists()


def test_invalid_prior_chain_refuses_before_downstream_attempt(
    monkeypatch, tmp_path: Path
):
    output = tmp_path / "prediction.json"
    attempt = tmp_path / "prediction_attempt.jsonl"
    monkeypatch.setattr(confirmation, "DEFAULT_PREDICTION", output)
    monkeypatch.setattr(confirmation, "DEFAULT_PREDICTION_ATTEMPT", attempt)
    monkeypatch.setattr(
        confirmation, "_require_runtime_environment", confirmation._runtime_environment
    )
    monkeypatch.setattr(
        confirmation,
        "_require_public_tag",
        lambda tag, paths: "a" * 40 if tag == confirmation.PROTOCOL_TAG else "b" * 40,
    )
    monkeypatch.setattr(
        confirmation,
        "_validate_prior_stage",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            PermissionError("invalid prior chain")
        ),
    )
    monkeypatch.setattr(
        confirmation,
        "_run_claimed_stage",
        lambda *args, **kwargs: pytest.fail("attempt must not be claimed"),
    )
    with pytest.raises(PermissionError, match="invalid prior chain"):
        confirmation.run_prediction(output)
    assert not attempt.exists()


@pytest.mark.parametrize(
    "runner_name",
    ("run_development", "run_prediction", "run_score"),
)
def test_stage_rejects_custom_output_before_public_tag_check(
    runner_name: str, monkeypatch, tmp_path: Path
):
    monkeypatch.setattr(
        confirmation,
        "_require_public_tag",
        lambda *args, **kwargs: pytest.fail("authorization check should not run"),
    )
    with pytest.raises(PermissionError, match="stage output must be"):
        getattr(confirmation, runner_name)(tmp_path / "custom.json")


@pytest.mark.parametrize(
    ("runner_name", "output_name", "attempt_name"),
    (
        ("run_development", "DEFAULT_DEVELOPMENT", "DEFAULT_DEVELOPMENT_ATTEMPT"),
        ("run_prediction", "DEFAULT_PREDICTION", "DEFAULT_PREDICTION_ATTEMPT"),
        ("run_score", "DEFAULT_SCORE", "DEFAULT_SCORE_ATTEMPT"),
    ),
)
def test_existing_attempt_blocks_stage_before_authorization(
    runner_name: str,
    output_name: str,
    attempt_name: str,
    monkeypatch,
    tmp_path: Path,
):
    output = tmp_path / "stage.json"
    attempt = tmp_path / "attempt.jsonl"
    attempt.write_text('{"event":"STARTED"}\n')
    monkeypatch.setattr(confirmation, output_name, output)
    monkeypatch.setattr(confirmation, attempt_name, attempt)
    monkeypatch.setattr(
        confirmation,
        "_require_public_tag",
        lambda *args, **kwargs: pytest.fail("authorization check should not run"),
    )
    with pytest.raises(PermissionError, match="already has"):
        getattr(confirmation, runner_name)(output)
