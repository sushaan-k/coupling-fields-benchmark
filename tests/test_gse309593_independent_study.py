from __future__ import annotations

import copy
import csv
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from experiments import confirm_gse309593_independent_study as subject


def _source_model() -> dict[str, object]:
    mappings = subject._expected_source_mappings()
    size = len(mappings)
    zeros = np.zeros((size, size)).tolist()
    primary_configuration = {
        "graph_neighbors": 1,
        "heterogeneity_penalty": 1.0,
        "ridge_penalty": 0.1,
        "graph_penalty": 0.3,
        "transport_multiplier": 1.0,
    }
    methods = {
        name: {
            "status": "VALID",
            **(
                {
                    "pooled_coordinate": zeros,
                    "configuration": {
                        "family": "deviance",
                        "transport_multiplier": 1.0,
                    },
                }
                if name == "matched_deviance_residual"
                else {
                    "population_log_odds": zeros,
                    **(
                        {"transport_multiplier": 1.0}
                        if name in subject.CLASSICAL_METHODS
                        else {
                            "configuration": dict(primary_configuration),
                            "fit_certificate": {"gradient_norm": 0.0},
                        }
                    ),
                }
            ),
        }
        for name in subject.REQUIRED_METHODS
    }
    methods["common_effect_cmle"]["fit_certificate"] = {
        "gradient_norm": 0.0,
        "condition_number": 1.0,
    }
    methods["pooled_saturated_poisson"]["no_structural_zero"] = True
    support = np.full(size * size, 14, dtype=np.int64)
    return {
        "schema": "gse288020-independent-study-source-model/1.0",
        "scope": (
            "fixed GSE288020 MGUS source model for a separately frozen external "
            "study; not a GSE288020 held-MM result"
        ),
        "public_export_location": (
            "results/development/gse288020_development_v1.json#/"
            "source_only_external_study_model"
        ),
        "source_accession": "GSE288020",
        "source_condition": "MGUS",
        "selection_provenance": {
            "split_salt": "GSE288020-MGUS-SPLIT-v1",
            "designated_calibration_donors": list(subject.EXPECTED_SOURCE_CALIBRATION),
            "designated_pilot_donors": list(subject.EXPECTED_SOURCE_PILOT),
            "retained_calibration_donors": list(subject.EXPECTED_SOURCE_CALIBRATION),
            "retained_pilot_donors": list(subject.EXPECTED_SOURCE_PILOT),
            "primary_configuration": dict(primary_configuration),
            "deviance_residual_configuration": {
                "family": "deviance",
                "transport_multiplier": 1.0,
            },
            "gse288020_mm_values_used": False,
            "gse309593_values_used": False,
        },
        "refit_axis": {
            "designated_mgus_donors": list(subject.EXPECTED_SOURCE_SUBJECTS),
            "retained_mgus_donors": list(subject.EXPECTED_SOURCE_SUBJECTS),
            "retained_mgus_donor_count": 14,
        },
        "external_study_eligibility": {
            "internal_gse288020_refit_allows_12_to_14_retained_mgus_donors": True,
            "external_study_requires_all_14_designated_mgus_donors": True,
            "external_study_requires_7_calibration_and_7_pilot_donors": True,
            "external_study_requires_all_five_methods_valid": True,
            "target_assay_access_requires_external_study_ready": True,
        },
        "declared_method_order": list(subject.REQUIRED_METHODS),
        "source_support": {
            "marker_order": [row["adt_target"] for row in mappings],
            "ordered_pair_axis": "RNA-major, then ADT marker order",
            "informative_source_donors_per_marker": [14] * size,
            "informative_source_donors_per_ordered_pair": support.tolist(),
            "informative_source_donors_per_ordered_pair_sha256": (
                subject._array_sha256(support)
            ),
            "retained_ordered_pair_support_mask": [True] * (size * size),
            "retained_ordered_pair_count": size * size,
        },
        "numerical_certificate": {
            "finite_coordinate_checks": {
                name: True for name in subject.REQUIRED_METHODS
            },
            "core_checks": {
                "exactly_14_retained_mgus_donors": True,
                "exactly_7_retained_calibration_and_7_retained_pilot": True,
                "all_256_pairs_have_at_least_2_informative_source_donors": True,
                "primary_residual_destroyed_coordinates_are_finite": True,
                "primary_optimizer_certificate_pass": True,
                "destroyed_optimizer_certificate_pass": True,
                "no_mm_or_gse309593_values_used": True,
            },
            "classical_checks": {
                "both_classical_coordinates_are_finite": True,
                "common_cmle_gradient_and_condition_certificate_pass": True,
                "pooled_poisson_no_structural_zero_pass": True,
            },
            "core_passes": True,
            "classical_head_to_head_ready": True,
            "external_study_ready": True,
            "passes": True,
        },
        "methods": methods,
        "method_refusals": {},
    }


def _supported_candidates() -> list[dict[str, str]]:
    return subject._source_supported_candidates(subject._candidate(), _source_model())


def _write_tenx_h5(
    path: Path,
    symbols: list[str],
    *,
    noninteger: bool = False,
    include_antibody: bool = False,
    background_genes: int = 400,
) -> None:
    cells = 600
    background = [f"GENE{index}" for index in range(background_genes)]
    names = symbols + background + (["CD4"] if include_antibody else [])
    types = ["Gene Expression"] * (len(symbols) + len(background)) + (
        ["Antibody Capture"] if include_antibody else []
    )
    indices = []
    data = []
    indptr = [0]
    for cell in range(cells):
        for feature in range(len(names)):
            if (cell + feature) % 3:
                indices.append(feature)
                data.append(0.5 if noninteger else 1 + (cell + feature) % 4)
        indptr.append(len(indices))
    with h5py.File(path, "w") as handle:
        matrix = handle.create_group("matrix")
        matrix.create_dataset(
            "barcodes",
            data=np.asarray([f"CELL-{index}".encode() for index in range(cells)]),
        )
        features = matrix.create_group("features")
        features.create_dataset(
            "name", data=np.asarray([value.encode() for value in names])
        )
        features.create_dataset(
            "feature_type", data=np.asarray([value.encode() for value in types])
        )
        matrix.create_dataset("data", data=np.asarray(data))
        matrix.create_dataset("indices", data=np.asarray(indices, dtype=np.int64))
        matrix.create_dataset("indptr", data=np.asarray(indptr, dtype=np.int64))
        matrix.create_dataset(
            "shape", data=np.asarray([len(names), cells], dtype=np.int64)
        )


def _write_adt_csv(
    path: Path,
    selected: list[str],
    markers: list[str],
    *,
    constant: bool = False,
    constant_markers: set[str] | None = None,
) -> None:
    fixed = set(markers) if constant else set(constant_markers or ())
    with gzip.open(path, "wt", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["cell"] + markers)
        for index, identifier in enumerate(selected):
            values = [
                0 if marker in fixed else (index + marker_index) % 7
                for marker_index, marker in enumerate(markers)
            ]
            writer.writerow([identifier] + values)


def test_public_candidate_is_preserved_and_amendment_is_source_only():
    assert subject._sha256(subject.DEFAULT_CANDIDATE) == (
        "07c1979bcee8009db14265a2360f2f46527f674ac5835cf54f098f2f10bdc3e9"
    )
    amendment = json.loads(subject.DEFAULT_AMENDMENT.read_text())
    correction = amendment["source_split_correction"]
    assert correction["corrected_calibration_subjects"] == list(
        subject.EXPECTED_SOURCE_CALIBRATION
    )
    assert correction["corrected_pilot_subjects"] == list(subject.EXPECTED_SOURCE_PILOT)
    assert "14 MGUS" in correction["corrected_final_refit_rule"]
    assert "never enter" in correction["corrected_final_refit_rule"]
    assert set(amendment["target_access_attestation"].values()) == {0}


def test_source_model_contract_uses_only_14_mgus_and_all_five_methods():
    model = _source_model()
    subject._validate_source_model(model)
    assert len(model["refit_axis"]["retained_mgus_donors"]) == 14
    assert set(model["methods"]) == set(subject.REQUIRED_METHODS)


@pytest.mark.parametrize(
    "lineage_mutation",
    [
        None,
        "attempt_event",
        "attempt_runtime",
        "result_status",
        "protocol_commit",
        "outer_primary",
        "outer_classical",
        "outer_donors",
        "outer_refit",
    ],
)
def test_source_authorization_validates_the_canonical_nested_model_and_lineage(
    tmp_path: Path, monkeypatch, lineage_mutation: str | None
):
    protocol_path = tmp_path / "protocol.json"
    source_path = tmp_path / "results/development/gse288020_development_v1.json"
    attempt_path = (
        tmp_path / "results/development/gse288020_development_attempt_v1.jsonl"
    )
    runtime_path = (
        tmp_path / "results/development/gse288020_runtime_environment_v1.json"
    )
    authorization_path = tmp_path / "source_authorization.json"
    source_path.parent.mkdir(parents=True)
    model = _source_model()
    protocol = {
        "source_model_contract": {
            "canonical_artifact_path": (
                "results/development/gse288020_development_v1.json"
            ),
            "canonical_attempt_path": (
                "results/development/gse288020_development_attempt_v1.jsonl"
            ),
            "canonical_runtime_path": (
                "results/development/gse288020_runtime_environment_v1.json"
            ),
            "json_pointer": "/source_only_external_study_model",
        },
        "expected_source_marker_mappings": subject._expected_source_mappings(),
    }
    protocol_path.write_text(json.dumps(protocol))
    runtime = subject._runtime_environment()
    runtime_path.write_text(
        json.dumps(
            {
                "schema": "gse288020-runtime-environment/1.0",
                "required_runtime": runtime,
            }
        )
    )
    protocol_commit = "a" * 40
    development_commit = "b" * 40
    outer_core_models = {}
    for name in ("primary", "destroyed_link"):
        outer_core_models[name] = copy.deepcopy(model["methods"][name])
        outer_core_models[name].pop("status")
    outer_classical_models = {}
    for nested_name, outer_name in {
        "common_effect_cmle": "common_effect_stratified_cmle",
        "pooled_saturated_poisson": "pooled_saturated_poisson_interaction",
    }.items():
        outer_classical_models[outer_name] = copy.deepcopy(
            model["methods"][nested_name]
        )
        outer_classical_models[outer_name].pop("status")
    development = {
        "schema": "gse288020-development/1.0",
        "status": "TERMINAL_PILOT_PROMOTION_FAILURE",
        "protocol_commit": protocol_commit,
        "authorization_commit": protocol_commit,
        "stage": "development",
        "runtime_environment": runtime,
        "calibration_donors": list(subject.EXPECTED_SOURCE_CALIBRATION),
        "pilot_donors": list(subject.EXPECTED_SOURCE_PILOT),
        "retained_development_donors": list(subject.EXPECTED_SOURCE_SUBJECTS),
        "retained_development_donor_count": 14,
        "support": {
            "calibration": {
                "eligible_donors": list(subject.EXPECTED_SOURCE_CALIBRATION)
            },
            "pilot": {"eligible_donors": list(subject.EXPECTED_SOURCE_PILOT)},
        },
        "selection": {
            "primary": copy.deepcopy(
                model["selection_provenance"]["primary_configuration"]
            ),
            "residual_candidates": [
                {
                    "configuration": copy.deepcopy(
                        model["selection_provenance"][
                            "deviance_residual_configuration"
                        ]
                    ),
                    "mean_pilot_loss": 1.0,
                }
            ],
        },
        "classical_selection": {
            "methods": {
                "common_effect_stratified_cmle": {"transport_multiplier": 1.0},
                "pooled_saturated_poisson_interaction": {
                    "transport_multiplier": 1.0
                },
            },
            "refusals": {},
        },
        "gse288020_held_prediction_models": {
            "primary_residual_destroyed_models": outer_core_models,
            "classical_models": {"models": outer_classical_models},
        },
        "source_only_external_study_model": model,
    }
    attempt_records = [
        {
            "authorization_commit": protocol_commit,
            "created_at_utc": "2026-08-29T00:00:00Z",
            "event": "STARTED",
            "output": "results/development/gse288020_development_v1.json",
            "runtime_environment": runtime,
            "stage": "development",
        },
        {
            "authorization_commit": protocol_commit,
            "created_at_utc": "2026-08-29T00:01:00Z",
            "event": "FINISHED",
            "output": "results/development/gse288020_development_v1.json",
            "output_sha256": None,
            "runtime_environment": runtime,
            "stage": "development",
            "status": "TERMINAL_PILOT_PROMOTION_FAILURE",
        },
    ]
    if lineage_mutation == "attempt_event":
        attempt_records[0]["event"] = "FINISHED"
    elif lineage_mutation == "attempt_runtime":
        attempt_records[0]["runtime_environment"] = {"python": "different"}
    elif lineage_mutation == "result_status":
        development["status"] = "UNREGISTERED_STATUS"
        attempt_records[1]["status"] = "UNREGISTERED_STATUS"
    elif lineage_mutation == "protocol_commit":
        development["protocol_commit"] = "d" * 40
    elif lineage_mutation == "outer_primary":
        development["selection"]["primary"] = {
            **development["selection"]["primary"],
            "transport_multiplier": 0.5,
        }
    elif lineage_mutation == "outer_classical":
        development["classical_selection"]["methods"][
            "common_effect_stratified_cmle"
        ]["transport_multiplier"] = 0.5
    elif lineage_mutation == "outer_donors":
        development["calibration_donors"][0] = "R003"
    elif lineage_mutation == "outer_refit":
        development["gse288020_held_prediction_models"][
            "primary_residual_destroyed_models"
        ]["primary"]["configuration"]["transport_multiplier"] = 0.5
    source_path.write_text(json.dumps(development, allow_nan=False))
    attempt_records[1]["output_sha256"] = subject._sha256(source_path)
    attempt_path.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in attempt_records)
        + "\n"
    )
    authorization = {
        "schema": "gse309593-independent-study-source-authorization/1.0",
        "status": "SOURCE_MODEL_AND_RECIPIENT_RNA_ACCESS_AUTHORIZED",
        "recipient_rna_access_authorized": True,
        "gse309593_assay_identifier_or_barcode_access_before_authorization": 0,
        "gse288020_mm_diagnostic_outcome_is_not_an_access_gate": True,
        "gse288020_mm_internal_test_outcome_values_used": 0,
        "gse288020_protocol_tag": subject.GSE288020_PROTOCOL_TAG,
        "gse288020_protocol_commit": protocol_commit,
        "gse288020_development_tag": subject.GSE288020_DEVELOPMENT_TAG,
        "gse288020_development_commit": development_commit,
        "source_development_path": (
            "results/development/gse288020_development_v1.json"
        ),
        "source_development_sha256": subject._sha256(source_path),
        "source_development_bytes": source_path.stat().st_size,
        "source_development_attempt_sha256": subject._sha256(attempt_path),
        "source_development_attempt_bytes": attempt_path.stat().st_size,
        "source_runtime_sha256": subject._sha256(runtime_path),
        "source_runtime_bytes": runtime_path.stat().st_size,
        "source_model_sha256": subject._canonical_json_sha256(model),
        "protocol_sha256": subject._sha256(protocol_path),
        "candidate_designation_sha256": subject._sha256(subject.DEFAULT_CANDIDATE),
        "source_split_amendment_sha256": subject._sha256(subject.DEFAULT_AMENDMENT),
        "runner_sha256": subject._sha256(Path(subject.__file__)),
        "transitive_bindings": {},
    }
    authorization_path.write_text(json.dumps(authorization))
    monkeypatch.setattr(subject, "ROOT", tmp_path)
    monkeypatch.setattr(subject, "DEFAULT_PROTOCOL", protocol_path)
    monkeypatch.setattr(subject, "DEFAULT_SOURCE_AUTHORIZATION", authorization_path)
    monkeypatch.setattr(subject, "_binding_hashes", lambda: {})
    monkeypatch.setattr(subject, "_require_remote_tag_commit", lambda *_: None)
    commits = {
        subject.GSE288020_PROTOCOL_TAG: protocol_commit,
        subject.GSE288020_DEVELOPMENT_TAG: development_commit,
        subject.SOURCE_AUTHORIZATION_TAG: "c" * 40,
    }
    monkeypatch.setattr(subject, "_require_public_tag", lambda tag, *_: commits[tag])
    if lineage_mutation is not None:
        with pytest.raises(PermissionError):
            subject._require_source_authorization()
        return
    loaded_authorization, loaded_model, commit = subject._require_source_authorization()
    assert (
        loaded_authorization["source_model_sha256"]
        == authorization["source_model_sha256"]
    )
    assert loaded_model == model
    assert commit == "c" * 40


def test_classical_refusal_blocks_target_study_access():
    model = _source_model()
    model["methods"]["pooled_saturated_poisson"] = {
        "status": "REFUSED",
        "refusal": {
            "code": "POOLED_SATURATED_POISSON_REFUSED",
            "reason": "pooled saturated Poisson interaction has a structural zero",
        },
    }
    model["method_refusals"]["pooled_saturated_poisson"] = {
        "code": "POOLED_SATURATED_POISSON_REFUSED",
        "reason": "pooled saturated Poisson interaction has a structural zero",
    }
    model["numerical_certificate"]["finite_coordinate_checks"][
        "pooled_saturated_poisson"
    ] = False
    model["numerical_certificate"]["classical_checks"][
        "both_classical_coordinates_are_finite"
    ] = False
    model["numerical_certificate"]["classical_checks"][
        "pooled_poisson_no_structural_zero_pass"
    ] = False
    model["numerical_certificate"]["classical_head_to_head_ready"] = False
    model["numerical_certificate"]["external_study_ready"] = False
    model["numerical_certificate"]["passes"] = False
    with pytest.raises(PermissionError, match="external-study numerical certificate"):
        subject._validate_source_model(model)


@pytest.mark.parametrize(
    "mutation",
    [
        "append_mm",
        "mm_used",
        "target_used",
        "wrong_split",
        "wrong_config",
        "missing_comparator",
    ],
)
def test_source_model_contract_rejects_outcome_leakage_and_axis_drift(mutation: str):
    model = _source_model()
    if mutation == "append_mm":
        model["refit_axis"]["retained_mgus_donors"].append("E2228")
    elif mutation == "mm_used":
        model["selection_provenance"]["gse288020_mm_values_used"] = True
    elif mutation == "target_used":
        model["selection_provenance"]["gse309593_values_used"] = True
    elif mutation == "wrong_split":
        model["selection_provenance"]["retained_calibration_donors"][0] = "R003"
    elif mutation == "wrong_config":
        model["selection_provenance"]["primary_configuration"][
            "transport_multiplier"
        ] = 0.6
    else:
        del model["methods"]["pooled_saturated_poisson"]
    with pytest.raises(PermissionError):
        subject._validate_source_model(model)


def test_exact_panel_intersection_is_ordered_and_handles_cross_study_adt_labels():
    candidates = _supported_candidates()
    rna = [row["rna_symbol"] for row in candidates]
    adt = [row["adt_target"] for row in candidates]
    panel = subject._final_panel(subject._candidate(), _source_model(), rna, adt)
    assert len(panel) == 11
    assert [row["rna_symbol"] for row in panel] == rna
    cd8 = next(row for row in panel if row["rna_symbol"] == "CD8A")
    assert cd8["source_adt_target"] == "CD8a"
    assert cd8["recipient_adt_target"] == "CD8"


def test_cell_selection_is_deterministic_and_uses_no_values():
    identifiers = [f"CELL-{index}" for index in range(600)]
    first_indices, first_ids = subject._selected_cells("FH1001", identifiers)
    second_indices, second_ids = subject._selected_cells("FH1001", identifiers)
    assert first_indices == second_indices
    assert first_ids == second_ids
    assert len(first_ids) == 512 == len(set(first_ids))


def test_tenx_rna_reducer_resolves_gene_expression_and_never_opens_adt(tmp_path: Path):
    symbols = [row["rna_symbol"] for row in _supported_candidates()]
    path = tmp_path / "rna.h5"
    _write_tenx_h5(path, symbols)
    reduced = subject._reduce_rna_h5(path, "FH1001", symbols)
    assert reduced["available_rna_symbols"] == symbols
    assert len(reduced["selected_ids"]) == 512
    assert set(reduced["states"]) == set(symbols)
    assert reduced["audit"]["format"] == "10x_feature_by_cell_csc"
    assert reduced["audit"]["adt_files_opened"] == 0
    assert reduced["audit"]["adt_values_read"] == 0


def test_rna_reducer_refuses_noninteger_matrix(tmp_path: Path):
    symbols = [row["rna_symbol"] for row in _supported_candidates()]
    path = tmp_path / "normalized.h5"
    _write_tenx_h5(path, symbols, noninteger=True)
    with pytest.raises(subject.ProtocolRefusal, match="RNA_MATRIX_IS_NOT_RAW_COUNTS"):
        subject._reduce_rna_h5(path, "FH1001", symbols)


def test_rna_reducer_refuses_combined_assay_matrix_before_decoding_values(
    tmp_path: Path,
):
    symbols = [row["rna_symbol"] for row in _supported_candidates()]
    path = tmp_path / "combined.h5"
    _write_tenx_h5(path, symbols, include_antibody=True)
    with pytest.raises(
        subject.ProtocolRefusal, match="RNA_H5_CONTAINS_NON_RNA_MATRIX_VALUES"
    ):
        subject._reduce_rna_h5(path, "FH1001", symbols)


def test_rna_reducer_applies_the_source_cell_qc_before_selection(tmp_path: Path):
    symbols = [row["rna_symbol"] for row in _supported_candidates()]
    path = tmp_path / "low_complexity.h5"
    _write_tenx_h5(path, symbols, background_genes=100)
    with pytest.raises(subject.ProtocolRefusal, match="RNA_QC_SUPPORT_BELOW_512"):
        subject._reduce_rna_h5(path, "FH1001", symbols)


def test_rna_reducer_refuses_invalid_sparse_terminal_pointer(tmp_path: Path):
    symbols = [row["rna_symbol"] for row in _supported_candidates()]
    path = tmp_path / "invalid_sparse.h5"
    _write_tenx_h5(path, symbols)
    with h5py.File(path, "r+") as handle:
        pointer = handle["matrix/indptr"]
        pointer[-1] = int(pointer[-1]) - 1
    with pytest.raises(subject.ProtocolRefusal, match="H5_SPARSE_STRUCTURE_INVALID"):
        subject._reduce_rna_h5(path, "FH1001", symbols)


@pytest.mark.parametrize("mutation", ["fractional", "duplicate", "out_of_range"])
def test_rna_reducer_refuses_invalid_sparse_indices(tmp_path: Path, mutation: str):
    symbols = [row["rna_symbol"] for row in _supported_candidates()]
    path = tmp_path / f"invalid_sparse_{mutation}.h5"
    _write_tenx_h5(path, symbols)
    with h5py.File(path, "r+") as handle:
        matrix = handle["matrix"]
        indices = np.asarray(matrix["indices"][()])
        if mutation == "fractional":
            indices = indices.astype(float)
            indices[0] += 0.5
            del matrix["indices"]
            matrix.create_dataset("indices", data=indices)
        elif mutation == "duplicate":
            indices[1] = indices[0]
            matrix["indices"][:] = indices
        else:
            indices[0] = int(matrix["shape"][0])
            matrix["indices"][:] = indices
    with pytest.raises(
        subject.ProtocolRefusal,
        match="H5_SPARSE_(STRUCTURE_INVALID|DUPLICATE_INDEX|INDEX_OUT_OF_RANGE)",
    ):
        subject._reduce_rna_h5(path, "FH1001", symbols)


def test_anndata_reducer_refuses_multimodal_matrix_before_numeric_qc(tmp_path: Path):
    symbols = [row["rna_symbol"] for row in _supported_candidates()]
    path = tmp_path / "combined_anndata.h5"
    features = symbols + [f"GENE{index}" for index in range(400)] + ["CD4-ADT"]
    with h5py.File(path, "w") as handle:
        obs = handle.create_group("obs")
        obs.create_dataset(
            "_index",
            data=np.asarray([f"CELL-{index}".encode() for index in range(600)]),
        )
        var = handle.create_group("var")
        var.create_dataset(
            "_index", data=np.asarray([value.encode() for value in features])
        )
        var.create_dataset(
            "feature_types",
            data=np.asarray(
                [b"Gene Expression"] * (len(features) - 1) + [b"Antibody Capture"]
            ),
        )
        handle.create_dataset("X", data=np.zeros((600, len(features)), dtype=np.int64))
    with pytest.raises(
        subject.ProtocolRefusal, match="RNA_H5_CONTAINS_NON_RNA_MATRIX_VALUES"
    ):
        subject._reduce_rna_h5(path, "FH1001", symbols)


def test_adt_reducer_requires_exact_rna_selected_axis_and_has_fixed_tie_margin(
    tmp_path: Path,
):
    selected = [f"CELL-{index}" for index in range(512)]
    markers = [row["adt_target"] for row in _supported_candidates()]
    path = tmp_path / "adt.csv.gz"
    _write_adt_csv(path, selected, markers)
    reduced = subject._read_adt_csv(path, "FH1001", selected, markers)
    assert reduced["available_adt_targets"] == markers
    assert all(int(values.sum()) == 256 for values in reduced["states"].values())
    assert reduced["audit"]["rna_state_files_opened"] == 0
    assert reduced["audit"]["rna_state_values_read"] == 0


def test_adt_reducer_forbids_missing_selected_identifier_without_fallback(
    tmp_path: Path,
):
    selected = [f"CELL-{index}" for index in range(512)]
    markers = [row["adt_target"] for row in _supported_candidates()]
    path = tmp_path / "adt.csv.gz"
    _write_adt_csv(path, selected[:-1], markers)
    with pytest.raises(
        subject.ProtocolRefusal, match="ADT_SELECTED_IDENTIFIER_MISSING"
    ):
        subject._read_adt_csv(path, "FH1001", selected, markers)


def test_adt_reducer_excludes_a_salt_defined_constant_median_split(tmp_path: Path):
    selected = [f"CELL-{index}" for index in range(512)]
    markers = [row["adt_target"] for row in _supported_candidates()]
    path = tmp_path / "constant_adt.csv.gz"
    _write_adt_csv(path, selected, markers, constant=True)
    reduced = subject._read_adt_csv(path, "FH1001", selected, markers)
    assert set(reduced["marker_support"].values()) == {False}
    assert all(margin == [512, 0] for margin in reduced["column_margins"].values())
    assert all(int(values.sum()) == 0 for values in reduced["states"].values())


def test_adt_reducer_keeps_mixed_supported_and_unsupported_marker_axis(tmp_path: Path):
    selected = [f"CELL-{index}" for index in range(512)]
    markers = [row["adt_target"] for row in _supported_candidates()]
    unsupported = set(markers[:2])
    path = tmp_path / "mixed_adt.csv.gz"
    _write_adt_csv(path, selected, markers, constant_markers=unsupported)
    reduced = subject._read_adt_csv(path, "FH1001", selected, markers)
    assert reduced["available_adt_targets"] == markers
    assert {
        marker
        for marker, supported in reduced["marker_support"].items()
        if not supported
    } == unsupported
    for marker in markers:
        expected_margin = [512, 0] if marker in unsupported else [256, 256]
        assert reduced["column_margins"][marker] == expected_margin


def test_adt_stage_validator_binds_mixed_subject_marker_support(monkeypatch):
    candidate = subject._candidate()
    targets = [row["adt_target"] for row in _supported_candidates()]
    unsupported = targets[0]
    digest = "a" * 64
    source_digest = "b" * 64
    monkeypatch.setattr(
        subject,
        "_require_rna_stage",
        lambda: (
            {
                "source_model_sha256": source_digest,
                "subjects": [
                    {
                        "subject_id": row["subject_id"],
                        "selected_axis_sha256": digest,
                    }
                    for row in candidate["recipient_cohort"]["subjects"]
                ],
            },
            "c" * 40,
        ),
    )
    real_sha256 = subject._sha256
    monkeypatch.setattr(
        subject,
        "_sha256",
        lambda path: (
            digest if Path(path) == subject.DEFAULT_RNA else real_sha256(Path(path))
        ),
    )
    records = []
    for expected in candidate["recipient_cohort"]["subjects"]:
        support = {target: target != unsupported for target in targets}
        records.append(
            {
                "subject_id": expected["subject_id"],
                "gsm": expected["gsm"],
                "batch": expected["batch"],
                "adt_csv_name": expected["adt_csv_gz"]["name"],
                "adt_csv_bytes": expected["adt_csv_gz"]["bytes"],
                "adt_csv_sha256": digest,
                "selected_axis_sha256": digest,
                "column_margins": {
                    target: ([256, 256] if supported else [512, 0])
                    for target, supported in support.items()
                },
                "adt_marker_support": support,
                "adt_variation_qc": {
                    target: {
                        "distinct_raw_values": 7 if supported else 1,
                        "lower_boundary_value": 2 if supported else 0,
                        "upper_boundary_value": 3 if supported else 0,
                        "boundary_tie_cells": 0 if supported else 512,
                        "maximum_boundary_tie_cells": 128,
                        "passes": supported,
                    }
                    for target, supported in support.items()
                },
                "adt_state_sha256": {target: digest for target in targets},
                "access_audit": {
                    "orientation": "cells_by_markers",
                    "selected_identifier_values_read": 512,
                    "selected_adt_numeric_values_read": 512 * len(targets),
                    "rna_state_files_opened": 0,
                    "rna_state_values_read": 0,
                },
            }
        )
    payload = {
        "schema": "gse309593-independent-study-adt-stage/1.0",
        "status": "ADT_STAGE_FROZEN_WITHOUT_RNA_STATE_ACCESS",
        "created_at_utc": "2026-08-29T00:00:00Z",
        "rna_stage_commit": "c" * 40,
        "rna_stage_sha256": digest,
        "source_model_sha256": source_digest,
        "recipient_subjects": [row["subject_id"] for row in records],
        "available_adt_targets": targets,
        "subjects": records,
        "private_artifacts": {
            "adt_states_sha256": digest,
            "adt_states_bytes": 1,
            "paths_serialized": 0,
        },
        "access_boundary": {
            "selection_bridge_read": True,
            "rna_state_artifact_path_received": False,
            "rna_state_files_opened": 0,
            "rna_state_values_read": 0,
            "all_512_rna_selected_identifiers_required_exactly_once": True,
            "fallback_or_resampling_permitted": False,
        },
        "stage": "adt",
        "runtime_environment": {},
        "attempt_tag": subject.ADT_ATTEMPT_TAG,
        "attempt_commit": "d" * 40,
    }
    subject._validate_adt_stage_payload(payload)
    payload["subjects"][0]["column_margins"][unsupported] = [256, 256]
    with pytest.raises(PermissionError, match="ADT fixed margin differs"):
        subject._validate_adt_stage_payload(payload)


def test_all_methods_reconstruct_the_same_recipient_margins():
    candidates = _supported_candidates()[:9]
    panel = subject._final_panel(
        subject._candidate(),
        _source_model(),
        [row["rna_symbol"] for row in candidates],
        [row["adt_target"] for row in candidates],
    )
    size = len(panel)
    rows = np.broadcast_to(np.asarray([300, 212]), (size, size, 2)).copy()
    columns = np.broadcast_to(np.asarray([256, 256]), (size, size, 2)).copy()
    predictions = subject._predicted_tables(_source_model(), panel, rows, columns)
    assert set(predictions) == set(subject.REQUIRED_METHODS)
    for values in predictions.values():
        assert np.allclose(values.sum(axis=-1), rows)
        assert np.allclose(values.sum(axis=-2), columns)


def test_one_shot_attempt_is_publicly_bound_and_cannot_rerun(
    tmp_path: Path, monkeypatch
):
    attempt = tmp_path / "attempt.jsonl"
    output = tmp_path / "output.json"
    monkeypatch.setattr(subject, "ROOT", tmp_path)
    monkeypatch.setattr(subject, "PROTOCOL_BINDINGS", ())
    monkeypatch.setattr(
        subject, "STAGE_PATHS", {"rna": (attempt, output, "attempt-tag")}
    )
    monkeypatch.setattr(subject, "_prerequisites", lambda stage: {"stage": stage})
    monkeypatch.setattr(subject, "_require_public_tag", lambda tag, paths: "a" * 40)
    monkeypatch.setattr(subject, "_validate_stage_payload", lambda *_: None)
    subject.claim_stage("rna")
    with pytest.raises(PermissionError):
        subject.claim_stage("rna")
    result = subject._run_claimed_stage(
        "rna", lambda: {"schema": "synthetic/1.0", "status": "PASS"}
    )
    assert result["status"] == "PASS"
    assert [json.loads(line)["event"] for line in attempt.read_text().splitlines()] == [
        "STARTED",
        "EXECUTION_BEGINS_AFTER_PUBLIC_ATTEMPT",
        "SUCCEEDED",
    ]
    with pytest.raises(PermissionError):
        subject._run_claimed_stage("rna", lambda: {})


def test_terminal_refusal_is_sanitized_and_closes_attempt(tmp_path: Path, monkeypatch):
    attempt = tmp_path / "attempt.jsonl"
    output = tmp_path / "output.json"
    monkeypatch.setattr(subject, "ROOT", tmp_path)
    monkeypatch.setattr(subject, "PROTOCOL_BINDINGS", ())
    monkeypatch.setattr(
        subject, "STAGE_PATHS", {"rna": (attempt, output, "attempt-tag")}
    )
    monkeypatch.setattr(subject, "_prerequisites", lambda stage: {"stage": stage})
    monkeypatch.setattr(subject, "_require_public_tag", lambda tag, paths: "a" * 40)
    monkeypatch.setattr(subject, "_validate_stage_payload", lambda *_: None)
    subject.claim_stage("rna")
    payload = subject._run_claimed_stage(
        "rna", lambda: (_ for _ in ()).throw(subject.ProtocolRefusal("SUPPORT_FAIL"))
    )
    assert payload["status"] == "TERMINAL_RNA_REFUSAL"
    assert payload["reason_code"] == "SUPPORT_FAIL"
    assert "path" not in json.dumps(payload).lower()
    assert len(attempt.read_text().splitlines()) == 3


def _public_stage_fixture(tmp_path: Path, *, terminal: bool) -> tuple[Path, Path]:
    attempt = tmp_path / "rna_attempt.jsonl"
    output = tmp_path / "rna.json"
    runtime = {"runtime": "frozen"}
    attempt_commit = "a" * 40
    started = {
        "schema": "gse309593-independent-study-attempt/1.0",
        "status": "ONE_SHOT_ATTEMPT_CLAIMED",
        "event": "STARTED",
        "stage": "rna",
        "created_at_utc": "2026-08-29T00:00:00Z",
        "output": "rna.json",
        "target_files_opened_before_this_record": 0,
        "target_identifiers_read_before_this_record": 0,
        "target_assay_values_read_before_this_record": 0,
        "prerequisites": {},
        "runtime_environment": runtime,
    }
    execution = {
        "created_at_utc": "2026-08-29T00:01:00Z",
        "event": "EXECUTION_BEGINS_AFTER_PUBLIC_ATTEMPT",
        "attempt_tag": "attempt-tag",
        "attempt_commit": attempt_commit,
        "stage": "rna",
        "runtime_environment": runtime,
    }
    if terminal:
        payload = {
            "schema": "gse309593-independent-study-terminal-refusal/1.0",
            "status": "TERMINAL_RNA_REFUSAL",
            "stage": "rna",
            "created_at_utc": "2026-08-29T00:02:00Z",
            "attempt_started_at_utc": started["created_at_utc"],
            "reason_code": "SUPPORT_FAIL",
            "adaptive_rescue_permitted": False,
            "runtime_environment": runtime,
            "attempt_tag": "attempt-tag",
            "attempt_commit": attempt_commit,
        }
    else:
        payload = {
            "schema": "synthetic/1.0",
            "status": "PASS",
            "stage": "rna",
            "runtime_environment": runtime,
            "attempt_tag": "attempt-tag",
            "attempt_commit": attempt_commit,
        }
    output.write_text(json.dumps(payload))
    finished = {
        "created_at_utc": "2026-08-29T00:03:00Z",
        "event": "TERMINAL_REFUSAL" if terminal else "SUCCEEDED",
        "attempt_tag": "attempt-tag",
        "attempt_commit": attempt_commit,
        "output_sha256": subject._sha256(output),
        "stage": "rna",
        "runtime_environment": runtime,
    }
    if terminal:
        finished["reason_code"] = "SUPPORT_FAIL"
    else:
        finished["status"] = "PASS"
    attempt.write_text(
        "\n".join(
            json.dumps(record, sort_keys=True)
            for record in (started, execution, finished)
        )
        + "\n"
    )
    return attempt, output


def _patch_public_stage_validation(
    monkeypatch, tmp_path: Path, attempt: Path, output: Path
) -> None:
    monkeypatch.setattr(subject, "ROOT", tmp_path)
    monkeypatch.setattr(subject, "PROTOCOL_BINDINGS", ())
    monkeypatch.setattr(
        subject, "STAGE_PATHS", {"rna": (attempt, output, "attempt-tag")}
    )
    monkeypatch.setattr(subject, "STAGE_RESULT_TAGS", {"rna": "result-tag"})
    monkeypatch.setattr(subject, "_prerequisites", lambda _: {})
    monkeypatch.setattr(
        subject, "_require_runtime_environment", lambda: {"runtime": "frozen"}
    )
    monkeypatch.setattr(subject, "_require_public_tag", lambda *_: "b" * 40)
    monkeypatch.setattr(subject, "_validate_attempt_tag_snapshot", lambda *_: None)
    monkeypatch.setattr(subject, "_validate_stage_payload", lambda *_: None)


def test_completed_stage_semantic_verifier_rejects_status_mutation(
    tmp_path: Path, monkeypatch
):
    attempt, output = _public_stage_fixture(tmp_path, terminal=False)
    _patch_public_stage_validation(monkeypatch, tmp_path, attempt, output)
    payload, _ = subject._validated_public_stage(
        "rna", "result-tag", attempt, output, "PASS"
    )
    assert payload["status"] == "PASS"
    lines = [json.loads(line) for line in attempt.read_text().splitlines()]
    lines[-1]["status"] = "MUTATED"
    attempt.write_text("\n".join(json.dumps(row) for row in lines) + "\n")
    with pytest.raises(PermissionError, match="lineage"):
        subject._validated_public_stage(
            "rna", "result-tag", attempt, output, "PASS"
        )


def test_terminal_stage_semantic_verifier_rejects_reason_mutation(
    tmp_path: Path, monkeypatch
):
    attempt, output = _public_stage_fixture(tmp_path, terminal=True)
    _patch_public_stage_validation(monkeypatch, tmp_path, attempt, output)
    payload, _ = subject._validated_terminal_stage("rna")
    assert payload["reason_code"] == "SUPPORT_FAIL"
    lines = [json.loads(line) for line in attempt.read_text().splitlines()]
    lines[-1]["reason_code"] = "MUTATED"
    attempt.write_text("\n".join(json.dumps(row) for row in lines) + "\n")
    with pytest.raises(PermissionError, match="lineage"):
        subject._validated_terminal_stage("rna")


def test_subject_bootstrap_and_batch_gate_report_sample_size(monkeypatch):
    monkeypatch.setattr(subject, "BOOTSTRAPS", 1000)
    subjects = [f"S{index}" for index in range(21)]
    batches = {value: f"B{index % 7}" for index, value in enumerate(subjects)}
    primary = np.full(len(subjects), 0.70)
    comparator = np.full(len(subjects), 1.00)
    result = subject._held_comparison(
        subjects, batches, primary, comparator, classical=False, seed_offset=0
    )
    assert result["passes"]
    assert result["subjects"] == 21
    assert result["bootstrap_draws"] == 1000
    assert len(result["batch_mean_differences"]) == 7
    assert result["batch_sign_flip"]["method"] == "exact"
    assert result["batch_sign_flip"]["draws"] == 128
    assert set(result["leave_one_batch_out_mean_differences"]) == set(batches.values())
    assert result["batch_sign_flip"][
        "observed_donor_equal_mean_difference"
    ] == pytest.approx(result["mean_paired_difference"])


def test_batch_sign_flip_preserves_donor_equal_weighting():
    result = subject._batch_sign_flip({"B1": -1.0, "B2": -0.1}, {"B1": 1, "B2": 4})
    assert result["observed_donor_equal_mean_difference"] == pytest.approx(-0.28)


def test_private_paths_inside_public_repository_are_forbidden(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(subject, "ROOT", tmp_path)
    with pytest.raises(PermissionError, match="outside"):
        subject._private_path(tmp_path / "states.json")


@pytest.mark.parametrize("stage", ["rna", "adt"])
def test_assay_scratch_directory_must_be_outside_public_repository(
    tmp_path: Path, monkeypatch, stage: str
):
    monkeypatch.setattr(subject, "ROOT", tmp_path)
    body = subject._rna_stage_body if stage == "rna" else subject._adt_stage_body
    with pytest.raises(PermissionError, match="outside"):
        body(tmp_path / "scratch", tmp_path / "bridge.json", tmp_path / "state.json")


def test_recipient_schema_uses_the_frozen_ordered_common_intersection():
    axis = subject._ordered_schema_intersection(None, ["A", "B", "C", "D"])
    axis = subject._ordered_schema_intersection(axis, ["D", "A", "C"])
    axis = subject._ordered_schema_intersection(axis, ["C", "A", "X"])
    assert axis == ["A", "C"]
    with pytest.raises(subject.ProtocolRefusal, match="FEWER_THAN_NINE"):
        subject._require_common_schema(
            [f"M{index}" for index in range(8)],
            "COMMON_SCHEMA_HAS_FEWER_THAN_NINE_CANDIDATES",
        )
    exact = [f"M{index}" for index in range(9)]
    assert subject._require_common_schema(exact, "unused") == exact


@pytest.mark.parametrize(
    "payload",
    [
        {"states": [[0, 1]]},
        {"subjects": [{"selected_ids": ["AAAC"]}]},
        {"note": str(subject.ROOT / "private.json")},
    ],
)
def test_public_payload_rejects_private_state_identifiers_and_local_paths(payload):
    with pytest.raises(PermissionError):
        subject._validate_public_payload(payload)


def test_public_stage_rejects_a_lightweight_tag(monkeypatch):
    monkeypatch.setattr(
        subject.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="commit\n"),
    )
    with pytest.raises(PermissionError, match="not annotated"):
        subject._require_public_tag("lightweight", ())


def test_protocol_binds_runner_tests_runtime_and_environment():
    required = {
        "experiments/confirm_gse309593_independent_study.py",
        "tests/test_gse309593_independent_study.py",
        "mapreg/heterogeneity_adaptive_coupling.py",
        "mapreg/table_prediction.py",
        "requirements.txt",
        "pyproject.toml",
    }
    assert required <= set(subject.PROTOCOL_BINDINGS)
    assert not subject.DEFAULT_SOURCE_AUTHORIZATION.exists()
    assert not subject.DEFAULT_SCORE_AUTHORIZATION.exists()
