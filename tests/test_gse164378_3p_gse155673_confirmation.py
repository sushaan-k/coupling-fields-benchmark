import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
import stat

import numpy as np
import pytest

from experiments import confirm_gse164378_3p_gse155673 as confirmation


def _write_mtx(path: Path, body: str) -> None:
    with gzip.open(path, "wt") as stream:
        stream.write("%%MatrixMarket matrix coordinate integer general\n")
        stream.write(body)


def test_frozen_candidate_protocol_and_bindings_are_exact():
    candidate = confirmation._candidate()
    protocol = confirmation._protocol()
    markers = confirmation._markers(candidate)
    assert len(markers) == 24
    assert candidate["cell_selection_salt"] == confirmation.CELL_SELECTION_SALT
    assert candidate["source_split"]["calibration"] == list(
        confirmation.CALIBRATION_DONORS
    )
    assert candidate["source_split"]["validation"] == list(
        confirmation.VALIDATION_DONORS
    )
    assert candidate["source_split"]["validation_batch_components"] == [
        list(component) for component in confirmation.VALIDATION_COMPONENTS
    ]
    assert protocol["public_tags"]["prediction"] == confirmation.PREDICTION_TAG
    assert (
        protocol["transport_contract"]["download_attempts"]
        == confirmation.DOWNLOAD_ATTEMPTS
        == 3
    )
    assert (
        confirmation.DESTROYED_LINK_SALT
        in protocol["primary_estimator"]["destroyed_link_control"]
    )
    assert "Untuned raw Poisson is never eligible to lock" in protocol[
        "comparators"
    ]["lock_rule"]
    assert (
        "data/confirmation/gse164378_3p_gse155673/score_authorization_template_v1.json"
        in confirmation.PROTOCOL_BINDINGS
    )
    assert [
        confirmation._held_id(row) for row in candidate["held_donors"]
    ] == list(confirmation.HELD_DONORS)
    phenotype, severity = confirmation._held_groups(candidate)
    assert {donor for donor, group in phenotype.items() if group == "healthy"} == set(
        confirmation.HEALTHY_DONORS
    )
    assert {donor for donor, group in severity.items() if group == "moderate"} == set(
        confirmation.MODERATE_DONORS
    )
    assert {donor for donor, group in severity.items() if group == "severe"} == set(
        confirmation.SEVERE_DONORS
    )
    assert confirmation._canonical_json_sha256(
        [
            confirmation._marker_record(row)
            for row in candidate["marker_contract"]["markers"]
        ]
    ) == confirmation.FROZEN_MARKER_AXIS_SHA256
    assert (
        confirmation._canonical_json_sha256(candidate["source_donors"])
        == confirmation.FROZEN_SOURCE_DONOR_AXIS_SHA256
    )
    assert (
        confirmation._canonical_json_sha256(candidate["held_donors"])
        == confirmation.FROZEN_HELD_DONOR_AXIS_SHA256
    )


def test_candidate_rejects_marker_and_donor_axis_mutations(
    tmp_path: Path, monkeypatch
):
    candidate = confirmation._read_json(confirmation.DEFAULT_CANDIDATE)
    path = tmp_path / "candidate.json"
    monkeypatch.setattr(confirmation, "DEFAULT_CANDIDATE", path)

    marker_mutation = copy.deepcopy(candidate)
    marker_mutation["marker_contract"]["markers"][0]["marker_id"] = (
        "NOT_THE_FROZEN_MARKER"
    )
    path.write_text(json.dumps(marker_mutation))
    with pytest.raises(PermissionError, match="marker records differ"):
        confirmation._candidate()

    source_mutation = copy.deepcopy(candidate)
    source_mutation["source_donors"].append(
        copy.deepcopy(source_mutation["source_donors"][0])
    )
    path.write_text(json.dumps(source_mutation))
    with pytest.raises(PermissionError, match="source donor records differ"):
        confirmation._candidate()


def test_feature_resolution_requires_exact_ids_labels_and_modalities():
    marker = confirmation._markers(confirmation._candidate())[0]
    source_rna = [[marker["rna_symbol"], marker["rna_symbol"]]]
    held_rna = [[marker["rna_ensembl_id"], marker["rna_symbol"], "Gene Expression"]]
    source_adt = [[marker["source_adt_feature"], marker["source_adt_feature"]]]
    held_adt = [
        [
            marker["held_adt_feature"],
            marker["held_adt_label"],
            "Antibody Capture",
        ]
    ]
    assert confirmation._resolve_feature_rows(
        source_rna, [marker], modality="rna", held=False
    ) == {0: 0}
    assert confirmation._resolve_feature_rows(
        held_rna, [marker], modality="rna", held=True
    ) == {0: 0}
    assert confirmation._resolve_feature_rows(
        source_adt, [marker], modality="adt", held=False
    ) == {0: 0}
    assert confirmation._resolve_feature_rows(
        held_adt, [marker], modality="adt", held=True
    ) == {0: 0}
    held_rna[0][1] = "WRONG"
    with pytest.raises(confirmation.ConfirmationRefusal, match="ONE_TO_ONE"):
        confirmation._resolve_feature_rows(
            held_rna, [marker], modality="rna", held=True
        )
    held_adt[0][1] = "WRONG"
    with pytest.raises(confirmation.ConfirmationRefusal, match="ONE_TO_ONE"):
        confirmation._resolve_feature_rows(
            held_adt, [marker], modality="adt", held=True
        )


def test_score_authorization_binds_every_frozen_and_prescore_artifact(
    tmp_path: Path, monkeypatch
):
    template = confirmation._read_json(
        confirmation.DEFAULT_SCORE_AUTHORIZATION_TEMPLATE
    )
    monkeypatch.setattr(confirmation, "ROOT", tmp_path)
    paths = {
        "DEFAULT_PREDICTION": "prediction.json",
        "DEFAULT_PROTOCOL": "protocol.json",
        "DEFAULT_CANDIDATE": "candidate.json",
        "DEFAULT_MANIFEST": "manifest.json",
        "DEFAULT_METADATA_PREFLIGHT": "preflight.json",
        "DEFAULT_RUNTIME": "runtime.json",
        "DEFAULT_SCORE_AUTHORIZATION_TEMPLATE": "authorization-template.json",
        "DEFAULT_TESTS": "tests.py",
        "DEFAULT_SOURCE": "source.json",
        "DEFAULT_RNA": "rna.json",
        "DEFAULT_ADT": "adt.json",
    }
    for attribute, filename in paths.items():
        path = tmp_path / filename
        path.write_text(f"{attribute}\n")
        monkeypatch.setattr(confirmation, attribute, path)
    monkeypatch.setattr(
        confirmation, "_require_public_tag", lambda tag, bound: "a" * 40
    )
    monkeypatch.setattr(confirmation, "_require_runtime_environment", lambda: {})

    bindings = confirmation._score_authorization_bindings(
        {"status": "frozen"}, "b" * 40
    )
    assert bindings["prediction_path"] == "prediction.json"
    assert bindings["prediction_public_commit"] == "b" * 40
    assert bindings["outcome_access_authorized"] is True
    assert bindings["publication_required_before_score"] is True
    assert bindings["recipient_joint_tables_formed_before_authorization"] == 0
    for key in (
        "candidate_designation_sha256",
        "source_manifest_sha256",
        "metadata_preflight_sha256",
        "runtime_environment_sha256",
        "score_authorization_template_sha256",
        "runner_sha256",
        "tests_sha256",
        "source_result_sha256",
        "rna_result_sha256",
        "adt_result_sha256",
    ):
        assert len(bindings[key]) == 64
    actual_keys = set(bindings) | {"schema", "status", "created_at_utc"}
    assert set(template) - {"blocker"} <= actual_keys


def test_public_lineage_requires_git_ancestry(monkeypatch):
    class Result:
        def __init__(self, returncode: int):
            self.returncode = returncode

    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return Result(0)

    monkeypatch.setattr(confirmation.subprocess, "run", run)
    confirmation._require_commit_ancestor("a" * 40, "b" * 40)
    assert calls[0][0][1:3] == ["merge-base", "--is-ancestor"]
    monkeypatch.setattr(
        confirmation.subprocess, "run", lambda command, **kwargs: Result(1)
    )
    with pytest.raises(PermissionError, match="not descendant"):
        confirmation._require_commit_ancestor("a" * 40, "b" * 40)


def test_completion_rejects_attempt_tag_without_exact_preexecution_snapshot(
    monkeypatch,
):
    started = {"stage": "prediction", "prerequisites": {}}
    monkeypatch.setattr(confirmation, "_remote_tag_commit", lambda tag: "b" * 40)
    monkeypatch.setattr(confirmation, "_require_commit_ancestor", lambda *args: None)
    monkeypatch.setattr(
        confirmation, "_published_path_bytes", lambda commit, relative: b"wrong\n"
    )
    with pytest.raises(PermissionError, match="pre-execution snapshot"):
        confirmation._require_public_attempt_snapshot(
            "prediction", started, "a" * 40
        )

    monkeypatch.setattr(
        confirmation,
        "_published_path_bytes",
        lambda commit, relative: confirmation._jsonl_record_bytes(started),
    )
    assert (
        confirmation._require_public_attempt_snapshot(
            "prediction", started, "a" * 40
        )
        == "b" * 40
    )


def test_matrix_market_row_firewall_never_converts_unauthorized_values(
    tmp_path: Path,
):
    path = tmp_path / "matrix.mtx.gz"
    _write_mtx(path, "3 2 2\n1 1 7\n2 1 SECRET\n")
    values, _ = confirmation._read_matrix_market_subset(
        path,
        expected_rows=3,
        expected_columns=2,
        selected_columns={0: 0},
        retained_rows={0: 0},
        authorized_rows={0},
        collect_totals=False,
    )
    np.testing.assert_array_equal(values, [[7]])
    with pytest.raises(
        confirmation.ConfirmationRefusal, match="AUTHORIZED_MATRIX_VALUE"
    ):
        confirmation._read_matrix_market_subset(
            path,
            expected_rows=3,
            expected_columns=2,
            selected_columns={0: 0},
            retained_rows={1: 0},
            authorized_rows={1},
            collect_totals=False,
        )


def test_matrix_market_refuses_duplicate_authorized_coordinate(tmp_path: Path):
    path = tmp_path / "duplicate.mtx.gz"
    _write_mtx(path, "1 1 2\n1 1 2\n1 1 3\n")
    with pytest.raises(confirmation.ConfirmationRefusal, match="DUPLICATED"):
        confirmation._read_matrix_market_subset(
            path,
            expected_rows=1,
            expected_columns=1,
            selected_columns={0: 0},
            retained_rows={0: 0},
            authorized_rows={0},
            collect_totals=False,
        )


def test_transport_retries_partial_exception_without_journaling(
    tmp_path: Path, monkeypatch
):
    payload = b"complete"

    class Broken(io.BytesIO):
        def __init__(self):
            super().__init__(b"partial")
            self.reads = 0

        def read(self, size=-1):
            self.reads += 1
            if self.reads == 1:
                return super().read(3)
            raise ConnectionResetError("truncated")

    responses = iter([Broken(), io.BytesIO(payload)])
    monkeypatch.setattr(confirmation, "urlopen", lambda url: next(responses))
    journal = []
    monkeypatch.setattr(
        confirmation,
        "_append_assay_access",
        lambda *args: journal.append(args),
    )
    destination = tmp_path / "scratch" / "assay.bin"
    record = {
        "filename": destination.name,
        "url": f"https://example.test/{destination.name}",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    assert (
        confirmation._fetch(
            record,
            destination,
            stage="source",
            modality="rna",
            unit_id="unit",
        ).read_bytes()
        == payload
    )
    assert len(journal) == 1
    assert stat.S_IMODE(destination.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    assert not destination.with_suffix(".bin.part").exists()


def test_private_artifact_is_owner_only(tmp_path: Path):
    destination = tmp_path / "private" / "states.json"
    confirmation._write_private(destination, {"states": [0, 1]})
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600


def test_complete_transport_mismatch_is_journaled_then_refused(
    tmp_path: Path, monkeypatch
):
    payload = b"wrong"
    monkeypatch.setattr(confirmation, "urlopen", lambda url: io.BytesIO(payload))
    journal = []
    monkeypatch.setattr(
        confirmation,
        "_append_assay_access",
        lambda *args: journal.append(args),
    )
    destination = tmp_path / "assay.bin"
    record = {
        "filename": destination.name,
        "url": f"https://example.test/{destination.name}",
        "bytes": len(payload) + 1,
        "sha256": None,
    }
    with pytest.raises(confirmation.ConfirmationRefusal, match="SIZE_DIFFERS"):
        confirmation._fetch(
            record,
            destination,
            stage="source",
            modality="rna",
            unit_id="unit",
        )
    assert len(journal) == 1


def test_cell_selection_is_salted_deterministic_and_exact_budget():
    values = [f"cell-{index:04d}" for index in range(500)]
    expected = sorted(
        values,
        key=lambda value: (
            hashlib.sha256(
                f"{confirmation.CELL_SELECTION_SALT}|P1|{value}".encode()
            ).hexdigest(),
            value,
        ),
    )[: confirmation.CELL_BUDGET]
    assert confirmation._deterministic_selection(values, "P1") == expected
    assert confirmation._deterministic_selection(values[::-1], "P1") == expected


def test_source_marker_lock_requires_rna_and_adt_validity_in_all_eight_donors():
    marker_count = 18
    selected = {
        donor: [f"{donor}-{index}" for index in range(confirmation.CELL_BUDGET)]
        for donor in confirmation.SOURCE_DONORS
    }
    base = np.tile(np.arange(confirmation.CELL_BUDGET)[:, None] % 2, (1, marker_count))
    rna = {donor: base.copy() for donor in confirmation.SOURCE_DONORS}
    adt = {donor: base.copy() for donor in confirmation.SOURCE_DONORS}
    totals = {
        donor: np.full(confirmation.CELL_BUDGET, 10, dtype=np.int64)
        for donor in confirmation.SOURCE_DONORS
    }
    rna["P8"][:, 17] = 0
    adt["P7"][:, 16] = 0
    locked, records = confirmation._source_records(selected, rna, adt, totals)
    assert locked == list(range(16))
    assert set(records) == set(confirmation.SOURCE_DONORS)


def _source_records_for_tuning(marker_count: int = 16):
    table = np.full((marker_count, marker_count, 2, 2), 96, dtype=np.int64)
    profile = np.arange(marker_count, dtype=float)
    return {
        donor: {
            "tables": table,
            "destroyed_tables": table,
            "rna_profile": profile + index,
            "adt_profile": profile + 2 * index,
        }
        for index, donor in enumerate(confirmation.SOURCE_DONORS)
    }


def test_source_tuning_locks_classical_separately_from_better_raw_and_refits_all_eight(
    monkeypatch,
):
    primary = confirmation.PrimaryConfig(0.1, 0.01, 0.1, 0.5)
    zero = confirmation.PrimaryConfig(0.1, 0.01, 0.0, 0.5)
    monkeypatch.setattr(
        confirmation,
        "_select_primary",
        lambda records, graphs: {
            "selected": zero if graphs == (0.0,) else primary,
            "losses": {donor: 0.05 for donor in confirmation.VALIDATION_DONORS},
            "component_equal_loss": 0.05,
            "complete_candidates": 1,
        },
    )
    fit_sizes = []

    def fit_primary(tables, rna, adt, config):
        fit_sizes.append(len(rna))
        return {
            "family": "graph_regularized_exact_fixed_margin_hierarchical_coupling",
            "population_log_odds": np.zeros(tables.shape[1:3]),
            "transport_multiplier": config.transport_multiplier,
        }

    monkeypatch.setattr(confirmation, "_fit_primary", fit_primary)
    monkeypatch.setattr(
        confirmation,
        "_model_losses",
        lambda records, donors, model: {donor: 0.05 for donor in donors},
    )
    classical_loss = {
        confirmation.RAW_RESIDUAL_METHOD: 0.10,
        "poisson_independence_signed_deviance_residual": 0.20,
        "common_effect_stratified_cmle": 0.30,
        "pooled_saturated_poisson": 0.40,
        "paule_mandel_random_effects_log_odds": 0.50,
    }
    monkeypatch.setattr(
        confirmation,
        "_select_classical",
        lambda records, method: (
            {},
            {donor: classical_loss[method] for donor in confirmation.VALIDATION_DONORS},
            1.0,
        ),
    )
    monkeypatch.setattr(
        confirmation,
        "_fit_classical",
        lambda method, tables: {
            "family": "target_margin_independence",
            "method": method,
        },
    )
    markers = [{"marker_id": str(index)} for index in range(16)]
    result = confirmation._source_tuning(_source_records_for_tuning(), markers)
    assert result["locked_classical_method"] == (
        "poisson_independence_signed_deviance_residual"
    )
    assert confirmation.RAW_RESIDUAL_METHOD in result["models"]
    assert fit_sizes[-3:] == [8, 8, 8]


def test_source_gate_requires_strict_gain_in_each_validation_component(monkeypatch):
    primary = confirmation.PrimaryConfig(0.1, 0.01, 0.1, 0.5)
    monkeypatch.setattr(
        confirmation,
        "_select_primary",
        lambda records, graphs: {
            "selected": primary,
            "losses": {donor: 0.1 for donor in confirmation.VALIDATION_DONORS},
            "component_equal_loss": 0.1,
            "complete_candidates": 1,
        },
    )
    monkeypatch.setattr(
        confirmation,
        "_fit_primary",
        lambda tables, rna, adt, config: {"family": "target_margin_independence"},
    )
    monkeypatch.setattr(
        confirmation,
        "_model_losses",
        lambda records, donors, model: {
            "P2": 0.1,
            "P4": 0.1,
            "P6": 0.3,
            "P8": 0.3,
        },
    )
    monkeypatch.setattr(
        confirmation,
        "_select_classical",
        lambda records, method: (
            {},
            {donor: 0.2 for donor in confirmation.VALIDATION_DONORS},
            1.0,
        ),
    )
    with pytest.raises(confirmation.ConfirmationRefusal, match="EVERY_BATCH"):
        confirmation._source_tuning(_source_records_for_tuning(), [])


def _synthetic_prediction_inputs():
    candidate = confirmation._candidate()
    donors = [confirmation._held_id(row) for row in candidate["held_donors"]]
    panel = confirmation._markers(candidate)[:16]
    locked = "common_effect_stratified_cmle"
    methods = [
        "primary",
        "graph_zero_ablation",
        "destroyed_link",
        confirmation.RAW_RESIDUAL_METHOD,
        locked,
        confirmation.INDEPENDENCE_METHOD,
    ]
    models = {method: {"family": "target_margin_independence"} for method in methods}
    source = {
        "source_locked_marker_indices": list(range(16)),
        "marker_panel": panel,
        "model": {
            "models": models,
            "available_methods": methods,
            "locked_classical_method": locked,
            "primary_selection": {"selected_configuration": {"graph_penalty": 0.1}},
        },
    }
    axes = {donor: {"axis_valid": [True] * 16} for donor in donors}
    rna = {
        "held_donors": donors,
        "marker_panel": panel,
        "selected_axis_sha256": {donor: "a" * 64 for donor in donors},
        "row_margins": {donor: [[192, 192]] * 16 for donor in donors},
        "rna_axis_quality": axes,
    }
    adt = {
        "held_donors": donors,
        "marker_panel": panel,
        "selected_axis_sha256": rna["selected_axis_sha256"],
        "column_margins": {donor: [[192, 192]] * 16 for donor in donors},
        "adt_axis_quality": {donor: {"axis_valid": [True] * 16} for donor in donors},
    }
    return source, rna, adt


def test_prediction_freezes_margins_and_predictions_without_private_state_access(
    monkeypatch,
):
    source, rna, adt = _synthetic_prediction_inputs()
    monkeypatch.setattr(
        confirmation,
        "_require_completed",
        lambda stage: {"source": source, "rna": rna, "adt": adt}[stage],
    )
    monkeypatch.setattr(confirmation, "_sha256", lambda path: "b" * 64)
    result = confirmation._prediction_stage_body()
    assert result["held_private_state_artifacts_opened"] == 0
    assert result["held_joint_tables_formed"] == 0
    assert len(result["samples"]) == 12
    assert all(row["valid_ordered_pairs"] == 16**2 for row in result["samples"])


def test_prediction_refuses_global_and_per_donor_support_failures(monkeypatch):
    source, rna, adt = _synthetic_prediction_inputs()
    donors = rna["held_donors"]
    for donor in donors[:3]:
        rna["rna_axis_quality"][donor]["axis_valid"][0] = False
    monkeypatch.setattr(
        confirmation,
        "_require_completed",
        lambda stage: {"source": source, "rna": rna, "adt": adt}[stage],
    )
    with pytest.raises(confirmation.ConfirmationRefusal, match="TEN_OF_TWELVE"):
        confirmation._prediction_stage_body()

    source, rna, adt = _synthetic_prediction_inputs()
    for marker in range(4):
        rna["rna_axis_quality"][donors[0]]["axis_valid"][marker] = False
    monkeypatch.setattr(
        confirmation,
        "_require_completed",
        lambda stage: {"source": source, "rna": rna, "adt": adt}[stage],
    )
    with pytest.raises(confirmation.ConfirmationRefusal, match="EIGHTY_PERCENT"):
        confirmation._prediction_stage_body()


def test_full_held_comparison_passes_only_with_all_prespecified_checks():
    candidate = confirmation._candidate()
    donors = [confirmation._held_id(row) for row in candidate["held_donors"]]
    phenotype, severity = confirmation._held_groups(candidate)
    result = confirmation._comparison(
        donors,
        phenotype,
        severity,
        np.ones(12),
        np.full(12, 2.0),
    )
    assert result["passes"]
    assert result["favorable_donors"] == 12
    assert result["exact_paired_donor_sign_flip"]["assignments"] == 4096
    assert result["exact_donor_sign_test"]["one_sided_p"] == pytest.approx(1 / 4096)
    assert (
        result["disease_stratified_bootstrap"]["mean_difference_98_75th_percentile"] < 0
    )


def test_zero_comparator_loss_is_a_completed_endpoint_failure():
    candidate = confirmation._candidate()
    donors = [confirmation._held_id(row) for row in candidate["held_donors"]]
    phenotype, severity = confirmation._held_groups(candidate)
    result = confirmation._comparison(
        donors,
        phenotype,
        severity,
        np.ones(12),
        np.zeros(12),
    )
    assert not result["passes"]
    assert result["relative_loss_reduction"] is None
    bootstrap = result["disease_stratified_bootstrap"]
    assert bootstrap["relative_loss_reduction_97_5_percent_interval"] is None
    assert bootstrap["relative_loss_reduction_undefined_draws"] == 20_000


def test_raw_poisson_is_reported_but_descriptive_when_locked_gate_fails(
    monkeypatch,
):
    candidate = confirmation._candidate()
    donors = list(confirmation.HELD_DONORS)
    locked = "common_effect_stratified_cmle"
    methods = [
        "primary",
        "graph_zero_ablation",
        "destroyed_link",
        confirmation.RAW_RESIDUAL_METHOD,
        locked,
        confirmation.INDEPENDENCE_METHOD,
    ]
    markers = [{"marker_id": "M"}]
    truth_state = np.asarray([[index % 2] for index in range(384)], dtype=np.uint8)
    truth = confirmation._binary_tables(truth_state, truth_state)
    rows, columns = confirmation._margins(truth)
    method_values = {
        "primary": 1.0,
        locked: 2.0,
        confirmation.RAW_RESIDUAL_METHOD: 3.0,
        confirmation.INDEPENDENCE_METHOD: 4.0,
        "graph_zero_ablation": 5.0,
        "destroyed_link": 6.0,
    }
    samples = [
        {
            "donor_id": donor,
            "row_margins": rows.tolist(),
            "column_margins": columns.tolist(),
            "valid_ordered_pair_mask": [[True]],
            "valid_ordered_pair_mask_sha256": "h",
            "valid_ordered_pairs": 1,
            "required_valid_ordered_pairs": 1,
            "predicted_tables": {
                method: [[[[value, 0], [0, 0]]]]
                for method, value in method_values.items()
            },
            "prediction_sha256": {method: "h" for method in methods},
        }
        for donor in donors
    ]
    prediction = {
        "samples": samples,
        "methods": methods,
        "marker_panel": markers,
        "locked_classical_method": locked,
    }
    source = {
        "model": {
            "locked_classical_method": locked,
            "primary_selection": {"selected_configuration": {"graph_penalty": 0.1}},
        }
    }
    certificate = {"sha256": "x" * 64, "bytes": 1}
    axis = {donor: "a" * 64 for donor in donors}
    rna_public = {
        "rna_states": certificate,
        "selected_axis_sha256": axis,
        "rna_axis_quality": {donor: {"axis_valid": [True]} for donor in donors},
    }
    adt_public = {
        "adt_states": certificate,
        "selected_axis_sha256": axis,
        "adt_axis_quality": {donor: {"axis_valid": [True]} for donor in donors},
    }
    private = {
        "rna": {
            "schema": "gse164378-3p-gse155673-private-rna-states/1.0",
            "states": {donor: truth_state.tolist() for donor in donors},
            "selected_axis_sha256": axis,
        },
        "adt": {
            "schema": "gse164378-3p-gse155673-private-adt-states/1.0",
            "states": {donor: truth_state.tolist() for donor in donors},
            "selected_axis_sha256": axis,
        },
    }
    monkeypatch.setattr(confirmation, "_candidate", lambda: candidate)
    monkeypatch.setattr(
        confirmation,
        "_require_completed",
        lambda stage: {
            "prediction": prediction,
            "source": source,
            "rna": rna_public,
            "adt": adt_public,
        }[stage],
    )
    monkeypatch.setattr(confirmation, "_require_score_authorization", lambda: {})
    monkeypatch.setattr(
        confirmation,
        "_private_artifact",
        lambda path, cert: private["rna" if "rna" in Path(path).name else "adt"],
    )
    monkeypatch.setattr(confirmation, "_array_sha256", lambda value: "h")
    monkeypatch.setattr(confirmation, "_sha256", lambda path: "s" * 64)
    monkeypatch.setattr(
        confirmation,
        "_donor_loss",
        lambda observed, estimate, mask: float(np.asarray(estimate).flat[0]),
    )

    def comparison(donor_axis, phenotypes, severities, primary, comparator):
        marker = float(np.asarray(comparator)[0])
        return {"passes": marker != 2.0, "comparator_marker": marker}

    monkeypatch.setattr(confirmation, "_comparison", comparison)
    result = confirmation._score_stage_body(Path("rna.json"), Path("adt.json"))
    assert not result["primary_estimator_confirmed"]
    assert not result["broad_classical_support"]
    assert result["primary_vs_untuned_raw_poisson"]["serial_gate_status"] == (
        "DESCRIPTIVE_ONLY_LOCKED_GATE_FAILED"
    )
    assert confirmation.INDEPENDENCE_METHOD in result["samples"][0]["losses"]
    assert result["graph_zero_serial_secondary"]["status"] == (
        "NOT_EVALUATED_CLASSICAL_GATE_FAILED"
    )
