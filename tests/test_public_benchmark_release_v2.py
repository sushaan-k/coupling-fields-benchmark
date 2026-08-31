from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

from experiments.build_public_benchmark_release import (
    ABSENT_GSE179221_DOWNSTREAM_PATHS,
    CHECKSUM_PATH,
    COMPARISONS_PATH,
    GSE317605_RESULT_COMMIT,
    GSE317605_RESULT_PATH,
    GSE317605_RESULT_SHA256,
    GSE317605_RESULT_TAG,
    MANIFEST_PATH,
    PANELS_PATH,
    SEQUENCE_PATH,
    build,
)
from scripts.verify_public_benchmark_release import verify


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = (PANELS_PATH, COMPARISONS_PATH, SEQUENCE_PATH, MANIFEST_PATH, CHECKSUM_PATH)


def _rows(relative: Path) -> list[dict[str, str]]:
    with (ROOT / relative).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def test_release_builder_is_deterministic_and_verifies() -> None:
    build()
    first = {relative: (ROOT / relative).read_bytes() for relative in OUTPUTS}
    build()
    second = {relative: (ROOT / relative).read_bytes() for relative in OUTPUTS}
    assert first == second
    verify(ROOT)


def test_metric_aware_ledgers_include_completed_and_refused_evidence() -> None:
    panels = _rows(PANELS_PATH)
    comparisons = _rows(COMPARISONS_PATH)
    sequence = _rows(SEQUENCE_PATH)
    by_panel = {row["panel_id"]: row for row in panels}
    by_comparison = {row["comparison_id"]: row for row in comparisons}

    assert len(panels) == 37
    assert len(comparisons) == 29
    assert len(sequence) == 91
    assert sum(row["outcome_scored"] == "YES" for row in panels) == 12
    assert sum(row["inference_role"] == "procedural_refusal" for row in panels) == 24

    stephenson = by_panel["stephenson_newcastle_confirmation"]
    assert stephenson["inference_role"] == "confirmatory"
    assert float(stephenson["primary_value"]) == 0.012196845982760404
    residual = by_comparison["stephenson_newcastle_confirmation__best_residual"]
    assert float(residual["relative_improvement"]) == 0.17460498883932063
    assert float(residual["paired_difference_ci_95_high"]) < 0
    assert residual["decision"] == "PASS"

    gse239 = by_panel["gse239452_held_post_access_correction"]
    assert gse239["inference_role"] == "post_access_correction"
    assert gse239["result_artifact"] == (
        "results/gse239452_citeseq_post_access_correction.json"
    )
    assert gse239["result_sha256"] == (
        "1eafd82805a0bc6d94c05afdc4160fd6917e1145d64077fb52a770e09f45793b"
    )
    corrected_residual = by_comparison[
        "gse239452_held_post_access_correction__best_residual"
    ]
    assert float(corrected_residual["comparator_value"]) == 0.01413148577805464
    assert float(corrected_residual["relative_improvement"]) == (
        0.39805586032248474
    )
    assert float(corrected_residual["paired_difference_ci_95_low"]) == (
        -0.007095088337784283
    )
    assert float(corrected_residual["paired_difference_ci_95_high"]) == (
        -0.0042321214194290585
    )
    assert int(corrected_residual["favorable_units"]) == 9
    assert int(corrected_residual["total_units"]) == 9
    assert corrected_residual["result_artifact"] == gse239["result_artifact"]
    corrected_destroyed = by_comparison[
        "gse239452_held_post_access_correction__destroyed_link"
    ]
    assert corrected_destroyed["result_artifact"] == gse239["result_artifact"]
    exact = by_comparison[
        "gse239452_held_post_access_correction__posthoc_primary_vs_common_effect_exact_cmle"
    ]
    assert float(exact["relative_improvement"]) < 0
    assert exact["inference_role"] == "post_hoc_nonconfirmatory"

    gse342939 = by_panel["gse342939_ra_bcell_source_terminal"]
    assert gse342939["inference_role"] == "procedural_refusal"
    assert gse342939["outcome_scored"] == "NO"
    assert gse342939["primary_value"] == ""
    assert gse342939["decision"] == "TERMINAL_SOURCE_EXECUTION_REFUSAL"
    assert gse342939["result_sha256"] == (
        "563d8dd1aa3cfba8cf1c336b2162513fc9e852605392f4d733767dd98efc0eb0"
    )

    gse342939_sequence = [
        row
        for row in sequence
        if row["panel_id"] == "gse342939_ra_bcell_source_terminal"
    ]
    assert len(gse342939_sequence) == 8
    assert gse342939_sequence[-1]["status"] == "TERMINAL_SOURCE_EXECUTION_REFUSAL"
    assert gse342939_sequence[-1]["outcome_access"] == (
        "ALL_SOURCE_MATRICES_REDUCED_HELD_UNOPENED"
    )
    assert gse342939_sequence[-1]["public_before_outcome"] == "NOT_APPLICABLE"

    gse342939_result = json.loads(
        (ROOT / gse342939["result_artifact"]).read_text(encoding="utf-8")
    )
    assert gse342939_result["held_numeric_access_authorized"] is False
    assert gse342939_result["access_audit"]["held_numeric_urls_requested"] == 0
    assert gse342939_result["rerun_permitted"] is False
    assert len(gse342939_result["source_files"]) == 28
    assert all(
        source_file["completed"]
        and source_file["reduction_completed"]
        and source_file["deleted"]
        for source_file in gse342939_result["source_files"]
    )

    gse317605 = by_panel["gse317605_longitudinal_calibration_terminal"]
    assert gse317605["analysis_phase"] == "calibration_gate"
    assert gse317605["inference_role"] == "procedural_refusal"
    assert gse317605["evaluation_unit"] == "calibration patient"
    assert int(gse317605["unit_count"]) == 7
    assert int(gse317605["entity_count"]) == 256
    assert gse317605["decision"] == "CALIBRATION_FAIL"
    assert gse317605["outcome_scored"] == "NO"
    assert gse317605["primary_value"] == ""
    assert gse317605["result_artifact"] == GSE317605_RESULT_PATH
    assert gse317605["result_sha256"] == GSE317605_RESULT_SHA256
    assert not any(
        row["panel_id"] == gse317605["panel_id"] for row in comparisons
    )

    gse317605_result = json.loads(
        (ROOT / GSE317605_RESULT_PATH).read_text(encoding="utf-8")
    )
    selected = gse317605_result["selection"]
    assert selected["losses"]["primary"]["mean"] == 0.006529760758433352
    assert selected["losses"]["classical_time_conditioned_ridge_poisson"][
        "mean"
    ] == 0.006608620233208813
    poisson_gate = selected["gate"]["comparisons"][
        "classical_time_conditioned_ridge_poisson"
    ]
    assert poisson_gate["relative_reduction"] == 0.011932819861426913
    assert poisson_gate["favorable_patients"] == 6
    assert poisson_gate["passes"] is False
    assert selected["selected_primary"]["hypergraph_penalty"] == 0.0
    assert selected["selected_primary"] == selected["selected_graph_zero"]
    assert gse317605_result["pilot_matrix_requests"] == 0
    assert gse317605_result["held_matrix_requests"] == 0

    gse317605_sequence = [
        row
        for row in sequence
        if row["panel_id"] == "gse317605_longitudinal_calibration_terminal"
    ]
    assert [row["stage"] for row in gse317605_sequence] == [
        "protocol",
        "candidate_designation",
        "sample_manifest",
        "implementation",
        "calibration_attempt",
        "calibration_consumption",
        "calibration_access_journal",
        "calibration_result",
    ]
    assert all(
        row["public_commit_or_tag"] == GSE317605_RESULT_TAG
        for row in gse317605_sequence[-3:]
    )
    assert gse317605_sequence[-1]["artifact_sha256"] == GSE317605_RESULT_SHA256
    assert gse317605_sequence[-1]["outcome_access"] == (
        "CALIBRATION_ONLY_PILOT_AND_HELD_UNREQUESTED"
    )
    assert (
        subprocess.check_output(
            ["git", "rev-list", "-n", "1", GSE317605_RESULT_TAG],
            cwd=ROOT,
            text=True,
        ).strip()
        == GSE317605_RESULT_COMMIT
    )
    legacy_pooled = by_comparison[
        "gse239452_held_post_access_correction__"
        "posthoc_primary_vs_pooled_table_log_odds_conditional_reconstruction"
    ]
    assert legacy_pooled["comparator_method"] == (
        "pooled_table_log_odds_with_conditional_reconstruction"
    )
    stephenson_legacy_pooled = by_comparison[
        "stephenson_newcastle_confirmation__"
        "posthoc_primary_vs_pooled_table_log_odds_conditional_reconstruction"
    ]
    assert stephenson_legacy_pooled["comparator_method"] == (
        "pooled_table_log_odds_with_conditional_reconstruction"
    )
    assert not any(
        "posthoc_primary_vs_pooled_poisson_loglinear_interaction" in comparison_id
        for comparison_id in by_comparison
    )
    standard_poisson = by_comparison[
        "gse239452_held_post_access_correction__"
        "posthoc_standard_fixed_interaction_poisson"
    ]
    assert standard_poisson["inference_role"] == "post_hoc_nonconfirmatory"
    assert standard_poisson["comparator_method"] == (
        "standard_pooled_saturated_poisson_fixed_interaction"
    )
    assert standard_poisson["metric"] == "mean_multinomial_deviance_per_cell"
    assert float(standard_poisson["primary_value"]) == 0.008506365049036143
    assert float(standard_poisson["comparator_value"]) == 0.009982413964423759
    assert float(standard_poisson["relative_improvement"]) == 0.14786492732600487
    assert float(standard_poisson["relative_improvement_ci_95_low"]) == (
        0.12184115234996165
    )
    assert float(standard_poisson["relative_improvement_ci_95_high"]) == (
        0.17369414932349705
    )
    assert int(standard_poisson["favorable_units"]) == 9
    assert int(standard_poisson["total_units"]) == 9
    assert float(standard_poisson["p_value"]) == 1 / 512
    assert standard_poisson["decision"] == "DESCRIPTIVE"
    gse239_sequence = [
        row for row in sequence if row["panel_id"] == gse239["panel_id"]
    ]
    assert [row["stage"] for row in gse239_sequence] == [
        "protocol",
        "prediction",
        "score",
        "post_access_numerical_correction",
        "standard_fixed_interaction_poisson_audit",
    ]
    assert gse239_sequence[1]["artifact"] == (
        "results/gse239452_citeseq_predictions.json"
    )
    assert gse239_sequence[2]["artifact"] == (
        "results/gse239452_citeseq_confirmation.json"
    )
    assert gse239_sequence[3]["artifact"] == gse239["result_artifact"]
    assert gse239_sequence[3]["status"] == "POST_ACCESS_CORRECTION_COMPLETE"
    assert gse239_sequence[-1]["stage"] == (
        "standard_fixed_interaction_poisson_audit"
    )
    assert gse239_sequence[-1]["public_before_outcome"] == "NO"

    bmmc = by_comparison[
        "scmmib_bmmc_adaptive_development__primary_vs_best_residual"
    ]
    combat = by_comparison[
        "combat_oxford_adaptive_development__primary_vs_best_residual"
    ]
    assert float(bmmc["relative_improvement"]) == 0.17469564898373324
    assert float(combat["relative_improvement"]) == 0.2593504619127258
    assert bmmc["inference_role"] == "retrospective_adaptive_development"
    assert combat["inference_role"] == "retrospective_adaptive_development"

    gse314 = by_panel["gse314416_pilot_terminal"]
    assert gse314["decision"] == "TERMINAL_PILOT_REFUSAL"
    assert gse314["outcome_scored"] == "YES"

    sequence_counts = {
        panel_id: sum(row["panel_id"] == panel_id for row in sequence)
        for panel_id in (
            "scmmib_bmmc_terminal",
            "gse279451_terminal",
            "gse299043_terminal",
        )
    }
    assert sequence_counts == {
        "scmmib_bmmc_terminal": 4,
        "gse279451_terminal": 4,
        "gse299043_terminal": 4,
    }


def test_kotliarov_binary_v2_is_a_source_execution_refusal() -> None:
    panels = _rows(PANELS_PATH)
    comparisons = _rows(COMPARISONS_PATH)
    sequence = _rows(SEQUENCE_PATH)
    panel_id = "kotliarov_pbmc_binary_v2_source_terminal"
    panel = next(row for row in panels if row["panel_id"] == panel_id)

    assert panel["analysis_phase"] == "source_development_gate"
    assert panel["inference_role"] == "procedural_refusal"
    assert panel["decision"] == "TERMINAL_SOURCE_EXECUTION_REFUSAL"
    assert panel["outcome_scored"] == "NO"
    assert panel["primary_value"] == ""
    assert not any(row["panel_id"] == panel_id for row in comparisons)

    stages = [row for row in sequence if row["panel_id"] == panel_id]
    assert [row["stage"] for row in stages] == [
        "source_freeze",
        "source_authorization",
        "source_attempt",
        "source_result",
        "postrun_access_certificate",
    ]
    assert stages[-2]["status"] == "TERMINAL_SOURCE_EXECUTION_REFUSAL"
    assert stages[-2]["outcome_access"] == "DEVELOPMENT_ONLY_HELD_ADT_UNAUTHORIZED"
    assert stages[-2]["artifact_sha256"] == (
        "12aacf4dc05efabcd2d745abc0319f6a2676e5d26eb50054849005424b1a071c"
    )
    assert stages[-1]["status"] == (
        "DETERMINISTIC_CODE_PATH_AUDIT_HELD_ADT_UNREACHABLE"
    )
    assert stages[-1]["artifact_sha256"] == (
        "1fed7f94958a07a71a195e80ce2b88f326ff2f47274733133c6b4f7dfd47d0d6"
    )


def test_gse179221_is_a_zero_numeric_source_feature_axis_refusal() -> None:
    panels = _rows(PANELS_PATH)
    comparisons = _rows(COMPARISONS_PATH)
    sequence = _rows(SEQUENCE_PATH)
    panel_id = "gse179221_bmmc_source_terminal"
    panel = next(row for row in panels if row["panel_id"] == panel_id)

    assert panel["analysis_phase"] == "source_feature_axis_preflight"
    assert panel["inference_role"] == "procedural_refusal"
    assert panel["decision"] == "TERMINAL_SOURCE_EXECUTION_REFUSAL"
    assert panel["outcome_scored"] == "NO"
    for field in (
        "primary_metric",
        "metric_direction",
        "primary_value",
        "ci_95_low",
        "ci_95_high",
    ):
        assert panel[field] == ""
    assert not any(row["panel_id"] == panel_id for row in comparisons)

    result_path = ROOT / "results/development/gse179221_bmmc_source_v1.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["reason_code"] == "COGNATE_AXIS_NOT_EXACTLY_UNIQUE"
    assert result["passes_source_promotion_gate"] is False
    assert result["held_h5_access_authorized"] is False
    assert result["held_h5_access_eligible_after_public_source_pass"] is False
    assert "comparisons" not in result
    assert "models" not in result
    access = result["access_audit"]
    assert access["source_h5_get_count"] == 1
    assert access["source_h5_deleted_count"] == 1
    assert access["held_h5_get_count"] == 0
    assert access["all_donor_tar_get_count"] == 0
    assert access["maximum_simultaneous_h5_files"] == 1
    assert len(access["source_files"]) == 1
    assert access["source_files"][0]["decoded_h5_datasets"] == [
        "matrix/barcodes",
        "matrix/features/feature_type",
        "matrix/features/name",
    ]
    assert access["source_files"][0]["deleted_after_reduction"] is True
    assert not any((ROOT / relative).exists() for relative in ABSENT_GSE179221_DOWNSTREAM_PATHS)

    stages = [row for row in sequence if row["panel_id"] == panel_id]
    assert [row["stage"] for row in stages] == [
        "protocol",
        "candidate_designation",
        "implementation_amendment",
        "implementation",
        "source_attempt",
        "source_consumption",
        "source_result",
    ]
    assert stages[-1]["outcome_access"] == (
        "ONE_SOURCE_FEATURE_AXIS_OPENED_NO_COUNT_DATASET_HELD_UNOPENED"
    )
    assert stages[-1]["artifact_sha256"] == (
        "18982f0320c602dbc65df27a94675677dc006edd9951ac62fa3a1ad93e2a06f6"
    )


def test_gse214546_is_a_terminal_source_support_refusal() -> None:
    panels = _rows(PANELS_PATH)
    comparisons = _rows(COMPARISONS_PATH)
    sequence = _rows(SEQUENCE_PATH)
    panel_id = "gse214546_teaseq_source_terminal"
    panel = next(row for row in panels if row["panel_id"] == panel_id)

    assert panel["analysis_phase"] == "source_support_gate"
    assert panel["inference_role"] == "procedural_refusal"
    assert panel["evaluation_unit"] == "physical source donor"
    assert panel["unit_count"] == "2"
    assert panel["entity_count"] == "53"
    assert panel["decision"] == "TERMINAL_SOURCE_REFUSAL"
    assert panel["outcome_scored"] == "NO"
    assert panel["result_sha256"] == (
        "fb7ed8218c926cbc41a105b21a94116d8f73de5fd823b98137ac094b20d410ba"
    )
    for field in (
        "primary_metric",
        "metric_direction",
        "primary_value",
        "ci_95_low",
        "ci_95_high",
    ):
        assert panel[field] == ""
    assert not any(row["panel_id"] == panel_id for row in comparisons)

    stages = [row for row in sequence if row["panel_id"] == panel_id]
    assert [row["stage"] for row in stages] == [
        "candidate_protocol",
        "schema_amendment",
        "implementation_clarification",
        "cv_availability",
        "normalization_correction",
        "sparse_access_clarification",
        "crash_semantics",
        "implementation",
        "source_attempt",
        "source_result",
    ]
    assert [row["stage_ordinal"] for row in stages] == [str(i) for i in range(1, 11)]
    assert all(row["public_before_outcome"] == "YES" for row in stages[:-1])
    assert stages[-2]["public_commit_or_tag"] == (
        "gse214546-teaseq-v1-source-attempt"
    )
    assert stages[-1]["public_commit_or_tag"] == (
        "gse214546-teaseq-v1-source-refusal"
    )
    assert stages[-1]["outcome_access"] == (
        "TWO_SOURCE_H5_REQUESTED_ONE_REDUCED_HELD_UNOPENED"
    )
    assert stages[-1]["artifact_sha256"] == panel["result_sha256"]

    result = json.loads(
        (ROOT / "results/development/gse214546_teaseq_source_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["reason_code"] == "FEWER_THAN_512_MATCHED_SINGLETS"
    assert result["details"] == {}
    assert result["held_h5_requested"] is False
    assert result["rerun_permitted"] is False
    audit = result["access_audit"]
    assert [record["filename"] for record in audit] == [
        "GSM6611363_B065-P1_PB00593-04_filtered_metadata.csv.gz",
        "GSM6611363_B065-P1_PB00593-04.h5",
        "GSM6611365_B076-P1_PB00368-04_filtered_metadata.csv.gz",
        "GSM6611365_B076-P1_PB00368-04.h5",
    ]
    assert sum(record["observed_bytes"] for record in audit) == 159_784_214
    assert all(
        record["request_started"] and record["completed"] and record["deleted"]
        for record in audit
    )
    assert audit[0]["decode"] == {
        "barcode_column": "barcodes",
        "literal_true_singlets": 10_295,
        "rows": 11_191,
        "singlet_column": "singlet",
        "singlet_value": "TRUE",
        "unique_barcodes": 11_191,
    }
    assert audit[1]["h5_reduction_completed"] is True
    assert audit[1]["selected_cells"] == 512
    assert audit[1]["authorized_marker_count"] == 53
    assert len(audit[1]["dataset_access_events"]) == 3_078
    assert audit[2]["decode"]["literal_true_singlets"] == 25_364
    assert audit[3]["h5_reduction_completed"] is False
    assert set(audit[3]["datasets_read"]) == {
        "matrix/barcodes",
        "matrix/features/feature_type",
        "matrix/features/name",
        "matrix/shape",
    }
    assert len(audit[3]["dataset_access_events"]) == 4
    assert "selected_cells" not in audit[3]
    attempt = json.loads(
        (
            ROOT
            / "data/confirmation/gse214546_teaseq/source_attempt_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert len(attempt["held_gsms"]) == 8
    assert not any(
        gsm in record["filename"]
        for gsm in attempt["held_gsms"]
        for record in audit
    )
    assert (
        f'{panel["result_sha256"]}  {panel["result_artifact"]}\n'
        in (ROOT / CHECKSUM_PATH).read_text(encoding="utf-8")
    )


def test_unused_cambridge_terminal_row_contains_no_performance_claim() -> None:
    panels = _rows(PANELS_PATH)
    sequence = _rows(SEQUENCE_PATH)
    pending = next(
        row for row in panels if row["panel_id"] == "stephenson_unused_cambridge_terminal"
    )
    assert pending["inference_role"] == "infrastructure_unevaluable"
    assert pending["outcome_scored"] == "NO"
    assert pending["primary_value"] == ""
    assert "INFRASTRUCTURE" in pending["decision"]
    final_stage = next(
        row
        for row in sequence
        if row["sequence_id"]
        == "stephenson_unused_cambridge_terminal__04_prediction_and_score"
    )
    assert final_stage["artifact"] == (
        "data/confirmation/stephenson_unused_cambridge/"
        "prediction_terminal_record_v1_1.json"
    )
    assert final_stage["artifact_sha256"] == (
        "288cb761236235a0e6d3df3d3984568a108648548cd5e1c5c380c6b75be51042"
    )
    assert "INFRASTRUCTURE" in final_stage["status"]

    checksums = (ROOT / CHECKSUM_PATH).read_text(encoding="utf-8")
    assert "prediction_terminal_record_v1_1.json" in checksums
    assert "stephenson_unused_cambridge_predictions_v1_1.json" not in checksums
    for relative in (
        "experiments/correct_gse239452_residual_inversion.py",
        "results/gse239452_citeseq_post_access_correction.json",
        "tests/test_gse239452_post_access_correction.py",
    ):
        assert f"  {relative}\n" in checksums


def test_manifest_counts_and_claim_boundaries_match_ledgers() -> None:
    manifest = json.loads((ROOT / MANIFEST_PATH).read_text(encoding="utf-8"))
    assert manifest["schema"] == "coupling-fields-public-benchmark/2.0"
    assert manifest["snapshot_date"] == "2026-08-31"
    assert manifest["counts"] == {
        "comparison_records": 29,
        "panel_records": 37,
        "infrastructure_unevaluable_records": 1,
        "pending_records": 0,
        "procedural_refusal_records": 24,
        "scored_panel_records": 12,
        "sequence_records": 91,
    }
    assert manifest["archive_doi"] is None
    assert manifest["code_license"] == "MIT"
    assert "not registry-hosted" in manifest["analysis_plan_characterization"]
    assert (
        manifest["infrastructure_unevaluable"]["performance_values_recorded"]
        is False
    )
    assert manifest["infrastructure_unevaluable"]["scientific_decision"] is None
