"""Build the metric-aware public benchmark release ledgers.

The version-1 table is retained as a historical input. This builder normalizes
its heterogeneous metrics, adds the later completed analyses, and records
procedural attempts without assigning them performance values.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PANELS_PATH = Path("results/benchmark_panels_v2.tsv")
COMPARISONS_PATH = Path("results/benchmark_comparisons_v2.tsv")
SEQUENCE_PATH = Path("results/benchmark_sequence_v2.tsv")
MANIFEST_PATH = Path("benchmark_manifest.json")
CHECKSUM_PATH = Path("SHA256SUMS")

PANEL_FIELDS = (
    "panel_id",
    "panel",
    "accession",
    "assay_pair",
    "analysis_phase",
    "inference_role",
    "evaluation_unit",
    "unit_count",
    "entity_count",
    "primary_method",
    "primary_metric",
    "metric_direction",
    "primary_value",
    "ci_95_low",
    "ci_95_high",
    "decision",
    "outcome_scored",
    "result_artifact",
    "result_sha256",
    "notes",
)

COMPARISON_FIELDS = (
    "comparison_id",
    "panel_id",
    "analysis_phase",
    "inference_role",
    "comparison_role",
    "primary_method",
    "comparator_method",
    "metric",
    "metric_direction",
    "primary_value",
    "comparator_value",
    "primary_minus_comparator",
    "paired_difference_ci_95_low",
    "paired_difference_ci_95_high",
    "relative_improvement",
    "relative_improvement_ci_95_low",
    "relative_improvement_ci_95_high",
    "favorable_units",
    "total_units",
    "p_value",
    "decision",
    "result_artifact",
    "result_sha256",
    "notes",
)

SEQUENCE_FIELDS = (
    "sequence_id",
    "panel_id",
    "stage_ordinal",
    "stage",
    "status",
    "outcome_access",
    "public_before_outcome",
    "artifact",
    "artifact_sha256",
    "public_commit_or_tag",
    "notes",
)

LEGACY_IDS = {
    "NeurIPS 2021 BMMC CITE-seq": "scmmib_bmmc_terminal",
    "GSE279451 adult sepsis CITE-seq": "gse279451_terminal",
    "GSE299043 MLN held-site confirmation": "gse299043_terminal",
    "PerturbSci-Kinetics": "perturbsci_kinetics",
    "Frangieh Perturb-CITE-seq": "frangieh_perturb_citeseq",
    "Papalexi ECCITE-seq": "papalexi_ecciteseq",
    "MultiPerturb-seq RNA-ATAC": "multiperturb_rna_atac",
    "PerturbFate": "perturbfate",
    "ReSisTrace": "resistrace",
    "Arce T-cell RNA-protein confirmation": "arce_gse278572",
    "PoKI-seq held-donor confirmation": "poki_gse143417_terminal",
    "Lawlor HCA PBMC confirmation": "lawlor_hca_terminal",
    "Hao held-donor confirmation": "hao_gse164378_terminal",
    "Kotliarov PBMC held-batch confirmation": "kotliarov_pbmc_terminal",
}

SOURCE_TERMINALS = (
    (
        "gse158769_source_terminal",
        "GSE158769 CITE-seq source campaign",
        "GSE158769",
        "RNA-ADT coupling transfer",
        "results/development/gse158769_development_v1.json",
    ),
    (
        "gse164378_3p_gse155673_source_terminal",
        "GSE164378 3-prime to GSE155673 external-study campaign",
        "GSE164378; GSE155673",
        "cross-study RNA-ADT coupling transfer",
        "results/development/gse164378_3p_gse155673_source_v1.json",
    ),
    (
        "gse185381_aml_source_terminal",
        "GSE185381 control-to-AML campaign",
        "GSE185381",
        "control-to-AML RNA-ADT coupling transfer",
        "results/development/gse185381_aml_source_v1.json",
    ),
    (
        "gse189050_source_terminal",
        "GSE189050 SLE held-pool campaign",
        "GSE189050",
        "RNA-ADT coupling transfer",
        "results/development/gse189050_development_v1.json",
    ),
    (
        "gse202150_source_terminal",
        "GSE202150 acute-infection campaign",
        "GSE202150",
        "RNA-ADT coupling transfer",
        "results/development/gse202150_source_development_v1.json",
    ),
    (
        "gse288020_source_terminal",
        "GSE288020 MGUS-to-myeloma campaign",
        "GSE288020",
        "MGUS-to-myeloma RNA-ADT coupling transfer",
        "results/development/gse288020_development_v1.json",
    ),
    (
        "gse309593_source_terminal",
        "GSE309593 held-batch campaign",
        "GSE309593",
        "RNA-ADT coupling transfer",
        "results/development/gse309593_held_batches_source_v1.json",
    ),
    (
        "gse326573_source_terminal",
        "GSE326573 lung CITE-seq campaign",
        "GSE326573",
        "RNA-ADT coupling transfer",
        "results/development/gse326573_lung_source_v1.json",
    ),
    (
        "gse334503_source_terminal",
        "GSE334503 batch-to-batch source campaign",
        "GSE334503",
        "batch-to-batch RNA-ADT coupling transfer",
        "results/development/gse334503_source_terminal_decision_v1.json",
    ),
    (
        "gse144744_source_terminal",
        "GSE144744 multiple-sclerosis held-cohort campaign",
        "GSE144744",
        "RNA-ADT coupling transfer",
        "results/development/gse144744_ms_source_v1.json",
    ),
    (
        "gse181897_source_terminal",
        "GSE181897 control CITE-seq campaign",
        "GSE181897",
        "control RNA-ADT coupling transfer",
        "results/development/gse181897_source_model_terminal_v1.json",
    ),
)

# No artifact was produced at these downstream paths. Keeping them out of the
# checksum candidate set makes that absence explicit and harmless if a local
# scratch file later appears under one of the names.
ABSENT_UNUSED_CAMBRIDGE_PATHS = {
    "results/stephenson_unused_cambridge_predictions_v1_1.json",
    "data/confirmation/stephenson_unused_cambridge/score_authorization_v1_1.json",
    "data/confirmation/stephenson_unused_cambridge/score_attempt_v1_1.json",
    "results/stephenson_unused_cambridge_confirmation_v1_1.json",
}

ABSENT_GSE179221_DOWNSTREAM_PATHS = {
    "data/confirmation/gse179221_bmmc/held_margin_attempt_v1.json",
    "data/confirmation/gse179221_bmmc/held_margin_consumption_v1.json",
    "data/confirmation/gse179221_bmmc/private_held_rna_states_v1.npz",
    "data/confirmation/gse179221_bmmc/private_held_adt_states_v1.npz",
    "data/confirmation/gse179221_bmmc/score_authorization_v1.json",
    "data/confirmation/gse179221_bmmc/score_attempt_v1.json",
    "results/gse179221_bmmc_held_margins_v1.json",
    "results/gse179221_bmmc_predictions_v1.json",
    "results/gse179221_bmmc_confirmation_v1.json",
}

GSE317605_RESULT_PATH = (
    "data/confirmation/gse317605_longitudinal/calibration_result_v1.json"
)
GSE317605_RESULT_SHA256 = (
    "9b3fcec43e38d876a312c8488292264ed747a9c70b4a75609f1d9ac18948040e"
)
GSE317605_RESULT_TAG = "gse317605-longitudinal-v1-calibration-result"
GSE317605_RESULT_COMMIT = "7f229baca0b261e1a0ee832defcc5cfa96aad023"


def _reject_nonfinite(token: str) -> None:
    raise ValueError(f"non-finite JSON number: {token}")


def _load_json(relative: str | Path) -> dict[str, Any]:
    return json.loads(
        (ROOT / relative).read_text(encoding="utf-8"),
        parse_constant=_reject_nonfinite,
    )


def _sha256(relative: str | Path) -> str:
    digest = hashlib.sha256()
    with (ROOT / relative).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: str) -> float | None:
    if value in {"", "NA", "not scored"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _artifact_fields(relative: str) -> tuple[str, str]:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(relative)
    return relative, _sha256(relative)


def _legacy_role(panel: str, scored: bool) -> str:
    if panel == "Arce T-cell RNA-protein confirmation":
        return "analytical_holdout_post_access_correction"
    return "retrospective_public_benchmark" if scored else "procedural_refusal"


def _legacy_panels() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = ROOT / "results/final_public_benchmark_table.tsv"
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if {row["panel"] for row in rows} != set(LEGACY_IDS):
        raise ValueError("historical benchmark panel set changed")

    panels: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for row in rows:
        panel_id = LEGACY_IDS[row["panel"]]
        result_path, result_sha = _artifact_fields(row["result_artifact"])
        if result_sha != row["result_sha256"]:
            raise ValueError(f"historical result hash changed: {result_path}")
        primary_value = _number(row["primary_r"])
        scored = primary_value is not None or row["panel"] == "ReSisTrace"
        role = _legacy_role(row["panel"], scored)
        panels.append(
            {
                "panel_id": panel_id,
                "panel": row["panel"],
                "accession": row["accession"],
                "assay_pair": row["assay_pair"],
                "analysis_phase": "held_or_replicate_evaluation" if scored else "terminal_pre_score",
                "inference_role": role,
                "evaluation_unit": row["replication_unit"],
                "unit_count": "",
                "entity_count": row["n_targets"],
                "primary_method": row["primary_method"],
                "primary_metric": row["primary_metric"] if scored else "",
                "metric_direction": "higher" if primary_value is not None else "",
                "primary_value": primary_value,
                "ci_95_low": _number(row["primary_r_ci_low"]),
                "ci_95_high": _number(row["primary_r_ci_high"]),
                "decision": row["panel_decision"],
                "outcome_scored": "YES" if scored else "NO",
                "result_artifact": result_path,
                "result_sha256": result_sha,
                "notes": row["notes"],
            }
        )

        primary_rmse = _number(row["primary_standardized_rmse"])
        baseline_rmse = _number(row["strongest_baseline_standardized_rmse"])
        if primary_rmse is not None and baseline_rmse is not None:
            comparisons.append(
                {
                    "comparison_id": f"{panel_id}__declared_baseline_rmse",
                    "panel_id": panel_id,
                    "analysis_phase": "held_or_replicate_evaluation",
                    "inference_role": role,
                    "comparison_role": "declared_baseline",
                    "primary_method": row["primary_method"],
                    "comparator_method": row["strongest_declared_baseline"],
                    "metric": "standardized_rmse",
                    "metric_direction": "lower",
                    "primary_value": primary_rmse,
                    "comparator_value": baseline_rmse,
                    "primary_minus_comparator": primary_rmse - baseline_rmse,
                    "paired_difference_ci_95_low": "",
                    "paired_difference_ci_95_high": "",
                    "relative_improvement": (baseline_rmse - primary_rmse) / baseline_rmse,
                    "relative_improvement_ci_95_low": "",
                    "relative_improvement_ci_95_high": "",
                    "favorable_units": "",
                    "total_units": "",
                    "p_value": "",
                    "decision": row["estimator_superiority_decision"],
                    "result_artifact": result_path,
                    "result_sha256": result_sha,
                    "notes": "Historical declared-baseline comparison; see the source artifact for its resampling contract.",
                }
            )

        pairing_low = _number(row["pairing_control_ci_low"])
        pairing_high = _number(row["pairing_control_ci_high"])
        if row["pairing_control"] != "NA" and (
            pairing_low is not None or row["pairing_control_estimate"] != "NA"
        ):
            comparisons.append(
                {
                    "comparison_id": f"{panel_id}__pairing_control",
                    "panel_id": panel_id,
                    "analysis_phase": "held_or_replicate_evaluation",
                    "inference_role": role,
                    "comparison_role": "pairing_control",
                    "primary_method": row["primary_method"],
                    "comparator_method": row["pairing_control"],
                    "metric": row["pairing_control"],
                    "metric_direction": "higher",
                    "primary_value": "",
                    "comparator_value": "",
                    "primary_minus_comparator": _number(row["pairing_control_estimate"]),
                    "paired_difference_ci_95_low": pairing_low,
                    "paired_difference_ci_95_high": pairing_high,
                    "relative_improvement": "",
                    "relative_improvement_ci_95_low": "",
                    "relative_improvement_ci_95_high": "",
                    "favorable_units": "",
                    "total_units": "",
                    "p_value": "",
                    "decision": row["pairing_signal_decision"],
                    "result_artifact": result_path,
                    "result_sha256": result_sha,
                    "notes": "Historical pairing-control summary; empty scalar cells denote a contrast reported only by interval or text.",
                }
            )
    return panels, comparisons


def _loss_panel(
    *,
    panel_id: str,
    panel: str,
    accession: str,
    assay_pair: str,
    phase: str,
    role: str,
    unit: str,
    units: int,
    entities: int,
    method: str,
    value: float,
    decision: str,
    artifact: str,
    notes: str,
) -> dict[str, Any]:
    result_path, result_sha = _artifact_fields(artifact)
    return {
        "panel_id": panel_id,
        "panel": panel,
        "accession": accession,
        "assay_pair": assay_pair,
        "analysis_phase": phase,
        "inference_role": role,
        "evaluation_unit": unit,
        "unit_count": units,
        "entity_count": entities,
        "primary_method": method,
        "primary_metric": "mean_poisson_deviance_per_cell",
        "metric_direction": "lower",
        "primary_value": value,
        "ci_95_low": "",
        "ci_95_high": "",
        "decision": decision,
        "outcome_scored": "YES",
        "result_artifact": result_path,
        "result_sha256": result_sha,
        "notes": notes,
    }


def _loss_comparison(
    *,
    comparison_id: str,
    panel_id: str,
    phase: str,
    role: str,
    comparison_role: str,
    primary_method: str,
    comparator_method: str,
    primary_value: float,
    comparator_value: float,
    ci: list[float],
    relative_improvement: float,
    artifact: str,
    favorable_units: int | str,
    total_units: int | str,
    p_value: float | str = "",
    relative_ci: list[float] | None = None,
    decision: str,
    notes: str,
    metric: str = "mean_poisson_deviance_per_cell",
) -> dict[str, Any]:
    result_path, result_sha = _artifact_fields(artifact)
    return {
        "comparison_id": comparison_id,
        "panel_id": panel_id,
        "analysis_phase": phase,
        "inference_role": role,
        "comparison_role": comparison_role,
        "primary_method": primary_method,
        "comparator_method": comparator_method,
        "metric": metric,
        "metric_direction": "lower",
        "primary_value": primary_value,
        "comparator_value": comparator_value,
        "primary_minus_comparator": primary_value - comparator_value,
        "paired_difference_ci_95_low": ci[0],
        "paired_difference_ci_95_high": ci[1],
        "relative_improvement": relative_improvement,
        "relative_improvement_ci_95_low": "" if relative_ci is None else relative_ci[0],
        "relative_improvement_ci_95_high": "" if relative_ci is None else relative_ci[1],
        "favorable_units": favorable_units,
        "total_units": total_units,
        "p_value": p_value,
        "decision": decision,
        "result_artifact": result_path,
        "result_sha256": result_sha,
        "notes": notes,
    }


def _new_completed_evidence() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    panels: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []

    stephenson_path = "results/stephenson_citeseq_confirmation.json"
    stephenson = _load_json(stephenson_path)
    if stephenson["status"] != "CONFIRMATION_PASS":
        raise ValueError("Stephenson confirmation status changed")
    steph_primary = stephenson["comparisons"]["best_residual"]["primary_mean_deviance_per_cell"]
    panels.append(
        _loss_panel(
            panel_id="stephenson_newcastle_confirmation",
            panel="Stephenson Newcastle held-site confirmation",
            accession="E-MTAB-10026",
            assay_pair="same-cell RNA and surface-protein binary states",
            phase="held_site_confirmation",
            role="confirmatory",
            unit="physical sample",
            units=56,
            entities=81,
            method="hierarchical_exact_conditional_coupling",
            value=steph_primary,
            decision=stephenson["status"],
            artifact=stephenson_path,
            notes="Publicly frozen prediction preceded paired held-site scoring; the gate passed against the selected residual and destroyed-link controls.",
        )
    )
    for key, comparator, role_name in (
        ("best_residual", "selected_signed_deviance_residual", "confirmatory_gate"),
        ("destroyed_link", "destroyed_link", "confirmatory_gate"),
    ):
        record = stephenson["comparisons"][key]
        comparisons.append(
            _loss_comparison(
                comparison_id=f"stephenson_newcastle_confirmation__{key}",
                panel_id="stephenson_newcastle_confirmation",
                phase="held_site_confirmation",
                role="confirmatory",
                comparison_role=role_name,
                primary_method="hierarchical_exact_conditional_coupling",
                comparator_method=comparator,
                primary_value=record["primary_mean_deviance_per_cell"],
                comparator_value=record["comparator_mean_deviance_per_cell"],
                ci=record["paired_difference_95_ci"],
                relative_improvement=record["relative_reduction"],
                artifact=stephenson_path,
                favorable_units=record["favorable_samples"],
                total_units=record["sign_test"]["donors"],
                p_value=record["sign_test"]["one_sided_p"],
                decision="PASS" if record["passes"] else "REFUSE",
                notes="Prospectively frozen held-site gate; percentile paired bootstrap and exact one-sided sign test over physical samples.",
            )
        )

    gse239_sealed_path = "results/gse239452_citeseq_confirmation.json"
    gse239_sealed = _load_json(gse239_sealed_path)
    gse239_correction_path = "results/gse239452_citeseq_post_access_correction.json"
    gse239_correction = _load_json(gse239_correction_path)
    correction_bindings = gse239_correction.get("original_sealed_artifacts", {})
    correction_audit = gse239_correction.get("held", {}).get(
        "residual_reconstruction_audit", {}
    )
    correction_samples = gse239_correction.get("held", {}).get("samples", [])
    if (
        gse239_sealed.get("status") != "HELD_PASS"
        or gse239_correction.get("schema")
        != "gse239452-post-access-correction/1.0"
        or gse239_correction.get("status") != "POST_ACCESS_CORRECTION_COMPLETE"
        or gse239_correction.get("outcome_blind") is not False
        or gse239_correction.get("original_sealed_artifacts_overwritten") is not False
        or correction_bindings.get("held_score", {}).get("sha256")
        != _sha256(gse239_sealed_path)
        or correction_bindings.get("held_predictions", {}).get("sha256")
        != _sha256("results/gse239452_citeseq_predictions.json")
        or gse239_correction.get("corrected_runner", {}).get("sha256")
        != _sha256("experiments/confirm_gse239452_citeseq.py")
        or gse239_correction.get("correction_runner", {}).get("sha256")
        != _sha256("experiments/correct_gse239452_residual_inversion.py")
        or gse239_correction.get("held", {}).get("gate", {}).get("passes") is not True
        or correction_audit.get("tables_checked") != 729
        or correction_audit.get("original_coordinate_mismatches") != 80
        or correction_audit.get("corrected_coordinate_mismatches") != 0
        or len(correction_samples) != 9
        or any(
            sample.get("primary_prediction_matches_original") is not True
            for sample in correction_samples
        )
    ):
        raise ValueError("GSE239452 post-access correction boundary changed")
    gse239_gate = gse239_correction["held"]["gate"]
    gse239_residual = gse239_gate["comparisons"]["best_residual"]
    if (
        gse239_residual.get("primary_mean_loss") != 0.008506365049036143
        or gse239_residual.get("comparator_mean_loss") != 0.01413148577805464
        or gse239_residual.get("relative_deviance_reduction")
        != 0.39805586032248474
        or gse239_residual.get("paired_bootstrap_95_ci")
        != [-0.007095088337784283, -0.0042321214194290585]
        or gse239_residual.get("favorable_donors") != 9
        or gse239_residual.get("passes_all") is not True
    ):
        raise ValueError("GSE239452 corrected residual comparison changed")
    panels.append(
        _loss_panel(
            panel_id="gse239452_held_post_access_correction",
            panel="GSE239452 held-cohort post-access correction",
            accession="GSE239452",
            assay_pair="same-cell RNA and surface-protein binary states",
            phase="held_cohort_analysis",
            role="post_access_correction",
            unit="physical donor",
            units=len(correction_samples),
            entities=81,
            method="hierarchical_exact_conditional_coupling",
            value=gse239_residual["primary_mean_loss"],
            decision="HELD_PASS",
            artifact=gse239_correction_path,
            notes=(
                "The aggregate uses the post-access numerical correction while "
                "preserving the original sealed prediction and score chronology; "
                "this is not prospective confirmation."
            ),
        )
    )
    for key, comparator in (
        ("best_residual", "selected_classical_residual"),
        ("destroyed_link", "destroyed_link"),
    ):
        record = gse239_gate["comparisons"][key]
        comparisons.append(
            _loss_comparison(
                comparison_id=f"gse239452_held_post_access_correction__{key}",
                panel_id="gse239452_held_post_access_correction",
                phase="held_cohort_analysis",
                role="post_access_correction",
                comparison_role="post_access_gate",
                primary_method="hierarchical_exact_conditional_coupling",
                comparator_method=comparator,
                primary_value=record["primary_mean_loss"],
                comparator_value=record["comparator_mean_loss"],
                ci=record["paired_bootstrap_95_ci"],
                relative_improvement=record["relative_deviance_reduction"],
                artifact=gse239_correction_path,
                favorable_units=record["favorable_donors"],
                total_units=len(correction_samples),
                decision="PASS" if record["passes_all"] else "REFUSE",
                notes=(
                    "Post-access numerically corrected held-cohort comparison; "
                    "intervals resample physical donors."
                ),
            )
        )

    classical_path = "results/development/classical_interaction_baselines_posthoc.json"
    classical = _load_json(classical_path)
    if classical["status"] != "POST_HOC_NONCONFIRMATORY_BASELINE_AUDIT":
        raise ValueError("classical baseline audit status changed")
    classical_specs = (
        (
            "stephenson_newcastle_confirmation",
            "stephenson_newcastle_held_site",
            "primary_vs_common_effect_exact_cmle",
            "common_effect_exact_cmle",
            "primary_vs_common_effect_exact_cmle",
        ),
        (
            "stephenson_newcastle_confirmation",
            "stephenson_newcastle_held_site",
            "primary_vs_pooled_poisson_loglinear_interaction",
            "pooled_table_log_odds_with_conditional_reconstruction",
            "primary_vs_pooled_table_log_odds_conditional_reconstruction",
        ),
        (
            "gse239452_held_post_access_correction",
            "gse239452_held_cohort_post_access_correction",
            "primary_vs_common_effect_exact_cmle",
            "common_effect_exact_cmle",
            "primary_vs_common_effect_exact_cmle",
        ),
        (
            "gse239452_held_post_access_correction",
            "gse239452_held_cohort_post_access_correction",
            "primary_vs_pooled_poisson_loglinear_interaction",
            "pooled_table_log_odds_with_conditional_reconstruction",
            "primary_vs_pooled_table_log_odds_conditional_reconstruction",
        ),
    )
    for panel_id, study, artifact_key, comparator, comparison_suffix in classical_specs:
        record = classical["studies"][study]["comparisons"][artifact_key]
        comparisons.append(
            _loss_comparison(
                comparison_id=f"{panel_id}__posthoc_{comparison_suffix}",
                panel_id=panel_id,
                phase="post_hoc_classical_baseline_audit",
                role="post_hoc_nonconfirmatory",
                comparison_role="classical_head_to_head",
                primary_method="hierarchical_exact_conditional_coupling",
                comparator_method=comparator,
                primary_value=record["primary_mean_loss"],
                comparator_value=record["comparator_mean_loss"],
                ci=record["paired_difference_95_ci"],
                relative_improvement=record["relative_loss_reduction"],
                relative_ci=record["relative_loss_reduction_paired_bootstrap_95_ci"],
                artifact=classical_path,
                favorable_units=record["favorable_donors"],
                total_units=record["units"],
                p_value=record["exact_one_sided_sign_p"],
                decision="DESCRIPTIVE",
                notes="Definitions and comparisons were executed after held outcomes had been accessed.",
            )
        )

    standard_poisson_path = (
        "results/development/gse239452_standard_poisson_interaction_posthoc.json"
    )
    standard_poisson = _load_json(standard_poisson_path)
    standard_held = standard_poisson.get("held", {})
    standard_comparison = standard_held.get("comparison", {})
    standard_development = standard_poisson.get("development", {})
    standard_certificate = standard_development.get("refit_certificate", {})
    standard_streaming = standard_poisson.get("streaming_access", {})
    standard_samples = standard_held.get("samples", [])
    legacy_path = "results/development/classical_interaction_baselines_posthoc.json"
    if (
        standard_poisson.get("status") != "POST_HOC_NONCONFIRMATORY_HEAD_TO_HEAD"
        or standard_poisson.get("confirmatory") is not False
        or standard_poisson.get("reason_post_hoc")
        != (
            "The standard-Poisson reconstruction and comparison were defined "
            "after held outcomes had been accessed."
        )
        or standard_poisson.get("method", {}).get(
            "noncentral_hypergeometric_reconstruction_used"
        )
        is not False
        or standard_poisson.get("bindings", {}).get("legacy_mislabeled_audit_sha256")
        != _sha256(legacy_path)
        or standard_development.get("selected_alpha") != 1.0
        or standard_development.get("held_donors_used_for_selection") != []
        or standard_certificate.get("saturated_tables_reconstructed") != 81
        or standard_certificate.get("maximum_normalized_saturated_cell_error")
        > 1e-10
        or standard_comparison.get("post_hoc_nonconfirmatory") is not True
        or standard_comparison.get("units") != 9
        or standard_comparison.get("favorable_donors_primary_lower") != 9
        or standard_comparison.get("exact_one_sided_sign_test_p") != 1 / 512
        or len(standard_samples) != 9
        or any(
            sample.get("primary_loss") >= sample.get("standard_poisson_loss")
            for sample in standard_samples
        )
        or standard_streaming.get("official_individual_file_pairs") != 9
        or standard_streaming.get("held_truth_hashes_reproduced") != 9
        or standard_streaming.get("held_truth_tables_serialized") is not False
        or standard_streaming.get("raw_files_retained_after_each_donor") is not False
        or len(standard_streaming.get("official_archive_and_h5ad_provenance", []))
        != 9
        or any(
            len(donor.get("files", [])) != 2
            for donor in standard_streaming.get(
                "official_archive_and_h5ad_provenance", []
            )
        )
    ):
        raise ValueError("GSE239452 standard Poisson post-hoc boundary changed")
    comparisons.append(
        _loss_comparison(
            comparison_id=(
                "gse239452_held_post_access_correction__"
                "posthoc_standard_fixed_interaction_poisson"
            ),
            panel_id="gse239452_held_post_access_correction",
            phase="post_hoc_standard_poisson_head_to_head",
            role="post_hoc_nonconfirmatory",
            comparison_role="classical_head_to_head",
            primary_method="hierarchical_exact_conditional_coupling",
            comparator_method="standard_pooled_saturated_poisson_fixed_interaction",
            primary_value=standard_comparison["primary_mean_loss"],
            comparator_value=standard_comparison["standard_poisson_mean_loss"],
            ci=standard_comparison["paired_difference_95_percentile_ci"],
            relative_improvement=standard_comparison[
                "relative_loss_reduction_primary_vs_poisson"
            ],
            relative_ci=standard_comparison[
                "relative_loss_reduction_95_percentile_ci"
            ],
            artifact=standard_poisson_path,
            favorable_units=standard_comparison["favorable_donors_primary_lower"],
            total_units=standard_comparison["units"],
            p_value=standard_comparison["exact_one_sided_sign_test_p"],
            decision="DESCRIPTIVE",
            notes=(
                "Post-hoc nonconfirmatory comparison defined after held-outcome "
                "access. Alpha was selected on development donors only; held "
                "inference resamples nine physical donors."
            ),
            metric="mean_multinomial_deviance_per_cell",
        )
    )

    gse314_path = "results/development/gse314416_citeseq_development.json"
    gse314 = _load_json(gse314_path)
    if gse314["status"] != "TERMINAL_PILOT_REFUSAL":
        raise ValueError("GSE314416 pilot status changed")
    gse314_residual = gse314["pilot_gate"]["primary_vs_selected_residual"]
    panels.append(
        _loss_panel(
            panel_id="gse314416_pilot_terminal",
            panel="GSE314416 immunomicrobiome pilot",
            accession="GSE314416",
            assay_pair="same-cell RNA and surface-protein binary states",
            phase="adaptive_pilot",
            role="development",
            unit="physical donor",
            units=len(gse314["pilot_donors"]),
            entities=81,
            method="hierarchical_exact_conditional_coupling",
            value=gse314_residual["primary_mean_loss"],
            decision=gse314["status"],
            artifact=gse314_path,
            notes="The pilot missed the frozen five-percent and favorable-donor criteria; no held result was formed.",
        )
    )
    for key, comparator in (
        ("primary_vs_selected_residual", "selected_classical_residual"),
        ("primary_vs_destroyed_link", "destroyed_link"),
    ):
        record = gse314["pilot_gate"][key]
        comparisons.append(
            _loss_comparison(
                comparison_id=f"gse314416_pilot_terminal__{key}",
                panel_id="gse314416_pilot_terminal",
                phase="adaptive_pilot",
                role="development",
                comparison_role="development_gate",
                primary_method="hierarchical_exact_conditional_coupling",
                comparator_method=comparator,
                primary_value=record["primary_mean_loss"],
                comparator_value=record["comparator_mean_loss"],
                ci=record["paired_bootstrap_95_interval"],
                relative_improvement=record["relative_deviance_reduction"],
                artifact=gse314_path,
                favorable_units=record["favorable_donors"],
                total_units=len(gse314["pilot_donors"]),
                decision="PASS" if record["passes"] else "REFUSE",
                notes="Development-stage pilot comparison; it is not held confirmation.",
            )
        )
    secondary = gse314["secondary_broad_panel"]["comparison"]
    comparisons.append(
        _loss_comparison(
            comparison_id="gse314416_pilot_terminal__secondary_exact_vs_residual",
            panel_id="gse314416_pilot_terminal",
            phase="adaptive_pilot_secondary",
            role="development",
            comparison_role="secondary_broad_panel",
            primary_method="common_effect_exact_cmle",
            comparator_method="selected_classical_residual",
            primary_value=secondary["primary_mean_loss"],
            comparator_value=secondary["comparator_mean_loss"],
            ci=secondary["paired_bootstrap_95_interval"],
            relative_improvement=secondary["relative_deviance_reduction"],
            artifact=gse314_path,
            favorable_units=secondary["favorable_donors"],
            total_units=len(gse314["pilot_donors"]),
            decision="PASS" if secondary["passes"] else "REFUSE",
            notes="Secondary development comparison on the broad panel; no held inference.",
        )
    )

    adaptive_path = "results/development/exact_logodds_head_to_head_v1.json"
    adaptive = _load_json(adaptive_path)
    if adaptive["status"] != "RETROSPECTIVE_ADAPTIVE_DEVELOPMENT_ONLY":
        raise ValueError("adaptive binary audit status changed")
    adaptive_specs = (
        (
            "scmmib_bmmc_adaptive_development",
            "SCMMIB BMMC adaptive fit-to-bridge analysis",
            "openproblems-bio BMMC",
            "SCMMIB BMMC fit-to-bridge",
        ),
        (
            "combat_oxford_adaptive_development",
            "COMBAT Oxford adaptive calibration-to-pilot analysis",
            "COMBAT CITE-seq",
            "COMBAT Oxford calibration-to-pilot",
        ),
    )
    adaptive_by_name = {
        record["panel"]: record for record in adaptive["panels"].values()
    }
    for panel_id, panel_name, accession, source_name in adaptive_specs:
        record = adaptive_by_name[source_name]
        primary = record["selection"]["primary"]["mean_loss"]
        panels.append(
            _loss_panel(
                panel_id=panel_id,
                panel=panel_name,
                accession=accession,
                assay_pair="same-cell RNA and surface-protein binary states",
                phase="adaptive_development",
                role="retrospective_adaptive_development",
                unit="nonheld physical unit",
                units=record["unit_count"],
                entities=record["entity_count"],
                method="hierarchical_exact_conditional_coupling",
                value=primary,
                decision=record["status"],
                artifact=adaptive_path,
                notes="The same nonheld units selected the configuration and supplied this descriptive summary; graph zero was selected.",
            )
        )
        for key, comparator in (
            ("primary_vs_best_residual", "selected_classical_residual"),
            ("primary_vs_destroyed_link", "destroyed_link"),
        ):
            comparison = record["comparisons"][key]
            comparisons.append(
                _loss_comparison(
                    comparison_id=f"{panel_id}__{key}",
                    panel_id=panel_id,
                    phase="adaptive_development",
                    role="retrospective_adaptive_development",
                    comparison_role="descriptive_development",
                    primary_method="hierarchical_exact_conditional_coupling",
                    comparator_method=comparator,
                    primary_value=comparison["primary_mean_deviance"],
                    comparator_value=comparison["comparator_mean_deviance"],
                    ci=comparison["paired_unit_bootstrap_95_ci"],
                    relative_improvement=comparison["relative_reduction"],
                    artifact=adaptive_path,
                    favorable_units=comparison["favorable_units"],
                    total_units=comparison["total_units"],
                    decision="DESCRIPTIVE",
                    notes="Retrospective adaptive development result; not held or confirmatory inference.",
                )
            )

    combat_path = "results/development/combat_citeseq_pilot_terminal_refusal.json"
    combat = _load_json(combat_path)
    if combat["status"] != "TERMINAL_PILOT_CANDIDATE_AVAILABILITY_REFUSAL":
        raise ValueError("COMBAT terminal status changed")
    result_path, result_sha = _artifact_fields(combat_path)
    panels.append(
        {
            "panel_id": "combat_terminal_pilot",
            "panel": "COMBAT CITE-seq frozen pilot",
            "accession": "COMBAT CITE-seq",
            "assay_pair": "same-cell RNA and surface-protein binary states",
            "analysis_phase": "adaptive_pilot",
            "inference_role": "procedural_refusal",
            "evaluation_unit": "physical sample",
            "unit_count": 24,
            "entity_count": 81,
            "primary_method": "Haldane_Paule_Mandel_product_graph_field",
            "primary_metric": "",
            "metric_direction": "",
            "primary_value": "",
            "ci_95_low": "",
            "ci_95_high": "",
            "decision": combat["status"],
            "outcome_scored": "NO",
            "result_artifact": result_path,
            "result_sha256": result_sha,
            "notes": "Comparator-family availability failed in the pilot; all 61 held samples remained unopened.",
        }
    )
    return panels, comparisons


def _terminal_panels() -> list[dict[str, Any]]:
    panels: list[dict[str, Any]] = []
    for panel_id, panel, accession, assay_pair, artifact in SOURCE_TERMINALS:
        result = _load_json(artifact)
        status = result.get("status")
        if not isinstance(status, str) or "TERMINAL" not in status:
            raise ValueError(f"source terminal status changed: {artifact}")
        result_path, result_sha = _artifact_fields(artifact)
        panels.append(
            {
                "panel_id": panel_id,
                "panel": panel,
                "accession": accession,
                "assay_pair": assay_pair,
                "analysis_phase": "source_or_development_gate",
                "inference_role": "procedural_refusal",
                "evaluation_unit": "",
                "unit_count": "",
                "entity_count": "",
                "primary_method": "hierarchical_exact_conditional_coupling",
                "primary_metric": "",
                "metric_direction": "",
                "primary_value": "",
                "ci_95_low": "",
                "ci_95_high": "",
                "decision": status,
                "outcome_scored": "NO",
                "result_artifact": result_path,
                "result_sha256": result_sha,
                "notes": "Terminal source/development-stage record; no held performance value is assigned in the aggregate ledger.",
            }
        )

    gse342939_path = "results/development/gse342939_ra_bcell_source_v1.json"
    gse342939 = _load_json(gse342939_path)
    access = gse342939.get("access_audit", {})
    files = gse342939.get("source_files", [])
    if (
        gse342939.get("status") != "TERMINAL_SOURCE_EXECUTION_REFUSAL"
        or gse342939.get("reason_code") != "NO_COMPLETE_PRIMARY_CANDIDATE"
        or gse342939.get("held_numeric_access_authorized") is not False
        or gse342939.get("rerun_permitted") is not False
        or access.get("held_numeric_urls_requested") != 0
        or len(files) != 28
        or not all(
            record.get("completed") is True
            and record.get("deleted") is True
            and record.get("reduction_completed") is True
            for record in files
        )
    ):
        raise ValueError("GSE342939 terminal source record changed")
    result_path, result_sha = _artifact_fields(gse342939_path)
    panels.append(
        {
            "panel_id": "gse342939_ra_bcell_source_terminal",
            "panel": "GSE342939 longitudinal RA B-cell source campaign",
            "accession": "GSE342939",
            "assay_pair": "paired longitudinal B-cell RNA-ADT coupling transfer",
            "analysis_phase": "source_gate",
            "inference_role": "procedural_refusal",
            "evaluation_unit": "physical source donor",
            "unit_count": 7,
            "entity_count": 2025,
            "primary_method": "longitudinal_graph_regularized_exact_conditional_coupling",
            "primary_metric": "",
            "metric_direction": "",
            "primary_value": "",
            "ci_95_low": "",
            "ci_95_high": "",
            "decision": gse342939["status"],
            "outcome_scored": "NO",
            "result_artifact": result_path,
            "result_sha256": result_sha,
            "notes": (
                "All 28 source matrices were reduced and deleted, but no primary "
                "configuration completed every source fold. The six held donors "
                "remained unopened."
            ),
        }
    )

    gse317605 = _load_json(GSE317605_RESULT_PATH)
    selection = gse317605.get("selection", {})
    losses = selection.get("losses", {})
    gate = selection.get("gate", {})
    poisson_gate = gate.get("comparisons", {}).get(
        "classical_time_conditioned_ridge_poisson", {}
    )
    access = gse317605.get("access_ledger", {})
    tag_commit = subprocess.check_output(
        ["git", "rev-list", "-n", "1", GSE317605_RESULT_TAG],
        cwd=ROOT,
        text=True,
    ).strip()
    if (
        _sha256(GSE317605_RESULT_PATH) != GSE317605_RESULT_SHA256
        or tag_commit != GSE317605_RESULT_COMMIT
        or gse317605.get("schema") != "gse317605-calibration-result/1.0"
        or gse317605.get("stage") != "calibration"
        or gse317605.get("status") != "CALIBRATION_FAIL"
        or gse317605.get("rerun_permitted") is not False
        or len(gse317605.get("patient_order", [])) != 7
        or len(gse317605.get("visit_patient_axis", [])) != 28
        or set(gse317605.get("visit_timepoint_axis", []))
        != {"T01", "T02", "T03", "T04"}
        or gse317605.get("pilot_matrix_requests") != 0
        or gse317605.get("held_matrix_requests") != 0
        or access.get("expected_files") != 210
        or access.get("finished_files") != 210
        or access.get("failed_files") != 0
        or access.get("deleted_files") != 210
        or access.get("exact_manifest_reconciliation_passes") is not True
        or selection.get("selected_primary", {}).get("hypergraph_penalty") != 0.0
        or selection.get("selected_primary") != selection.get("selected_graph_zero")
        or losses.get("primary", {}).get("mean") != 0.006529760758433352
        or losses.get("graph_zero_retuned_exact_coupling", {}).get("mean")
        != losses.get("primary", {}).get("mean")
        or losses.get("classical_time_conditioned_ridge_poisson", {}).get("mean")
        != 0.006608620233208813
        or poisson_gate.get("relative_reduction") != 0.011932819861426913
        or poisson_gate.get("favorable_patients") != 6
        or poisson_gate.get("passes") is not False
        or gate.get("passes") is not False
    ):
        raise ValueError("GSE317605 terminal calibration record changed")
    result_path, result_sha = _artifact_fields(GSE317605_RESULT_PATH)
    panels.append(
        {
            "panel_id": "gse317605_longitudinal_calibration_terminal",
            "panel": "GSE317605 longitudinal PBMC CITE-seq calibration campaign",
            "accession": "GSE317605",
            "assay_pair": "longitudinal same-cell RNA and surface-protein binary states",
            "analysis_phase": "calibration_gate",
            "inference_role": "procedural_refusal",
            "evaluation_unit": "calibration patient",
            "unit_count": 7,
            "entity_count": 256,
            "primary_method": "longitudinal_hypergraph_exact_conditional_coupling",
            "primary_metric": "",
            "metric_direction": "",
            "primary_value": "",
            "ci_95_low": "",
            "ci_95_high": "",
            "decision": gse317605["status"],
            "outcome_scored": "NO",
            "result_artifact": result_path,
            "result_sha256": result_sha,
            "notes": (
                "Unscored calibration refusal. Primary loss was 0.0065297608 "
                "versus 0.0066086202 for tuned time-conditioned ridge-Poisson "
                "(1.193% reduction; 6/7 patients), below the frozen 5% gate. "
                "The selected hypergraph penalty was zero and the primary "
                "equaled the retuned graph-zero fit. Pilot and held matrices "
                "were never requested."
            ),
        }
    )

    kotliarov_v2_path = "results/development/kotliarov_pbmc_binary_v2_source_v2.json"
    kotliarov_v2 = _load_json(kotliarov_v2_path)
    if kotliarov_v2["status"] != "TERMINAL_SOURCE_EXECUTION_REFUSAL":
        raise ValueError("Kotliarov binary-v2 source status changed")
    if kotliarov_v2["passes_source_promotion_gate"] is not False:
        raise ValueError("Kotliarov binary-v2 source unexpectedly passed")
    if kotliarov_v2["held_adt_access_authorized"] is not False:
        raise ValueError("Kotliarov binary-v2 source authorized held ADT access")
    if (
        kotliarov_v2["reason"]
        != "no frozen configuration completed every source-held fold"
    ):
        raise ValueError("Kotliarov binary-v2 terminal reason changed")
    if "comparisons" in kotliarov_v2:
        raise ValueError("Kotliarov binary-v2 source unexpectedly records comparisons")
    attempt_path = "data/confirmation/kotliarov_pbmc_binary_v2/source_attempt_v2.json"
    if _sha256(attempt_path) != kotliarov_v2["source_attempt_sha256"]:
        raise ValueError("Kotliarov binary-v2 source-attempt hash changed")
    authorization = kotliarov_v2["authorization"]
    if _sha256(authorization["authorization_path"]) != authorization[
        "authorization_sha256"
    ]:
        raise ValueError("Kotliarov binary-v2 authorization hash changed")
    result_path, result_sha = _artifact_fields(kotliarov_v2_path)
    panels.append(
        {
            "panel_id": "kotliarov_pbmc_binary_v2_source_terminal",
            "panel": "Kotliarov PBMC binary-v2 source campaign",
            "accession": "KotliarovPBMCData",
            "assay_pair": "same-cell RNA and surface-protein binary states",
            "analysis_phase": "source_development_gate",
            "inference_role": "procedural_refusal",
            "evaluation_unit": "physical donor",
            "unit_count": 10,
            "entity_count": 81,
            "primary_method": "hierarchical_exact_conditional_coupling",
            "primary_metric": "",
            "metric_direction": "",
            "primary_value": "",
            "ci_95_low": "",
            "ci_95_high": "",
            "decision": kotliarov_v2["status"],
            "outcome_scored": "NO",
            "result_artifact": result_path,
            "result_sha256": result_sha,
            "notes": (
                "No frozen configuration completed every source-held fold, so no "
                "comparison decision was produced. Held ADT access remained "
                "unauthorized; this is a procedural source-execution refusal, not "
                "a scientific negative."
            ),
        }
    )

    gse179221_path = "results/development/gse179221_bmmc_source_v1.json"
    gse179221 = _load_json(gse179221_path)
    gse179221_attempt_path = (
        "data/confirmation/gse179221_bmmc/source_attempt_v1.json"
    )
    gse179221_consumption_path = (
        "data/confirmation/gse179221_bmmc/source_consumption_v1.json"
    )
    gse179221_candidate = _load_json(
        "data/confirmation/gse179221_bmmc/candidate_designation_v1.json"
    )
    gse179221_consumption = _load_json(gse179221_consumption_path)
    access = gse179221.get("access_audit", {})
    source_files = access.get("source_files", [])
    first_candidate = gse179221_candidate["source_files"][0]
    first_url = gse179221_candidate["metadata_bindings"][
        "per_sample_url_template"
    ].format(gsm=first_candidate["gsm"], filename=first_candidate["filename"])
    expected_datasets = {
        "matrix/barcodes",
        "matrix/features/feature_type",
        "matrix/features/name",
    }
    if (
        gse179221.get("status") != "TERMINAL_SOURCE_EXECUTION_REFUSAL"
        or gse179221.get("reason_code") != "COGNATE_AXIS_NOT_EXACTLY_UNIQUE"
        or gse179221.get("passes_source_promotion_gate") is not False
        or gse179221.get("held_h5_access_authorized") is not False
        or gse179221.get("held_h5_access_eligible_after_public_source_pass")
        is not False
        or "comparisons" in gse179221
        or "models" in gse179221
        or gse179221.get("source_attempt_sha256")
        != _sha256(gse179221_attempt_path)
        or gse179221.get("source_consumption_sha256")
        != _sha256(gse179221_consumption_path)
        or gse179221.get("source_consumption") != gse179221_consumption
        or access.get("source_h5_get_count") != 1
        or access.get("source_h5_deleted_count") != 1
        or access.get("held_h5_get_count") != 0
        or access.get("all_donor_tar_get_count") != 0
        or access.get("maximum_simultaneous_h5_files") != 1
        or access.get("requested_urls") != [first_url]
        or len(source_files) != 1
        or source_files[0].get("gsm") != first_candidate["gsm"]
        or source_files[0].get("donor") != first_candidate["donor"]
        or source_files[0].get("filename") != first_candidate["filename"]
        or source_files[0].get("expected_bytes") != first_candidate["bytes"]
        or source_files[0].get("download_status") != "COMPLETE"
        or source_files[0].get("reduction_status") != "REFUSED"
        or source_files[0].get("reduction_reason_code")
        != "COGNATE_AXIS_NOT_EXACTLY_UNIQUE"
        or source_files[0].get("deleted_after_reduction") is not True
        or set(source_files[0].get("decoded_h5_datasets", []))
        != expected_datasets
    ):
        raise ValueError("GSE179221 terminal source boundary changed")
    present_downstream = sorted(
        relative
        for relative in ABSENT_GSE179221_DOWNSTREAM_PATHS
        if (ROOT / relative).exists()
    )
    if present_downstream:
        raise ValueError(
            f"GSE179221 terminal source run has downstream artifacts: {present_downstream}"
        )
    result_path, result_sha = _artifact_fields(gse179221_path)
    panels.append(
        {
            "panel_id": "gse179221_bmmc_source_terminal",
            "panel": "GSE179221 BMMC held-donor campaign",
            "accession": "GSE179221",
            "assay_pair": "same-cell RNA and surface-protein binary states",
            "analysis_phase": "source_feature_axis_preflight",
            "inference_role": "procedural_refusal",
            "evaluation_unit": "physical donor",
            "unit_count": 8,
            "entity_count": 81,
            "primary_method": "hierarchical_exact_conditional_coupling",
            "primary_metric": "",
            "metric_direction": "",
            "primary_value": "",
            "ci_95_low": "",
            "ci_95_high": "",
            "decision": gse179221["status"],
            "outcome_scored": "NO",
            "result_artifact": result_path,
            "result_sha256": result_sha,
            "notes": (
                "The exact cognate-axis gate refused on the first of eight source "
                "donors after barcode and feature-axis access. No count dataset, "
                "model, comparison, held file, prediction, or score was opened or "
                "formed."
            ),
        }
    )

    gse214546_path = "results/development/gse214546_teaseq_source_v1.json"
    gse214546 = _load_json(gse214546_path)
    gse214546_attempt_path = (
        "data/confirmation/gse214546_teaseq/source_attempt_v1.json"
    )
    gse214546_attempt = _load_json(gse214546_attempt_path)
    bindings = gse214546.get("bindings", {})
    bound_artifacts = {
        "candidate_sha256": (
            "data/confirmation/gse214546_teaseq/candidate_designation_v1.json"
        ),
        "amendment_sha256": (
            "data/confirmation/gse214546_teaseq/"
            "pre_access_schema_amendment_v1.json"
        ),
        "implementation_clarification_sha256": (
            "data/confirmation/gse214546_teaseq/"
            "pre_access_implementation_clarification_v1.json"
        ),
        "cv_availability_clarification_sha256": (
            "data/confirmation/gse214546_teaseq/"
            "pre_access_cv_availability_clarification_v1.json"
        ),
        "normalization_correction_sha256": (
            "data/confirmation/gse214546_teaseq/"
            "pre_access_normalization_correction_v1.json"
        ),
        "sparse_access_clarification_sha256": (
            "data/confirmation/gse214546_teaseq/"
            "pre_access_sparse_access_clarification_v1.json"
        ),
        "crash_semantics_clarification_sha256": (
            "data/confirmation/gse214546_teaseq/"
            "pre_access_crash_semantics_clarification_v1.json"
        ),
        "protocol_sha256": (
            "docs/GSE214546_TEASEQ_HELD_DONOR_CONFIRMATION_PROTOCOL_2026-08-30.md"
        ),
    }
    source_gsms = [
        "GSM6611363",
        "GSM6611365",
        "GSM6611366",
        "GSM6611367",
        "GSM6611371",
        "GSM6611372",
        "GSM6611373",
        "GSM6611377",
    ]
    held_gsms = [
        "GSM6611364",
        "GSM6611368",
        "GSM6611369",
        "GSM6611370",
        "GSM6611374",
        "GSM6611375",
        "GSM6611376",
        "GSM6611378",
    ]
    access = gse214546.get("access_audit", [])
    expected_downloads = [
        (
            "GSM6611363_B065-P1_PB00593-04_filtered_metadata.csv.gz",
            1_143_063,
            "9152caa317ed1d2eeb50dc6e36034de969f6a82ed24bd987eba31764d5d4eab4",
        ),
        (
            "GSM6611363_B065-P1_PB00593-04.h5",
            38_481_738,
            "fb7b1fddf5f21e8a7e0377911dc86b28c69c2505650de9061f64eb1871f9a9dd",
        ),
        (
            "GSM6611365_B076-P1_PB00368-04_filtered_metadata.csv.gz",
            2_769_550,
            "34990b13e213e70546659aea48075d420b6f39cd1dccddabd34ce1b547ebd157",
        ),
        (
            "GSM6611365_B076-P1_PB00368-04.h5",
            117_389_863,
            "fa95fdf563480f49e09477b17273803cad098fe72ace7b4d2f49731cb1f0f0e6",
        ),
    ]
    if len(access) != 4:
        raise ValueError("GSE214546 source download count changed")
    observed_downloads = [
        (
            record.get("filename"),
            record.get("observed_bytes"),
            record.get("sha256"),
        )
        for record in access
    ]
    first_metadata, first_h5, second_metadata, second_h5 = access
    first_event_counts = {
        dataset: sum(
            event.get("dataset") == dataset
            for event in first_h5.get("dataset_access_events", [])
        )
        for dataset in first_h5.get("datasets_read", [])
    }
    expected_first_event_counts = {
        "ADT/barcodes": 1,
        "ADT/data": 512,
        "ADT/features/id": 1,
        "ADT/indices": 512,
        "ADT/indptr": 512,
        "ADT/shape": 1,
        "matrix/barcodes": 1,
        "matrix/data": 511,
        "matrix/features/feature_type": 1,
        "matrix/features/name": 1,
        "matrix/indices": 512,
        "matrix/indptr": 512,
        "matrix/shape": 1,
    }
    expected_second_axes = {
        "matrix/barcodes",
        "matrix/features/feature_type",
        "matrix/features/name",
        "matrix/shape",
    }
    expected_public_tags = {
        "amendment_commit": "30057e6a4e4da37c3755b911f33c53d84918fdb6",
        "amendment_tag": "gse214546-teaseq-v1-pre-access-amendment",
        "candidate_commit": "53ae1fb470056b1ce17e78f977f52db328c89038",
        "candidate_tag": "gse214546-teaseq-v1-candidate",
        "crash_semantics_clarification_commit": (
            "39579621c11c4b7e4bb667343f6407309d237862"
        ),
        "crash_semantics_clarification_tag": (
            "gse214546-teaseq-v1-crash-semantics-clarification"
        ),
        "cv_availability_commit": "394a9bd601a0a02fb1b79540f8822c291be122c3",
        "cv_availability_tag": "gse214546-teaseq-v1-cv-availability",
        "implementation_clarification_commit": (
            "8b1b4a05931110d50c2946ec49386e01bd7daeef"
        ),
        "implementation_clarification_tag": (
            "gse214546-teaseq-v1-implementation-clarification"
        ),
        "implementation_commit": "e2bd6c8fc63c86f5f9c5d7429a33332262d47e2f",
        "implementation_tag": "gse214546-teaseq-v1-implementation",
        "normalization_correction_commit": (
            "5ff7994913d254fbad88a26b59ab367bebd2534a"
        ),
        "normalization_correction_tag": (
            "gse214546-teaseq-v1-normalization-correction"
        ),
        "sparse_access_clarification_commit": (
            "266b5ba73477534341b4bc4acd2e1e6211213e00"
        ),
        "sparse_access_clarification_tag": (
            "gse214546-teaseq-v1-sparse-access-clarification"
        ),
    }
    if (
        gse214546.get("schema") != "gse214546-teaseq-source-result/1.0"
        or gse214546.get("status") != "TERMINAL_SOURCE_REFUSAL"
        or gse214546.get("stage") != "source"
        or gse214546.get("reason_code") != "FEWER_THAN_512_MATCHED_SINGLETS"
        or gse214546.get("details") != {}
        or gse214546.get("held_h5_requested") is not False
        or gse214546.get("rerun_permitted") is not False
        or bindings.get("source_attempt_sha256") != _sha256(gse214546_attempt_path)
        or bindings.get("source_attempt_bytes")
        != (ROOT / gse214546_attempt_path).stat().st_size
        or bindings.get("source_attempt_tag")
        != "gse214546-teaseq-v1-source-attempt"
        or bindings.get("source_attempt_commit")
        != "15b3a8cbb77395943ffef28fb01be39703ce1b95"
        or bindings.get("claim_token_sha256")
        != "7174930e43f0f57caad4aeef6403aeb63752b895d7b9595c816c3a1f8925e946"
        or bindings.get("public_tags") != expected_public_tags
        or any(bindings.get(key) != _sha256(path) for key, path in bound_artifacts.items())
        or gse214546_attempt.get("schema")
        != "gse214546-teaseq-source-attempt/1.0"
        or gse214546_attempt.get("status")
        != "CLAIMED_BEFORE_FIRST_SOURCE_FILE_GET"
        or gse214546_attempt.get("held_numeric_access_authorized") is not False
        or gse214546_attempt.get("rerun_permitted") is not False
        or gse214546_attempt.get("source_gsms") != source_gsms
        or gse214546_attempt.get("held_gsms") != held_gsms
        or gse214546_attempt.get("claim_token_sha256")
        != bindings.get("claim_token_sha256")
        or any(
            bindings.get(key) != value
            for key, value in gse214546_attempt.get("bindings", {}).items()
        )
        or observed_downloads != expected_downloads
        or any(
            record.get("expected_bytes") != expected_bytes
            or record.get("request_started") is not True
            or record.get("completed") is not True
            or record.get("deleted") is not True
            for record, (_, expected_bytes, _) in zip(access, expected_downloads)
        )
        or sum(record[1] for record in expected_downloads) != 159_784_214
        or first_metadata.get("decode")
        != {
            "barcode_column": "barcodes",
            "literal_true_singlets": 10_295,
            "rows": 11_191,
            "singlet_column": "singlet",
            "singlet_value": "TRUE",
            "unique_barcodes": 11_191,
        }
        or first_h5.get("selected_cells") != 512
        or first_h5.get("authorized_marker_count") != 53
        or first_h5.get("selected_cell_axis_sha256")
        != "9565b4c84360a0fa527833fa17beb93c842a14e3ee6b601ad76242340ffd7fa2"
        or first_h5.get("selected_cell_indices_sha256")
        != "896b0846b11e358245caae647203681f7419f79fdc7adb89f650e71054332e3f"
        or first_h5.get("authorized_marker_axis_sha256")
        != "2794bd7eb9a548ff194cdaa6f20ddf05a22fe9254e2017ec1bb25f2e62a51943"
        or first_h5.get("h5_reduction_completed") is not True
        or first_h5.get("gex_access", {}).get("indices_decoded") != 781_366
        or first_h5.get("gex_access", {}).get("selected_data_values_decoded")
        != 3_323
        or first_h5.get("adt_access", {}).get("indices_decoded") != 20_226
        or first_h5.get("adt_access", {}).get("selected_data_values_decoded")
        != 19_587
        or first_event_counts != expected_first_event_counts
        or len(first_h5.get("dataset_access_events", [])) != 3_078
        or not all(
            event.get("started") is True and event.get("completed") is True
            for event in first_h5.get("dataset_access_events", [])
        )
        or second_metadata.get("decode")
        != {
            "barcode_column": "barcodes",
            "literal_true_singlets": 25_364,
            "rows": 28_337,
            "singlet_column": "singlet",
            "singlet_value": "TRUE",
            "unique_barcodes": 28_337,
        }
        or second_h5.get("h5_reduction_completed") is not False
        or set(second_h5.get("datasets_opened", [])) != expected_second_axes
        or set(second_h5.get("datasets_read", [])) != expected_second_axes
        or len(second_h5.get("dataset_access_events", [])) != 4
        or not all(
            event.get("started") is True
            and event.get("completed") is True
            and event.get("selection") == {"kind": "all"}
            for event in second_h5.get("dataset_access_events", [])
        )
        or "selected_cells" in second_h5
        or any(
            held_gsm in record.get("filename", "")
            for held_gsm in held_gsms
            for record in access
        )
    ):
        raise ValueError("GSE214546 terminal source boundary changed")
    result_path, result_sha = _artifact_fields(gse214546_path)
    panels.append(
        {
            "panel_id": "gse214546_teaseq_source_terminal",
            "panel": "GSE214546 TEA-seq held-donor campaign",
            "accession": "GSE214546",
            "assay_pair": "same-cell RNA and surface-protein binary states",
            "analysis_phase": "source_support_gate",
            "inference_role": "procedural_refusal",
            "evaluation_unit": "physical source donor",
            "unit_count": 2,
            "entity_count": 53,
            "primary_method": "age_conditioned_exact_conditional_coupling",
            "primary_metric": "",
            "metric_direction": "",
            "primary_value": "",
            "ci_95_low": "",
            "ci_95_high": "",
            "decision": gse214546["status"],
            "outcome_scored": "NO",
            "result_artifact": result_path,
            "result_sha256": result_sha,
            "notes": (
                "The first source donor completed its frozen 512-cell, 53-marker "
                "reduction. The second donor had fewer than 512 matched singlets; "
                "the exact overlap count was not serialized. The remaining six "
                "source H5s and all eight held H5s remained unopened."
            ),
        }
    )

    terminal_path = (
        "data/confirmation/stephenson_unused_cambridge/"
        "prediction_terminal_record_v1_1.json"
    )
    terminal = _load_json(terminal_path)
    if terminal["status"] != "TERMINAL_INFRASTRUCTURE_UNEVALUABLE":
        raise ValueError("unused-Cambridge terminal status changed")
    if terminal["observations"]["prediction_output_exists"]:
        raise ValueError("unused-Cambridge terminal record unexpectedly has predictions")
    if terminal["access_boundary"]["score_phase_started"]:
        raise ValueError("unused-Cambridge terminal record unexpectedly started scoring")
    result_path, result_sha = _artifact_fields(terminal_path)
    panels.append(
        {
            "panel_id": "stephenson_unused_cambridge_terminal",
            "panel": "Stephenson unused-Cambridge donor confirmation",
            "accession": "E-MTAB-10026",
            "assay_pair": "same-cell RNA and surface-protein binary states",
            "analysis_phase": "recovery_amended_prediction_terminal",
            "inference_role": "infrastructure_unevaluable",
            "evaluation_unit": "physical donor",
            "unit_count": "",
            "entity_count": 81,
            "primary_method": "hierarchical_exact_conditional_coupling",
            "primary_metric": "",
            "metric_direction": "",
            "primary_value": "",
            "ci_95_low": "",
            "ci_95_high": "",
            "decision": terminal["status"],
            "outcome_scored": "NO",
            "result_artifact": result_path,
            "result_sha256": result_sha,
            "notes": "The single authorized replacement produced no prediction or score. The terminal record classifies the attempt as infrastructure-unevaluable, not a scientific pass or failure.",
        }
    )
    return panels


def _sequence_row(
    panel_id: str,
    ordinal: int,
    stage: str,
    status: str,
    outcome_access: str,
    public_before_outcome: str,
    artifact: str,
    public_commit_or_tag: str = "",
    notes: str = "",
) -> dict[str, Any]:
    artifact_sha = "" if not artifact else _sha256(artifact)
    return {
        "sequence_id": f"{panel_id}__{ordinal:02d}_{stage}",
        "panel_id": panel_id,
        "stage_ordinal": ordinal,
        "stage": stage,
        "status": status,
        "outcome_access": outcome_access,
        "public_before_outcome": public_before_outcome,
        "artifact": artifact,
        "artifact_sha256": artifact_sha,
        "public_commit_or_tag": public_commit_or_tag,
        "notes": notes or "See artifact.",
    }


def _sequence(panels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    explicit_ids = {
        "scmmib_bmmc_terminal",
        "gse279451_terminal",
        "gse299043_terminal",
        "stephenson_newcastle_confirmation",
        "gse239452_held_post_access_correction",
        "gse314416_pilot_terminal",
        "scmmib_bmmc_adaptive_development",
        "combat_oxford_adaptive_development",
        "combat_terminal_pilot",
        "kotliarov_pbmc_binary_v2_source_terminal",
        "gse179221_bmmc_source_terminal",
        "gse214546_teaseq_source_terminal",
        "gse342939_ra_bcell_source_terminal",
        "gse317605_longitudinal_calibration_terminal",
        "stephenson_unused_cambridge_terminal",
    }
    rows = [
        _sequence_row(
            panel["panel_id"],
            1,
            "terminal_record",
            panel["decision"],
            "SCORED" if panel["outcome_scored"] == "YES" else "NOT_SCORED",
            "UNKNOWN",
            panel["result_artifact"],
            notes="Single-row historical sequence; inference role is defined in the panels ledger.",
        )
        for panel in panels
        if panel["panel_id"] not in explicit_ids
    ]

    rows.extend(
        (
            _sequence_row(
                "scmmib_bmmc_terminal",
                1,
                "protocol",
                "OUTCOME_ACCESS_DISABLED",
                "HELD_DISABLED",
                "YES",
                "docs/SCMMIB_BMMC_HELD_DONOR_CONFIRMATION_PROTOCOL_2026-08-28.md",
                "bmmc-held-donor-v1-protocol",
            ),
            _sequence_row(
                "scmmib_bmmc_terminal",
                2,
                "development_attempt_1",
                "TERMINAL_ATTEMPT_REFUSAL_NUMERICAL",
                "NONHELD_ONLY",
                "NOT_APPLICABLE",
                "results/development/scmmib_bmmc_exact_development_attempt_1_refusal.json",
            ),
            _sequence_row(
                "scmmib_bmmc_terminal",
                3,
                "development_attempt_2",
                "ABORTED_BYTE_FREEZE_TRANSITION",
                "NONHELD_ONLY",
                "NOT_APPLICABLE",
                "results/development/scmmib_bmmc_exact_development_attempt_2_aborted.json",
            ),
            _sequence_row(
                "scmmib_bmmc_terminal",
                4,
                "development_attempt_3",
                "TERMINAL_NUMERICAL_EQUIVALENCE_RETRY_REFUSAL",
                "NONHELD_ONLY",
                "NOT_APPLICABLE",
                "results/development/scmmib_bmmc_exact_development_attempt_3_terminal_refusal.json",
                notes="No held count slice, prediction, or score was formed.",
            ),
            _sequence_row(
                "gse279451_terminal",
                1,
                "protocol",
                "PUBLIC_FREEZE_VERIFIED",
                "HELD_DISABLED",
                "YES",
                "docs/GSE279451_SEPSIS_HELD_DONOR_CONFIRMATION_PROTOCOL_2026-08-28.md",
                "gse279451-sepsis-v1-protocol",
            ),
            _sequence_row(
                "gse279451_terminal",
                2,
                "development_acquisition",
                "TERMINAL_DEVELOPMENT_ATTEMPT_STARTED",
                "DEVELOPMENT_ONLY",
                "NOT_APPLICABLE",
                "data/development/gse279451_sepsis/development_attempt_v1.json",
            ),
            _sequence_row(
                "gse279451_terminal",
                3,
                "development_evaluation",
                "TERMINAL_DEVELOPMENT_EVALUATION_STARTED",
                "DEVELOPMENT_ONLY",
                "NOT_APPLICABLE",
                "data/development/gse279451_sepsis/evaluation_attempt_v1.json",
            ),
            _sequence_row(
                "gse279451_terminal",
                4,
                "terminal_record",
                "TERMINAL_DEVELOPMENT_EVALUATION_REFUSAL",
                "HELD_UNOPENED",
                "NOT_APPLICABLE",
                "results/development/gse279451_sepsis_evaluation_refusal.json",
                "gse279451-sepsis-v1-terminal-refusal",
            ),
            _sequence_row(
                "gse299043_terminal",
                1,
                "protocol",
                "PUBLIC_FREEZE_VERIFIED",
                "HELD_DISABLED",
                "YES",
                "docs/GSE299043_MLN_HELD_SITE_CONFIRMATION_PROTOCOL_2026-08-28.md",
                "gse299043-mln-v1-protocol",
            ),
            _sequence_row(
                "gse299043_terminal",
                2,
                "development_acquisition",
                "TERMINAL_DEVELOPMENT_ATTEMPT_STARTED",
                "DEVELOPMENT_ONLY",
                "NOT_APPLICABLE",
                "data/development/gse299043_mln/development_attempt_v1.json",
            ),
            _sequence_row(
                "gse299043_terminal",
                3,
                "development_refusal",
                "TERMINAL_DEVELOPMENT_ACQUISITION_REFUSAL",
                "HELD_UNOPENED",
                "NOT_APPLICABLE",
                "results/development/gse299043_mln_development_acquisition_refusal.json",
            ),
            _sequence_row(
                "gse299043_terminal",
                4,
                "terminal_audit",
                "TERMINAL_DEVELOPMENT_ACQUISITION_REFUSAL",
                "HELD_UNOPENED",
                "NOT_APPLICABLE",
                "results/development/gse299043_mln_terminal_acquisition_audit.json",
                "gse299043-mln-v1-terminal-refusal",
            ),
            _sequence_row(
                "stephenson_newcastle_confirmation",
                1,
                "protocol",
                "PUBLIC_FREEZE_VERIFIED",
                "DISABLED",
                "YES",
                "docs/STEPHENSON_CITESEQ_HELD_SITE_CONFIRMATION_PROTOCOL_2026-08-28.md",
                notes="Outcome-disabled public analysis plan.",
            ),
            _sequence_row(
                "stephenson_newcastle_confirmation",
                2,
                "prediction",
                "FROZEN_HELD_PREDICTIONS",
                "RNA_MARGINS_ONLY",
                "YES",
                "results/stephenson_citeseq_predictions.json",
                notes="Predictions were public before paired held-site truth scoring.",
            ),
            _sequence_row(
                "stephenson_newcastle_confirmation",
                3,
                "score_authorization",
                "OUTCOME_ACCESS_AUTHORIZED",
                "PAIRED_TRUTH_AUTHORIZED",
                "YES",
                "data/confirmation/stephenson_citeseq/score_authorization_v1.json",
            ),
            _sequence_row(
                "stephenson_newcastle_confirmation",
                4,
                "score",
                "CONFIRMATION_PASS",
                "PAIRED_TRUTH_SCORED",
                "NOT_APPLICABLE",
                "results/stephenson_citeseq_confirmation.json",
            ),
            _sequence_row(
                "gse239452_held_post_access_correction",
                1,
                "protocol",
                "PLAN_RETAINED",
                "DISABLED_AT_DESIGNATION",
                "UNKNOWN",
                "docs/GSE239452_PREGNANCY_CITESEQ_CONFIRMATION_PROTOCOL_2026-08-28.md",
            ),
            _sequence_row(
                "gse239452_held_post_access_correction",
                2,
                "prediction",
                "PREDICTIONS_FROZEN",
                "HELD_GEX_ACCESSED",
                "NO",
                "results/gse239452_citeseq_predictions.json",
                notes="Final inference is labeled post-access correction.",
            ),
            _sequence_row(
                "gse239452_held_post_access_correction",
                3,
                "score",
                "HELD_PASS",
                "PAIRED_TRUTH_SCORED",
                "NO",
                "results/gse239452_citeseq_confirmation.json",
                notes="Post-access corrected held-cohort analysis; not prospective confirmation.",
            ),
            _sequence_row(
                "gse239452_held_post_access_correction",
                4,
                "post_access_numerical_correction",
                "POST_ACCESS_CORRECTION_COMPLETE",
                "PAIRED_TRUTH_REUSED_POST_ACCESS",
                "NO",
                "results/gse239452_citeseq_post_access_correction.json",
                notes=(
                    "The residual inversion defect was corrected after outcome "
                    "access; the original sealed predictions and score remain "
                    "byte-identical chronology records."
                ),
            ),
            _sequence_row(
                "gse239452_held_post_access_correction",
                5,
                "standard_fixed_interaction_poisson_audit",
                "POST_HOC_NONCONFIRMATORY_HEAD_TO_HEAD",
                "PAIRED_TRUTH_REPRODUCED_POST_HOC",
                "NO",
                (
                    "results/development/"
                    "gse239452_standard_poisson_interaction_posthoc.json"
                ),
                "gse239452-standard-poisson-v1-result",
                notes=(
                    "The standard fixed-interaction Poisson comparator was defined "
                    "after held outcomes had been accessed; it is descriptive only."
                ),
            ),
            _sequence_row(
                "gse314416_pilot_terminal",
                1,
                "protocol",
                "FROZEN",
                "HELD_DISABLED",
                "YES",
                "docs/GSE314416_IMMUNOMICROBIOME_HELD_POOL_CONFIRMATION_PROTOCOL_2026-08-28.md",
                "gse314416-citeseq-v1.2-protocol",
            ),
            _sequence_row(
                "gse314416_pilot_terminal",
                2,
                "pilot",
                "TERMINAL_PILOT_REFUSAL",
                "PILOT_ONLY",
                "NOT_APPLICABLE",
                "results/development/gse314416_citeseq_development.json",
                notes="No held result was formed.",
            ),
            _sequence_row(
                "scmmib_bmmc_adaptive_development",
                1,
                "adaptive_development_audit",
                "RETROSPECTIVE_ADAPTIVE_DEVELOPMENT_ONLY",
                "NONHELD_ONLY",
                "NO",
                "results/development/exact_logodds_head_to_head_v1.json",
            ),
            _sequence_row(
                "combat_oxford_adaptive_development",
                1,
                "adaptive_development_audit",
                "RETROSPECTIVE_ADAPTIVE_DEVELOPMENT_ONLY",
                "NONHELD_ONLY",
                "NO",
                "results/development/exact_logodds_head_to_head_v1.json",
            ),
            _sequence_row(
                "combat_terminal_pilot",
                1,
                "protocol",
                "PUBLIC_FREEZE_VERIFIED",
                "HELD_DISABLED",
                "YES",
                "docs/COMBAT_CITESEQ_HELD_CONFIRMATION_PROTOCOL_2026-08-28.md",
                "combat-citeseq-v1-protocol",
            ),
            _sequence_row(
                "combat_terminal_pilot",
                2,
                "pilot",
                "TERMINAL_PILOT_CANDIDATE_AVAILABILITY_REFUSAL",
                "PILOT_ONLY",
                "NOT_APPLICABLE",
                "results/development/combat_citeseq_pilot_terminal_refusal.json",
                notes="All held margins and pairings remained unopened.",
            ),
            _sequence_row(
                "kotliarov_pbmc_binary_v2_source_terminal",
                1,
                "source_freeze",
                "SOURCE_ONLY_DEVELOPMENT_FROZEN_BEFORE_ADT_COUNT_ACCESS",
                "ADT_COUNT_DATASET_UNOPENED",
                "YES",
                "data/confirmation/kotliarov_pbmc_binary_v2/candidate_designation_v2.json",
                "kotliarov-pbmc-binary-v2-source-freeze",
            ),
            _sequence_row(
                "kotliarov_pbmc_binary_v2_source_terminal",
                2,
                "source_authorization",
                "SOURCE_PAIRED_DEVELOPMENT_ACCESS_AUTHORIZED",
                "DEVELOPMENT_PAIRED_VALUES_AUTHORIZED_HELD_ADT_DISABLED",
                "YES",
                "data/confirmation/kotliarov_pbmc_binary_v2/source_authorization_v2.json",
                "b94322b43c359d58cec931ca2973e7e40deb251a",
            ),
            _sequence_row(
                "kotliarov_pbmc_binary_v2_source_terminal",
                3,
                "source_attempt",
                "CLAIMED_ONE_SHOT_BEFORE_COUNT_DATASET_OPEN",
                "SOURCE_COUNT_ACCESS_BEGAN_AFTER_RECORD",
                "NO",
                "data/confirmation/kotliarov_pbmc_binary_v2/source_attempt_v2.json",
                "8bf42230fb9b406aeba0a510daf9d30336448261",
                notes=(
                    "The one-shot attempt record was written locally before source "
                    "count access and published with the terminal result."
                ),
            ),
            _sequence_row(
                "kotliarov_pbmc_binary_v2_source_terminal",
                4,
                "source_result",
                "TERMINAL_SOURCE_EXECUTION_REFUSAL",
                "DEVELOPMENT_ONLY_HELD_ADT_UNAUTHORIZED",
                "NOT_APPLICABLE",
                "results/development/kotliarov_pbmc_binary_v2_source_v2.json",
                "8bf42230fb9b406aeba0a510daf9d30336448261",
                notes=(
                    "No frozen configuration completed every source-held fold; no "
                    "comparisons or held run were produced."
                ),
            ),
            _sequence_row(
                "kotliarov_pbmc_binary_v2_source_terminal",
                5,
                "postrun_access_certificate",
                "DETERMINISTIC_CODE_PATH_AUDIT_HELD_ADT_UNREACHABLE",
                "HELD_ADT_VALUES_UNREACHED_BY_BOUND_CODE_PATH",
                "NOT_APPLICABLE",
                (
                    "data/confirmation/kotliarov_pbmc_binary_v2/"
                    "source_access_code_path_certificate_v2.json"
                ),
                "kotliarov-pbmc-binary-v2-source-terminal",
                notes=(
                    "Deterministic post-run code-path audit, explicitly not an "
                    "instrumented runtime observation; the one-shot attempt and "
                    "result remain unchanged."
                ),
            ),
            _sequence_row(
                "gse317605_longitudinal_calibration_terminal",
                1,
                "protocol",
                "FROZEN_BEFORE_ANY_COUNT_MATRIX_ACCESS",
                "METADATA_AND_NONGATING_AXES_ONLY",
                "YES",
                "data/confirmation/gse317605_longitudinal/protocol_v1.json",
                "gse317605-longitudinal-v1-candidate",
            ),
            _sequence_row(
                "gse317605_longitudinal_calibration_terminal",
                2,
                "candidate_designation",
                "DESIGNATED_BEFORE_ANY_COUNT_MATRIX_ACCESS",
                "METADATA_AND_NONGATING_AXES_ONLY",
                "YES",
                (
                    "data/confirmation/gse317605_longitudinal/"
                    "candidate_designation_v1.json"
                ),
                "gse317605-longitudinal-v1-candidate",
            ),
            _sequence_row(
                "gse317605_longitudinal_calibration_terminal",
                3,
                "sample_manifest",
                "FROZEN_FROM_PUBLIC_METADATA_AND_AXES_BEFORE_COUNT_MATRIX_ACCESS",
                "ALL_COUNT_MATRICES_UNOPENED",
                "YES",
                (
                    "data/confirmation/gse317605_longitudinal/"
                    "sample_manifest_v1.json"
                ),
                "gse317605-longitudinal-v1-candidate",
            ),
            _sequence_row(
                "gse317605_longitudinal_calibration_terminal",
                4,
                "implementation",
                "FROZEN_BEFORE_FIRST_COUNT_MATRIX_ACCESS",
                "ALL_COUNT_MATRICES_UNOPENED",
                "YES",
                (
                    "data/confirmation/gse317605_longitudinal/"
                    "pre_access_implementation_v1.json"
                ),
                "gse317605-longitudinal-v1-implementation",
            ),
            _sequence_row(
                "gse317605_longitudinal_calibration_terminal",
                5,
                "calibration_attempt",
                "CLAIMED_BEFORE_FIRST_STAGE_FILE_GET",
                "CALIBRATION_AUTHORIZED_PILOT_AND_HELD_DISABLED",
                "YES",
                (
                    "data/confirmation/gse317605_longitudinal/"
                    "calibration_attempt_v1.json"
                ),
                "gse317605-longitudinal-v1-calibration-attempt",
            ),
            _sequence_row(
                "gse317605_longitudinal_calibration_terminal",
                6,
                "calibration_consumption",
                "CALIBRATION_CAPABILITY_CONSUMED",
                "CALIBRATION_ONLY_PILOT_AND_HELD_DISABLED",
                "NO",
                (
                    "data/confirmation/gse317605_longitudinal/"
                    "calibration_consumption_v1.json"
                ),
                GSE317605_RESULT_TAG,
                notes=(
                    "The exclusive calibration capability was consumed before "
                    "the first file GET and published with the terminal result."
                ),
            ),
            _sequence_row(
                "gse317605_longitudinal_calibration_terminal",
                7,
                "calibration_access_journal",
                "CALIBRATION_ACCESS_RECONCILED",
                "210_CALIBRATION_FILES_DELETED_PILOT_AND_HELD_UNREQUESTED",
                "NOT_APPLICABLE",
                (
                    "data/confirmation/gse317605_longitudinal/"
                    "calibration_access_v1.jsonl"
                ),
                GSE317605_RESULT_TAG,
            ),
            _sequence_row(
                "gse317605_longitudinal_calibration_terminal",
                8,
                "calibration_result",
                "CALIBRATION_FAIL",
                "CALIBRATION_ONLY_PILOT_AND_HELD_UNREQUESTED",
                "NOT_APPLICABLE",
                GSE317605_RESULT_PATH,
                GSE317605_RESULT_TAG,
                notes=(
                    "The primary cleared the patient-count condition but its "
                    "1.193% ridge-Poisson loss reduction missed the frozen 5% "
                    "gate; the selected hypergraph penalty was zero."
                ),
            ),
            _sequence_row(
                "gse342939_ra_bcell_source_terminal",
                1,
                "protocol",
                "FROZEN_BEFORE_ANY_NUMERIC_MATRIX_ACCESS",
                "ALL_NUMERIC_MATRICES_UNOPENED",
                "YES",
                "docs/GSE342939_RA_BCELL_CITESEQ_HELD_DONOR_PROTOCOL_2026-08-29.md",
                "gse342939-ra-bcell-v1-candidate",
            ),
            _sequence_row(
                "gse342939_ra_bcell_source_terminal",
                2,
                "candidate_designation",
                "FROZEN_FROM_METADATA_AND_NONNUMERIC_AXES",
                "ALL_NUMERIC_MATRICES_UNOPENED",
                "YES",
                "data/confirmation/gse342939_ra_bcell/candidate_designation_v1.json",
                "gse342939-ra-bcell-v1-candidate",
            ),
            _sequence_row(
                "gse342939_ra_bcell_source_terminal",
                3,
                "implementation_amendment",
                "FROZEN_BEFORE_ANY_NUMERIC_MATRIX_ACCESS",
                "ALL_NUMERIC_MATRICES_UNOPENED",
                "YES",
                (
                    "data/confirmation/gse342939_ra_bcell/"
                    "pre_access_implementation_amendment_v1.json"
                ),
                "gse342939-ra-bcell-v1-pre-access-amendment",
            ),
            _sequence_row(
                "gse342939_ra_bcell_source_terminal",
                4,
                "streaming_reduction_clarification",
                "FROZEN_BEFORE_ANY_NUMERIC_MATRIX_ACCESS",
                "ALL_NUMERIC_MATRICES_UNOPENED",
                "YES",
                (
                    "data/confirmation/gse342939_ra_bcell/"
                    "pre_access_streaming_reduction_clarification_v1.json"
                ),
                "gse342939-ra-bcell-v1-streaming-reduction-clarification",
            ),
            _sequence_row(
                "gse342939_ra_bcell_source_terminal",
                5,
                "implementation",
                "IMPLEMENTATION_FROZEN_BEFORE_NUMERIC_ACCESS",
                "ALL_NUMERIC_MATRICES_UNOPENED",
                "YES",
                "experiments/confirm_gse342939_ra_bcell.py",
                "gse342939-ra-bcell-v1-implementation",
            ),
            _sequence_row(
                "gse342939_ra_bcell_source_terminal",
                6,
                "source_attempt",
                "CLAIMED_BEFORE_FIRST_NUMERIC_MATRIX_GET",
                "SOURCE_GET_AUTHORIZED_HELD_DISABLED",
                "YES",
                "data/confirmation/gse342939_ra_bcell/source_attempt_v1.json",
                "gse342939-ra-bcell-v1-source-attempt",
            ),
            _sequence_row(
                "gse342939_ra_bcell_source_terminal",
                7,
                "source_consumption",
                "CONSUMED_EXCLUSIVELY_BEFORE_FIRST_NUMERIC_MATRIX_GET",
                "SOURCE_GET_CONSUMED_HELD_DISABLED",
                "NO",
                "data/confirmation/gse342939_ra_bcell/source_consumption_v1.json",
                "gse342939-ra-bcell-v1-source",
                notes=(
                    "The exclusive consumption record preceded the first source "
                    "GET and was published with the terminal result."
                ),
            ),
            _sequence_row(
                "gse342939_ra_bcell_source_terminal",
                8,
                "source_result",
                "TERMINAL_SOURCE_EXECUTION_REFUSAL",
                "ALL_SOURCE_MATRICES_REDUCED_HELD_UNOPENED",
                "NOT_APPLICABLE",
                "results/development/gse342939_ra_bcell_source_v1.json",
                "gse342939-ra-bcell-v1-source",
                notes=(
                    "No primary configuration completed every source fold; the "
                    "six held donors remained unopened."
                ),
            ),
            _sequence_row(
                "gse179221_bmmc_source_terminal",
                1,
                "protocol",
                "FROZEN_BEFORE_ANY_H5_BODY_ACCESS",
                "ALL_H5_BODIES_UNOPENED",
                "YES",
                "docs/GSE179221_BMMC_CITESEQ_HELD_DONOR_PROTOCOL_2026-08-29.md",
                "gse179221-bmmc-v1-candidate",
            ),
            _sequence_row(
                "gse179221_bmmc_source_terminal",
                2,
                "candidate_designation",
                "FROZEN_FROM_PUBLIC_METADATA_BEFORE_ANY_H5_BODY_ACCESS",
                "ALL_H5_BODIES_UNOPENED",
                "YES",
                "data/confirmation/gse179221_bmmc/candidate_designation_v1.json",
                "gse179221-bmmc-v1-candidate",
            ),
            _sequence_row(
                "gse179221_bmmc_source_terminal",
                3,
                "implementation_amendment",
                "FROZEN_BEFORE_ANY_H5_BODY_ACCESS",
                "ALL_H5_BODIES_UNOPENED",
                "YES",
                (
                    "data/confirmation/gse179221_bmmc/"
                    "pre_access_implementation_amendment_v1.json"
                ),
                "gse179221-bmmc-v1-pre-access-amendment",
            ),
            _sequence_row(
                "gse179221_bmmc_source_terminal",
                4,
                "implementation",
                "IMPLEMENTATION_FROZEN_BEFORE_ANY_H5_BODY_ACCESS",
                "ALL_H5_BODIES_UNOPENED",
                "YES",
                "experiments/confirm_gse179221_bmmc.py",
                "gse179221-bmmc-v1-implementation",
            ),
            _sequence_row(
                "gse179221_bmmc_source_terminal",
                5,
                "source_attempt",
                "CLAIMED_ONE_SHOT_BEFORE_ANY_H5_GET",
                "SOURCE_GET_AUTHORIZED_HELD_DISABLED",
                "YES",
                "data/confirmation/gse179221_bmmc/source_attempt_v1.json",
                "gse179221-bmmc-v1-source-attempt",
            ),
            _sequence_row(
                "gse179221_bmmc_source_terminal",
                6,
                "source_consumption",
                "CONSUMED_EXCLUSIVELY_BEFORE_FIRST_H5_GET",
                "SOURCE_GET_CONSUMED_HELD_DISABLED",
                "NO",
                "data/confirmation/gse179221_bmmc/source_consumption_v1.json",
                "gse179221-bmmc-v1-source",
                notes=(
                    "The exclusive local consumption record preceded the first "
                    "source GET and was published with the terminal result."
                ),
            ),
            _sequence_row(
                "gse179221_bmmc_source_terminal",
                7,
                "source_result",
                "TERMINAL_SOURCE_EXECUTION_REFUSAL",
                "ONE_SOURCE_FEATURE_AXIS_OPENED_NO_COUNT_DATASET_HELD_UNOPENED",
                "NOT_APPLICABLE",
                "results/development/gse179221_bmmc_source_v1.json",
                "gse179221-bmmc-v1-source",
                notes=(
                    "The first source donor failed exact cognate-axis uniqueness; "
                    "no numerical matrix dataset or held file was opened."
                ),
            ),
            _sequence_row(
                "gse214546_teaseq_source_terminal",
                1,
                "candidate_protocol",
                "FROZEN_BEFORE_NUMERIC_MATRIX_ACCESS",
                "ALL_NUMERIC_MATRICES_UNOPENED",
                "YES",
                "docs/GSE214546_TEASEQ_HELD_DONOR_CONFIRMATION_PROTOCOL_2026-08-30.md",
                "gse214546-teaseq-v1-candidate",
            ),
            _sequence_row(
                "gse214546_teaseq_source_terminal",
                2,
                "schema_amendment",
                "FROZEN_BEFORE_NUMERIC_MATRIX_ACCESS",
                "ALL_NUMERIC_MATRICES_UNOPENED",
                "YES",
                (
                    "data/confirmation/gse214546_teaseq/"
                    "pre_access_schema_amendment_v1.json"
                ),
                "gse214546-teaseq-v1-pre-access-amendment",
            ),
            _sequence_row(
                "gse214546_teaseq_source_terminal",
                3,
                "implementation_clarification",
                "FROZEN_BEFORE_NUMERIC_MATRIX_ACCESS",
                "ALL_NUMERIC_MATRICES_UNOPENED",
                "YES",
                (
                    "data/confirmation/gse214546_teaseq/"
                    "pre_access_implementation_clarification_v1.json"
                ),
                "gse214546-teaseq-v1-implementation-clarification",
            ),
            _sequence_row(
                "gse214546_teaseq_source_terminal",
                4,
                "cv_availability",
                "FROZEN_BEFORE_NUMERIC_MATRIX_ACCESS",
                "ALL_NUMERIC_MATRICES_UNOPENED",
                "YES",
                (
                    "data/confirmation/gse214546_teaseq/"
                    "pre_access_cv_availability_clarification_v1.json"
                ),
                "gse214546-teaseq-v1-cv-availability",
            ),
            _sequence_row(
                "gse214546_teaseq_source_terminal",
                5,
                "normalization_correction",
                "FROZEN_BEFORE_NUMERIC_MATRIX_ACCESS",
                "ALL_NUMERIC_MATRICES_UNOPENED",
                "YES",
                (
                    "data/confirmation/gse214546_teaseq/"
                    "pre_access_normalization_correction_v1.json"
                ),
                "gse214546-teaseq-v1-normalization-correction",
            ),
            _sequence_row(
                "gse214546_teaseq_source_terminal",
                6,
                "sparse_access_clarification",
                "FROZEN_BEFORE_NUMERIC_MATRIX_ACCESS",
                "ALL_NUMERIC_MATRICES_UNOPENED",
                "YES",
                (
                    "data/confirmation/gse214546_teaseq/"
                    "pre_access_sparse_access_clarification_v1.json"
                ),
                "gse214546-teaseq-v1-sparse-access-clarification",
            ),
            _sequence_row(
                "gse214546_teaseq_source_terminal",
                7,
                "crash_semantics",
                "FROZEN_BEFORE_NUMERIC_MATRIX_ACCESS",
                "ALL_NUMERIC_MATRICES_UNOPENED",
                "YES",
                (
                    "data/confirmation/gse214546_teaseq/"
                    "pre_access_crash_semantics_clarification_v1.json"
                ),
                "gse214546-teaseq-v1-crash-semantics-clarification",
            ),
            _sequence_row(
                "gse214546_teaseq_source_terminal",
                8,
                "implementation",
                "IMPLEMENTATION_FROZEN_BEFORE_NUMERIC_ACCESS",
                "SOURCE_ACCESS_DISABLED_HELD_DISABLED",
                "YES",
                "experiments/confirm_gse214546_teaseq.py",
                "gse214546-teaseq-v1-implementation",
            ),
            _sequence_row(
                "gse214546_teaseq_source_terminal",
                9,
                "source_attempt",
                "CLAIMED_BEFORE_FIRST_SOURCE_FILE_GET",
                "SOURCE_ACCESS_ENABLED_HELD_DISABLED",
                "YES",
                "data/confirmation/gse214546_teaseq/source_attempt_v1.json",
                "gse214546-teaseq-v1-source-attempt",
            ),
            _sequence_row(
                "gse214546_teaseq_source_terminal",
                10,
                "source_result",
                "TERMINAL_SOURCE_REFUSAL",
                "TWO_SOURCE_H5_REQUESTED_ONE_REDUCED_HELD_UNOPENED",
                "NOT_APPLICABLE",
                "results/development/gse214546_teaseq_source_v1.json",
                "gse214546-teaseq-v1-source-refusal",
                notes=(
                    "The first source donor completed reduction; the second had "
                    "fewer than 512 matched singlets. The exact overlap count was "
                    "not serialized, and all eight held H5s remained unopened."
                ),
            ),
            _sequence_row(
                "stephenson_unused_cambridge_terminal",
                1,
                "protocol",
                "PUBLIC_FREEZE",
                "DISABLED",
                "YES",
                "docs/STEPHENSON_UNUSED_CAMBRIDGE_HELD_DONOR_PROTOCOL_2026-08-29.md",
                "stephenson-unused-cambridge-v1-protocol",
            ),
            _sequence_row(
                "stephenson_unused_cambridge_terminal",
                2,
                "preaccess",
                "PASS_H5AD_UNOPENED",
                "H5AD_UNOPENED",
                "YES",
                "results/development/stephenson_unused_cambridge_preaccess_v1.json",
            ),
            _sequence_row(
                "stephenson_unused_cambridge_terminal",
                3,
                "recovery_amendment",
                "ONE_REPLACEMENT_DEFINED",
                "RNA_MARGIN_ONLY_REPLACEMENT",
                "YES",
                "docs/STEPHENSON_UNUSED_CAMBRIDGE_PREDICTION_RECOVERY_AMENDMENT_2026-08-29.md",
                "stephenson-unused-cambridge-v1.1-recovery",
            ),
            _sequence_row(
                "stephenson_unused_cambridge_terminal",
                4,
                "prediction_and_score",
                "TERMINAL_INFRASTRUCTURE_UNEVALUABLE",
                "NO_PREDICTION_OR_SCORE_RECORDED",
                "NOT_APPLICABLE",
                "data/confirmation/stephenson_unused_cambridge/prediction_terminal_record_v1_1.json",
                "stephenson-unused-cambridge-v1.1-terminal",
                notes="Published terminal infrastructure record; neither a scientific pass nor a scientific failure.",
            ),
        )
    )
    return sorted(rows, key=lambda row: (row["panel_id"], row["stage_ordinal"]))


def _write_tsv(relative: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            if set(row) != set(fields):
                raise ValueError(f"row fields differ for {relative}: {row.get(fields[0])}")
            writer.writerow({field: "" if row[field] is None else row[field] for field in fields})


def _artifact_manifest(
    panels: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    sequence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    referenced = {
        row["result_artifact"] for row in panels if row["result_artifact"]
    }
    referenced.update(
        row["result_artifact"] for row in comparisons if row["result_artifact"]
    )
    referenced.update(row["artifact"] for row in sequence if row["artifact"])
    referenced.update(
        {
            str(PANELS_PATH),
            str(COMPARISONS_PATH),
            str(SEQUENCE_PATH),
            "results/final_public_benchmark_table.tsv",
            "results/development/gse299043_mln_development_acquisition_refusal.json",
            "LICENSE",
            "README.md",
            "docs/FINAL_PUBLIC_EVIDENCE_LEDGER.md",
            "experiments/build_public_benchmark_release.py",
            "scripts/verify_public_benchmark_release.py",
            "tests/test_public_benchmark_release_v2.py",
            "tests/test_gse179221_candidate.py",
            "tests/test_gse179221_bmmc_confirmation.py",
            "experiments/correct_gse239452_residual_inversion.py",
            "experiments/development/simulate_exact_conditional_heterogeneity.py",
            "experiments/development/stress_transfer_risk_bound.py",
            "experiments/evaluate_gse239452_standard_poisson_posthoc.py",
            "results/development/exact_conditional_heterogeneity_simulation_v1.json",
            "results/development/exact_conditional_heterogeneity_simulation_v1.tsv",
            "results/development/transfer_risk_bound_stress_v1.json",
            "results/development/transfer_risk_bound_stress_v1.tsv",
            "results/gse239452_citeseq_post_access_correction.json",
            "tests/test_exact_conditional_heterogeneity_simulation.py",
            "tests/test_gse239452_post_access_correction.py",
            "tests/test_gse239452_standard_poisson_posthoc.py",
            "tests/test_transfer_risk_bound_stress.py",
            "docs/FIXED_MARGIN_PREDICTIVE_REANALYSIS.md",
            "data/development/stephenson_sufficient_tables.npz",
            "data/development/stephenson_biological_sufficient_tables.npz",
            "data/development/stephenson_assay_qc_diagnostics.json",
            "mapreg/predictive_conditional.py",
            "experiments/development/reanalyze_stephenson_prediction.py",
            "experiments/development/fit_conditional_random_effects.R",
            "experiments/development/fixed_margin_biology.py",
            "experiments/development/assay_qc_sensitivity.py",
            "experiments/development/simulate_recipient_heterogeneity.py",
            "results/development/stephenson_predictive_reanalysis.json",
            "results/development/stephenson_predictive_reanalysis_summary.json",
            "results/development/stephenson_predictive_environment.json",
            "results/development/stephenson_biological_reanalysis.json",
            "results/development/stephenson_biological_reanalysis_replay.json",
            "results/development/stephenson_assay_qc_sensitivity.json",
            "results/development/recipient_heterogeneity_reanalysis.json",
            "reproduce.sh",
        }
    )
    artifacts = []
    for relative in sorted(referenced):
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        artifacts.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256(relative),
            }
        )
    return artifacts


def _write_manifest(
    panels: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    sequence: list[dict[str, Any]],
) -> None:
    scored = sum(row["outcome_scored"] == "YES" for row in panels)
    procedural = sum(row["inference_role"] == "procedural_refusal" for row in panels)
    infrastructure_unevaluable = sum(
        row["inference_role"] == "infrastructure_unevaluable" for row in panels
    )
    record = {
        "schema": "coupling-fields-public-benchmark/2.0",
        "release_name": "coupling-fields-v2-public-benchmark",
        "release_status": "RELEASED",
        "snapshot_date": "2026-09-04",
        "public_repository_url": "https://github.com/sushaan-k/coupling-fields-benchmark",
        "intended_release_tag": "coupling-fields-v2.0.4-public-benchmark",
        "archive_doi": None,
        "code_license": "MIT",
        "analysis_plan_characterization": "original pre-outcome plans and separately identified post hoc specifications; not registry-hosted preregistration",
        "post_hoc_extensions": {
            "confirmatory": False,
            "included_in_original_panel_count": False,
            "specification": "docs/FIXED_MARGIN_PREDICTIVE_REANALYSIS.md",
            "results": [
                "results/development/stephenson_predictive_reanalysis.json",
                "results/development/stephenson_biological_reanalysis.json",
                "results/development/stephenson_assay_qc_sensitivity.json",
                "results/development/recipient_heterogeneity_reanalysis.json",
            ],
        },
        "ledgers": {
            "panels": str(PANELS_PATH),
            "comparisons": str(COMPARISONS_PATH),
            "sequence": str(SEQUENCE_PATH),
            "historical_v1_table": "results/final_public_benchmark_table.tsv",
        },
        "counts": {
            "panel_records": len(panels),
            "scored_panel_records": scored,
            "procedural_refusal_records": procedural,
            "pending_records": 0,
            "infrastructure_unevaluable_records": infrastructure_unevaluable,
            "comparison_records": len(comparisons),
            "sequence_records": len(sequence),
        },
        "infrastructure_unevaluable": {
            "panel_id": "stephenson_unused_cambridge_terminal",
            "performance_values_recorded": False,
            "terminal_record": "data/confirmation/stephenson_unused_cambridge/prediction_terminal_record_v1_1.json",
            "scientific_decision": None,
        },
        "artifacts": _artifact_manifest(panels, comparisons, sequence),
    }
    (ROOT / MANIFEST_PATH).write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _checksum_paths() -> list[str]:
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT
    ).decode().split("\0")
    generated = {
        str(PANELS_PATH),
        str(COMPARISONS_PATH),
        str(SEQUENCE_PATH),
        str(MANIFEST_PATH),
        "experiments/build_public_benchmark_release.py",
        "experiments/correct_gse239452_residual_inversion.py",
        "experiments/development/simulate_exact_conditional_heterogeneity.py",
        "experiments/development/stress_transfer_risk_bound.py",
        "results/development/exact_conditional_heterogeneity_simulation_v1.json",
        "results/development/exact_conditional_heterogeneity_simulation_v1.tsv",
        "results/development/transfer_risk_bound_stress_v1.json",
        "results/development/transfer_risk_bound_stress_v1.tsv",
        "results/gse239452_citeseq_post_access_correction.json",
        "scripts/verify_public_benchmark_release.py",
        "tests/test_exact_conditional_heterogeneity_simulation.py",
        "tests/test_gse239452_post_access_correction.py",
        "tests/test_public_benchmark_release_v2.py",
        "tests/test_transfer_risk_bound_stress.py",
        "results/development/gse214546_teaseq_source_v1.json",
        "LICENSE",
    }
    candidates = set(filter(None, tracked)) | generated
    candidates.discard(str(CHECKSUM_PATH))
    candidates.difference_update(ABSENT_UNUSED_CAMBRIDGE_PATHS)
    return sorted(relative for relative in candidates if (ROOT / relative).is_file())


def _write_checksums() -> None:
    lines = [f"{_sha256(relative)}  {relative}" for relative in _checksum_paths()]
    (ROOT / CHECKSUM_PATH).write_text("\n".join(lines) + "\n", encoding="utf-8")


def build() -> None:
    panels, comparisons = _legacy_panels()
    new_panels, new_comparisons = _new_completed_evidence()
    panels.extend(new_panels)
    comparisons.extend(new_comparisons)
    panels.extend(_terminal_panels())
    panels.sort(key=lambda row: row["panel_id"])
    comparisons.sort(key=lambda row: row["comparison_id"])
    if len({row["panel_id"] for row in panels}) != len(panels):
        raise ValueError("duplicate panel_id")
    if len({row["comparison_id"] for row in comparisons}) != len(comparisons):
        raise ValueError("duplicate comparison_id")
    panel_ids = {row["panel_id"] for row in panels}
    if any(row["panel_id"] not in panel_ids for row in comparisons):
        raise ValueError("comparison references an unknown panel")
    sequence = _sequence(panels)

    _write_tsv(PANELS_PATH, PANEL_FIELDS, panels)
    _write_tsv(COMPARISONS_PATH, COMPARISON_FIELDS, comparisons)
    _write_tsv(SEQUENCE_PATH, SEQUENCE_FIELDS, sequence)
    _write_manifest(panels, comparisons, sequence)
    _write_checksums()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="rebuild in place and fail if any generated byte would change",
    )
    args = parser.parse_args()
    outputs = (PANELS_PATH, COMPARISONS_PATH, SEQUENCE_PATH, MANIFEST_PATH, CHECKSUM_PATH)
    before = {
        relative: (ROOT / relative).read_bytes()
        for relative in outputs
        if (ROOT / relative).is_file()
    }
    build()
    if args.check:
        missing = [str(relative) for relative in outputs if relative not in before]
        changed = [
            str(relative)
            for relative in outputs
            if relative in before and before[relative] != (ROOT / relative).read_bytes()
        ]
        if missing or changed:
            for relative in outputs:
                path = ROOT / relative
                if relative in before:
                    path.write_bytes(before[relative])
                elif path.exists():
                    path.unlink()
            raise SystemExit(
                f"release outputs are stale; missing={missing}, changed={changed}"
            )
    print("\n".join(str(relative) for relative in outputs))


if __name__ == "__main__":
    main()
