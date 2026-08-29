from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments.build_public_benchmark_release import (
    CHECKSUM_PATH,
    COMPARISONS_PATH,
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

    assert len(panels) == 32
    assert len(comparisons) == 28
    assert len(sequence) == 51
    assert sum(row["outcome_scored"] == "YES" for row in panels) == 12
    assert sum(row["inference_role"] == "procedural_refusal" for row in panels) == 19

    stephenson = by_panel["stephenson_newcastle_confirmation"]
    assert stephenson["inference_role"] == "confirmatory"
    assert float(stephenson["primary_value"]) == 0.012196845982760404
    residual = by_comparison["stephenson_newcastle_confirmation__best_residual"]
    assert float(residual["relative_improvement"]) == 0.17460498883932063
    assert float(residual["paired_difference_ci_95_high"]) < 0
    assert residual["decision"] == "PASS"

    gse239 = by_panel["gse239452_held_post_access_correction"]
    assert gse239["inference_role"] == "post_access_correction"
    exact = by_comparison[
        "gse239452_held_post_access_correction__posthoc_primary_vs_common_effect_exact_cmle"
    ]
    assert float(exact["relative_improvement"]) < 0
    assert exact["inference_role"] == "post_hoc_nonconfirmatory"

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


def test_manifest_counts_and_claim_boundaries_match_ledgers() -> None:
    manifest = json.loads((ROOT / MANIFEST_PATH).read_text(encoding="utf-8"))
    assert manifest["schema"] == "coupling-fields-public-benchmark/2.0"
    assert manifest["counts"] == {
        "comparison_records": 28,
        "panel_records": 32,
        "infrastructure_unevaluable_records": 1,
        "pending_records": 0,
        "procedural_refusal_records": 19,
        "scored_panel_records": 12,
        "sequence_records": 51,
    }
    assert manifest["archive_doi"] is None
    assert manifest["code_license"] is None
    assert "not registry-hosted" in manifest["analysis_plan_characterization"]
    assert (
        manifest["infrastructure_unevaluable"]["performance_values_recorded"]
        is False
    )
    assert manifest["infrastructure_unevaluable"]["scientific_decision"] is None
