from __future__ import annotations

import json

import numpy as np
import pytest

from experiments import evaluate_gse279451_sepsis_development as subject
from experiments import confirm_gse279451_sepsis as runner
from mapreg.heterogeneity_adaptive_coupling import (
    expected_binary_table_from_log_odds,
)


def _donor_tables(donor: int) -> np.ndarray:
    tables = np.empty((9, 9, 2, 2), dtype=np.int64)
    for rna in range(9):
        row_zero = 220 + (17 * donor + 29 * rna) % 460
        for adt in range(9):
            signal = ((3 * donor + 5 * rna + 7 * adt) % 31) - 15
            upper_left = row_zero // 2 + signal
            tables[rna, adt] = np.array(
                [
                    [upper_left, row_zero - upper_left],
                    [512 - upper_left, 512 - row_zero + upper_left],
                ]
            )
    return tables


def _reduced_payload(development_attempt_sha256: str = "d" * 64) -> dict[str, object]:
    records = []
    for donor, accession in enumerate(subject.DEVELOPMENT_DONORS):
        tables = _donor_tables(donor)
        destroyed = tables.copy()
        for rna in range(9):
            row_zero = int(tables[rna, 0, 0].sum())
            upper_left = row_zero // 2
            destroyed[rna, :, 0, 0] = upper_left
            destroyed[rna, :, 0, 1] = row_zero - upper_left
            destroyed[rna, :, 1, 0] = 512 - upper_left
            destroyed[rna, :, 1, 1] = 512 - row_zero + upper_left
        records.append(
            {
                "accession": accession,
                "sample": f"sample-{donor:02d}",
                "role": "development",
                "cells": 1024,
                "cell_selection_salt": subject.reducer.CELL_SELECTION_SALT,
                "markers": list(subject.MARKERS),
                "entity_count": 81,
                "tables": tables.reshape(81, 4).tolist(),
                "destroyed_tables": destroyed.reshape(81, 4).tolist(),
                "informative": [True] * 81,
                "rna_detection_prevalence": (
                    tables[:, 0, 1].sum(axis=1) / 1024.0
                ).tolist(),
                "adt_log_panel_fraction_mean": [
                    0.2 + 0.01 * donor + 0.03 * marker for marker in range(9)
                ],
                "predictions_materialized_sha256": None,
                "selected_barcode_axis_sha256": f"{donor:064x}",
                "matrix_sha256": f"{donor + 100:064x}",
            }
        )
    return {
        "schema": "gse279451-sepsis-reduced-development/1.0",
        "status": "NONHELD_REDUCTION_COMPLETE",
        "development_attempt_sha256": development_attempt_sha256,
        "source_manifest_sha256": "a" * 64,
        "development_donors": list(subject.DEVELOPMENT_DONORS),
        "held_donors": list(subject.HELD_DONORS),
        "markers": list(subject.MARKERS),
        "entity_count": 81,
        "primary_cells_per_donor": 1024,
        "cell_selection_salt": subject.reducer.CELL_SELECTION_SALT,
        "all_cells_sensitivity_included": False,
        "donors": records,
        "access_audit": {
            "development_matrix_members_decoded": 19,
            "held_matrix_members_opened": 0,
            "held_matrix_entries_decoded": 0,
            "maximum_concurrent_donor_matrices": 1,
        },
    }


def test_reduced_validator_accepts_only_the_19_sealed_donors(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = tmp_path / "development-attempt.json"
    attempt.write_text(json.dumps({"status": "TERMINAL_DEVELOPMENT_ATTEMPT_STARTED"}))
    monkeypatch.setattr(subject.reducer, "DEVELOPMENT_ATTEMPT", attempt)
    attempt_hash = subject._sha256(attempt)
    path = tmp_path / "reduced.json"
    path.write_text(json.dumps(_reduced_payload(attempt_hash)))
    data = subject._validated_reduced(path)
    assert data["tables"].shape == (19, 9, 9, 2, 2)
    assert data["destroyed_tables"].shape == data["tables"].shape
    assert np.all(data["support_counts"] == 81)

    payload = _reduced_payload(attempt_hash)
    payload["donors"][0]["accession"] = subject.HELD_DONORS[0]
    path.write_text(json.dumps(payload))
    with pytest.raises(PermissionError, match="donor order"):
        subject._validated_reduced(path)


def test_fold_graph_is_a_deterministic_undirected_tie_union() -> None:
    donor_axis = np.linspace(-2.0, 3.0, 18)
    profiles = np.tile(donor_axis[:, None], (1, 9))
    incidence = subject._knn_edge_incidence(profiles, 1)
    expected = np.zeros((9, 8))
    for edge, marker in enumerate(range(1, 9)):
        expected[0, edge] = 1.0
        expected[marker, edge] = 1.0
    np.testing.assert_array_equal(incidence, expected)
    np.testing.assert_array_equal(subject._knn_edge_incidence(profiles, 1), incidence)

    poisoned_omitted = np.full((1, 9), 1e100)
    np.testing.assert_array_equal(
        subject._knn_edge_incidence(np.vstack((profiles, poisoned_omitted))[:-1], 1),
        incidence,
    )


def test_fold_graph_refuses_a_zero_variance_marker() -> None:
    profiles = np.arange(18 * 9, dtype=float).reshape(18, 9)
    profiles[:, 4] = 2.0
    with pytest.raises(subject.GraphConstructionRefusal, match="CD33"):
        subject._knn_edge_incidence(profiles, 2)


def test_exact_conditional_prediction_matches_scalar_reference() -> None:
    tables = _donor_tables(3).reshape(81, 2, 2)
    recipient = subject._conditional_support(tables)
    log_odds = np.linspace(-1.5, 1.5, 81)
    predicted = subject._conditional_expected_tables(log_odds, recipient)
    for entity in (0, 17, 40, 80):
        table = tables[entity]
        expected = expected_binary_table_from_log_odds(
            log_odds[entity], table.sum(axis=1), table.sum(axis=0)
        )
        np.testing.assert_allclose(predicted[entity], expected, atol=1e-12)
    np.testing.assert_allclose(predicted.sum(axis=-1), tables.sum(axis=-1))
    np.testing.assert_allclose(predicted.sum(axis=-2), tables.sum(axis=-2))


def test_development_predictions_depend_on_target_margins_not_target_pairing() -> None:
    tables = _donor_tables(5).reshape(81, 2, 2)
    altered = tables.copy()
    altered[:, 0, 0] += 1
    altered[:, 0, 1] -= 1
    altered[:, 1, 0] -= 1
    altered[:, 1, 1] += 1
    log_odds = np.linspace(-0.5, 0.5, 81)
    first = subject._conditional_expected_tables(
        log_odds, subject._conditional_support(tables)
    )
    second = subject._conditional_expected_tables(
        log_odds, subject._conditional_support(altered)
    )
    np.testing.assert_allclose(first, second, atol=0.0, rtol=0.0)

    pooled = np.linspace(-0.01, 0.01, 81)
    for family in ("pearson", "deviance"):
        first = subject._predict_residual(
            pooled,
            tables,
            family=family,
            centered=True,
            alpha=1.0,
            target_null=subject._target_null_mean(tables, family),
        )
        second = subject._predict_residual(
            pooled,
            altered,
            family=family,
            centered=True,
            alpha=1.0,
            target_null=subject._target_null_mean(altered, family),
        )
        np.testing.assert_allclose(first, second, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("family", ["pearson", "deviance"])
@pytest.mark.parametrize("centered", [False, True])
def test_classical_head_to_head_preserves_target_margins(
    family: str, centered: bool
) -> None:
    source = np.stack([_donor_tables(donor) for donor in range(4)])
    target = _donor_tables(7)
    pooled, certificate = subject._residual_pool(source, family, centered)
    predicted = subject._predict_residual(
        pooled,
        target,
        family=family,
        centered=centered,
        alpha=1.25,
        target_null=subject._target_null_mean(target, family),
    )
    assert certificate["sample_size_normalized"]
    np.testing.assert_allclose(
        predicted.sum(axis=-1), target.reshape(81, 2, 2).sum(axis=-1), atol=1e-8
    )
    np.testing.assert_allclose(
        predicted.sum(axis=-2), target.reshape(81, 2, 2).sum(axis=-2), atol=1e-8
    )
    assert np.isfinite(subject._donor_loss(target, predicted))


def test_candidate_ledger_keeps_every_donor_loss_and_refusal() -> None:
    donors = list(subject.DEVELOPMENT_DONORS)
    book = subject._CandidateBook(
        "common_effect_ridge_only", [(0.01, 1.0), (0.1, 1.0)], donors
    )
    for fold in range(19):
        book.record((0.01, 1.0), fold, 0.8 + fold / 1000)
        book.record((0.1, 1.0), fold, 0.7 + fold / 1000)
    book.refuse((0.1, 1.0), 5, "certificate refusal")
    assert book.selected() == (0.01, 1.0)
    diagnostics = book.diagnostics()
    assert diagnostics["eligible_candidates"] == 1
    assert set(diagnostics["candidates"][0]["donor_losses"]) == set(donors)
    assert diagnostics["candidates"][1]["refusals"][donors[5]]


def test_development_gate_bootstrap_is_donor_labeled_and_locked() -> None:
    donors = list(subject.DEVELOPMENT_DONORS)
    primary = np.full(19, 0.80)
    comparator = np.full(19, 1.00)
    first = subject._comparison(donors, primary, comparator, "best_residual")
    second = subject._comparison(donors, primary, comparator, "best_residual")
    assert first == second
    assert first["relative_reduction"] == pytest.approx(0.20)
    assert first["bootstrap_upper_95"] < 0.0
    assert first["favorable_donors"] == 19
    assert first["passes_all"]
    assert set(first["donor_differences_primary_minus_comparator"]) == set(donors)
    generator = np.random.default_rng(subject.BOOTSTRAP_SEED)
    indices = generator.integers(0, 19, size=(subject.BOOTSTRAPS, 19), endpoint=False)
    reference = runner._gate_comparison(
        tuple(donors), primary, comparator, indices, favorable_required=15
    )
    for key in (
        "relative_reduction",
        "bootstrap_upper_95",
        "favorable_donors",
        "donor_differences_primary_minus_comparator",
        "passes_all",
    ):
        assert first[key] == reference[key]


def test_evaluator_json_is_strict_before_exclusive_write(tmp_path) -> None:
    poison = tmp_path / "poison.json"
    poison.write_text('{"value": Infinity}\n')
    with pytest.raises(ValueError, match="nonfinite JSON"):
        subject._validated_reduced(poison)
    output = tmp_path / "output.json"
    with pytest.raises(ValueError):
        subject._write_json_exclusive(output, {"value": float("nan")})
    assert not output.exists()


def test_evaluation_attempt_precedes_reduced_parse_and_failure_is_terminal(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reduced = tmp_path / "reduced.json"
    reduced.write_text('{"value": NaN}\n')
    acquisition_attempt = tmp_path / "development-attempt.json"
    acquisition_attempt.write_text(
        json.dumps({"status": "TERMINAL_DEVELOPMENT_ATTEMPT_STARTED"})
    )
    monkeypatch.setattr(subject, "INPUT", reduced)
    monkeypatch.setattr(
        subject, "EVALUATION_ATTEMPT", tmp_path / "evaluation-attempt.json"
    )
    monkeypatch.setattr(
        subject, "EVALUATION_REFUSAL", tmp_path / "evaluation-refusal.json"
    )
    monkeypatch.setattr(subject, "OUTPUT", tmp_path / "development.json")
    monkeypatch.setattr(subject.reducer, "DEVELOPMENT_ATTEMPT", acquisition_attempt)
    with pytest.raises(ValueError, match="nonfinite JSON"):
        subject.run_development(workers=1)
    assert subject.EVALUATION_ATTEMPT.is_file()
    refusal = json.loads(subject.EVALUATION_REFUSAL.read_text())
    assert refusal["status"] == "TERMINAL_DEVELOPMENT_EVALUATION_REFUSAL"
    assert refusal["development_attempt_sha256"] == subject._sha256(acquisition_attempt)
    assert refusal["rerun_permitted"] is False
    assert not subject.OUTPUT.exists()
    with pytest.raises(FileExistsError, match="evaluation artifact"):
        subject.run_development(workers=1)


def test_completed_gate_miss_is_a_result_but_unavailable_or_refit_is_refusal() -> None:
    assert subject._completed_development_status(True, [], None) == "DEVELOPMENT_PASS"
    assert subject._completed_development_status(False, [], None) == "DEVELOPMENT_FAIL"

    with pytest.raises(subject.DevelopmentEvaluationRefusal) as unavailable:
        subject._completed_development_status(False, ["primary"], None)
    assert unavailable.value.detail == {
        "unavailable_candidate_families": ["primary"],
        "final_refit_error": None,
    }

    with pytest.raises(subject.DevelopmentEvaluationRefusal) as refit:
        subject._completed_development_status(False, [], "optimizer refused")
    assert refit.value.detail == {
        "unavailable_candidate_families": [],
        "final_refit_error": "optimizer refused",
    }


def test_unavailable_family_writes_terminal_refusal_not_development_output(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reduced = tmp_path / "reduced.json"
    reduced.write_text("{}\n")
    acquisition_attempt = tmp_path / "development-attempt.json"
    acquisition_attempt.write_text(
        json.dumps({"status": "TERMINAL_DEVELOPMENT_ATTEMPT_STARTED"})
    )
    monkeypatch.setattr(subject, "INPUT", reduced)
    monkeypatch.setattr(
        subject, "EVALUATION_ATTEMPT", tmp_path / "evaluation-attempt.json"
    )
    monkeypatch.setattr(
        subject, "EVALUATION_REFUSAL", tmp_path / "evaluation-refusal.json"
    )
    monkeypatch.setattr(subject, "OUTPUT", tmp_path / "development.json")
    monkeypatch.setattr(subject.reducer, "DEVELOPMENT_ATTEMPT", acquisition_attempt)
    monkeypatch.setattr(subject, "_validated_reduced", lambda path: {})
    monkeypatch.setattr(
        subject,
        "_run_development_after_attempt",
        lambda data, workers: subject._completed_development_status(
            False, ["primary"], None
        ),
    )

    with pytest.raises(subject.DevelopmentEvaluationRefusal):
        subject.run_development(workers=1)
    refusal = json.loads(subject.EVALUATION_REFUSAL.read_text())
    assert refusal["evaluation_detail"] == {
        "unavailable_candidate_families": ["primary"],
        "final_refit_error": None,
    }
    assert not subject.OUTPUT.exists()


def test_parallel_fold_aggregation_matches_serial_with_lightweight_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def lightweight(
        data: dict[str, object], folds: tuple[int, ...] | None = None
    ) -> dict[str, object]:
        donors = data["donors"]
        books = subject._candidate_books(donors)
        independence = np.full(len(donors), np.nan)
        audits = []
        selected_folds = tuple(range(len(donors))) if folds is None else folds
        for fold in selected_folds:
            independence[fold] = 1.0 + fold / 1000.0
            audits.append({"omitted_donor": donors[fold], "status": "OK"})
            for family_index, book in enumerate(books.values()):
                for config_index, config in enumerate(book.configs):
                    book.record(
                        config,
                        fold,
                        0.5
                        + family_index / 100.0
                        + config_index / 1_000_000.0
                        + fold / 10_000.0,
                        {"fold": fold},
                    )
        return {
            "books": books,
            "independence": independence,
            "fold_graph_audit": audits,
        }

    monkeypatch.setattr(subject, "_cross_validate_serial", lightweight)
    data = {"donors": list(subject.DEVELOPMENT_DONORS)}
    serial = subject._cross_validate(data, workers=1)
    parallel = subject._cross_validate(data, workers=2)
    np.testing.assert_array_equal(serial["independence"], parallel["independence"])
    assert serial["fold_graph_audit"] == parallel["fold_graph_audit"]
    for family in serial["books"]:
        assert (
            serial["books"][family].diagnostics()
            == parallel["books"][family].diagnostics()
        )


def test_evaluator_has_no_raw_or_held_matrix_access_path() -> None:
    source = subject.Path(subject.__file__).read_text()
    assert "_stream_selected_rows" not in source
    assert "_validated_member" not in source
    assert "matrix.mtx" not in source
    assert "SOURCE_MANIFEST" not in source
