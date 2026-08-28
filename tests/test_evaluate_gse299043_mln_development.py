from __future__ import annotations

import json

import numpy as np
import pytest

from experiments import evaluate_gse299043_mln_development as subject


def _donor_tables(donor: int) -> np.ndarray:
    tables = np.empty((9, 9, 2, 2), dtype=np.int64)
    for rna in range(9):
        row_zero = 130 + (31 * donor + 23 * rna) % 250
        lower = max(0, row_zero - 256)
        upper = min(row_zero, 256)
        midpoint = (lower + upper) // 2
        for adt in range(9):
            shift = ((5 * donor + 7 * rna + 11 * adt) % 17) - 8
            upper_left = min(max(midpoint + shift, lower), upper)
            tables[rna, adt] = np.asarray(
                [
                    [upper_left, row_zero - upper_left],
                    [256 - upper_left, 256 - row_zero + upper_left],
                ]
            )
    return tables


def _reduced_payload(
    development_attempt_sha256: str,
    source_manifest_sha256: str,
) -> dict[str, object]:
    records = []
    piece_counts = [6] * 6 + [5] * 4
    piece_index = 0
    for donor_index, donor in enumerate(subject.DEVELOPMENT_DONORS):
        tables = _donor_tables(donor_index)
        destroyed = tables.copy()
        for rna in range(9):
            row_zero = int(tables[rna, 0, 0].sum())
            upper_left = min(max(row_zero // 2, row_zero - 256), 256)
            destroyed[rna, :, 0, 0] = upper_left
            destroyed[rna, :, 0, 1] = row_zero - upper_left
            destroyed[rna, :, 1, 0] = 256 - upper_left
            destroyed[rna, :, 1, 1] = 256 - row_zero + upper_left
        pieces = []
        for _ in range(piece_counts[donor_index]):
            pieces.append(
                {
                    "source_filename": f"GSE299043_{donor}_{piece_index:03d}.h5ad",
                    "source_sha256": f"{piece_index + 1:064x}",
                    "piece_sha256": f"{piece_index + 1000:064x}",
                }
            )
            piece_index += 1
        records.append(
            {
                "schema": "gse299043-mln-reduced-donor/1.0",
                "status": "DONOR_REDUCTION_COMPLETE",
                "donor": donor,
                "role": "development",
                "markers": list(subject.MARKERS),
                "entity_count": 81,
                "cells": 512,
                "cell_selection_salt": subject.reducer.CELL_SELECTION_SALT,
                "adt_tie_salt": subject.reducer.ADT_TIE_SALT,
                "selected_cell_axis_sha256": f"{donor_index + 2000:064x}",
                "rna_detection_prevalence": (
                    tables[:, 0, 1, :].sum(axis=-1) / 512.0
                ).tolist(),
                "adt_log_panel_fraction_mean": [
                    0.2 + donor_index / 100.0 + marker / 50.0
                    for marker in range(9)
                ],
                "tables": tables.reshape(81, 4).tolist(),
                "destroyed_tables": destroyed.reshape(81, 4).tolist(),
                "library_pieces": pieces,
            }
        )
    assert piece_index == 56
    return {
        "schema": "gse299043-mln-reduced-development/1.0",
        "status": "NONHELD_REDUCTION_COMPLETE",
        "development_attempt_sha256": development_attempt_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "development_donors": list(subject.DEVELOPMENT_DONORS),
        "held_donors": list(subject.HELD_DONORS),
        "markers": list(subject.MARKERS),
        "entity_count": 81,
        "primary_cells_per_donor": 512,
        "cell_selection_salt": subject.reducer.CELL_SELECTION_SALT,
        "adt_tie_salt": subject.reducer.ADT_TIE_SALT,
        "donors": records,
        "access_audit": {
            "development_h5ad_members_decoded": 56,
            "held_h5ad_members_opened": 0,
            "held_h5ad_members_decoded": 0,
            "maximum_concurrent_source_h5ads": 1,
        },
    }


def _write_reduced_fixture(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> tuple[object, object, object]:
    attempt = tmp_path / "development-attempt.json"
    source_template = tmp_path / "source-template.json"
    metadata = tmp_path / "metadata.tsv"
    source = tmp_path / "source-manifest.json"
    reduced = tmp_path / "reduced.json"
    source_template.write_text('{"source": "disabled"}\n')
    metadata.write_text("member\trole\n")
    bindings = {
        "development_evaluator_sha256": subject._sha256(subject.Path(subject.__file__)),
        "reducer_sha256": subject._sha256(subject.Path(subject.reducer.__file__)),
        **subject._transitive_bindings(),
    }
    development_names = [
        piece["source_filename"]
        for donor in _reduced_payload("d" * 64, "s" * 64)["donors"]
        for piece in donor["library_pieces"]
    ]
    members = [
        {
            "filename": filename,
            "donor": subject.DEVELOPMENT_DONORS[index % 10],
            "role": "development",
            "sha256": f"{index + 1:064x}",
            "local_path": None,
            "retained": False,
        }
        for index, filename in enumerate(development_names)
    ]
    members.extend(
        {
            "filename": f"held-{index:03d}.h5ad",
            "donor": subject.HELD_DONORS[index % 10],
            "role": "held",
            "sha256": None,
            "local_path": None,
            "retained": False,
        }
        for index in range(151)
    )
    source.write_text(
        json.dumps(
            {
                "schema": "gse299043-mln-source/1.0",
                "status": "NONHELD_SOURCE_ACCESS_AUTHORIZED",
                "bindings": bindings,
                "members": members,
                "access_audit": {
                    "development_h5ad_members_decoded": 56,
                    "held_h5ad_members_requested": 0,
                    "held_h5ad_members_opened": 0,
                    "held_h5ad_members_decoded": 0,
                    "maximum_concurrent_source_h5ads": 1,
                },
            }
        )
        + "\n"
    )
    source_hash = subject._sha256(source)
    attempt.write_text(
        json.dumps(
            {
                "status": "TERMINAL_DEVELOPMENT_ATTEMPT_STARTED",
                "source_template_sha256": subject._sha256(source_template),
                "metadata_preflight_sha256": subject._sha256(metadata),
                "artifact_bindings": bindings,
                "development_members_planned": 56,
                "held_h5ad_members_requested": 0,
                "first_network_request_starts_after_this_write": True,
                "rerun_permitted": False,
            }
        )
    )
    reduced.write_text(
        json.dumps(_reduced_payload(subject._sha256(attempt), source_hash))
    )
    monkeypatch.setattr(subject, "DEVELOPMENT_ATTEMPT", attempt)
    monkeypatch.setattr(subject, "SOURCE_TEMPLATE", source_template)
    monkeypatch.setattr(subject, "METADATA_PREFLIGHT", metadata)
    monkeypatch.setattr(subject, "SOURCE_MANIFEST", source)
    return reduced, attempt, source


def test_reduced_validator_accepts_only_the_frozen_ten_donors(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reduced, _, _ = _write_reduced_fixture(tmp_path, monkeypatch)
    data = subject._validated_reduced(reduced)
    assert data["tables"].shape == (10, 9, 9, 2, 2)
    assert data["destroyed_tables"].shape == data["tables"].shape
    assert np.all(data["support_counts"] == 81)

    payload = json.loads(reduced.read_text())
    payload["donors"][0]["donor"] = subject.HELD_DONORS[0]
    reduced.write_text(json.dumps(payload))
    with pytest.raises(PermissionError, match="donor order"):
        subject._validated_reduced(reduced)


def test_reduced_validator_requires_all_56_unique_development_members(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reduced, _, _ = _write_reduced_fixture(tmp_path, monkeypatch)
    payload = json.loads(reduced.read_text())
    payload["donors"][-1]["library_pieces"].pop()
    reduced.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="all 56"):
        subject._validated_reduced(reduced)


def test_candidate_families_are_prespecified_without_common_effect() -> None:
    books = subject._candidate_books(list(subject.DEVELOPMENT_DONORS))
    assert tuple(books) == (
        "primary",
        "destroyed_link",
        "label_permuted_graph",
        "hierarchical_ridge_only",
        "best_residual",
    )
    assert not any("common_effect" in family for family in books)
    assert subject.REQUIRED_FAMILIES == (
        "primary",
        "best_residual",
        "destroyed_link",
        "hierarchical_ridge_only",
    )


def test_development_gate_is_donor_labeled_deterministic_and_requires_eight() -> None:
    donors = list(subject.DEVELOPMENT_DONORS)
    primary = np.full(10, 0.8)
    comparator = np.full(10, 1.0)
    first = subject._comparison(donors, primary, comparator, "best_residual")
    second = subject._comparison(donors, primary, comparator, "best_residual")
    assert first == second
    assert first["relative_reduction"] == pytest.approx(0.2)
    assert first["bootstrap_upper_95"] < 0.0
    assert first["favorable_donors"] == 10
    assert first["required_favorable_donors"] == 8
    assert first["passes_all"]
    assert set(first["donor_differences_primary_minus_comparator"]) == set(donors)

    mixed = primary.copy()
    mixed[:3] = 1.1
    assert not subject._comparison(donors, mixed, comparator, "control")[
        "passes_all"
    ]


def test_margin_only_prediction_interfaces_preserve_recipient_margins() -> None:
    truth = _donor_tables(4).reshape(81, 2, 2)
    rows = truth.sum(axis=-1)
    columns = truth.sum(axis=-2)
    coordinate = np.linspace(-0.4, 0.4, 81)
    conditional = subject.predict_conditional_from_margins(coordinate, rows, columns)
    np.testing.assert_allclose(conditional.sum(axis=-1), rows)
    np.testing.assert_allclose(conditional.sum(axis=-2), columns)
    assert np.isfinite(subject.donor_loss(truth, conditional))

    for family in ("pearson", "deviance"):
        for centered in (False, True):
            residual = subject.predict_residual_from_margins(
                coordinate,
                rows,
                columns,
                family=family,
                centered=centered,
            )
            np.testing.assert_allclose(residual.sum(axis=-1), rows, atol=1e-8)
            np.testing.assert_allclose(residual.sum(axis=-2), columns, atol=1e-8)
            assert np.isfinite(subject.donor_loss(truth, residual))


def test_margin_only_prediction_does_not_depend_on_recipient_pairing() -> None:
    first = _donor_tables(2).reshape(81, 2, 2)
    second = first.copy()
    second[:, 0, 0] += 1
    second[:, 0, 1] -= 1
    second[:, 1, 0] -= 1
    second[:, 1, 1] += 1
    np.testing.assert_array_equal(first.sum(axis=-1), second.sum(axis=-1))
    np.testing.assert_array_equal(first.sum(axis=-2), second.sum(axis=-2))
    coordinate = np.linspace(-0.2, 0.2, 81)
    predicted_first = subject.predict_conditional_from_margins(
        coordinate, first.sum(axis=-1), first.sum(axis=-2)
    )
    predicted_second = subject.predict_conditional_from_margins(
        coordinate, second.sum(axis=-1), second.sum(axis=-2)
    )
    np.testing.assert_array_equal(predicted_first, predicted_second)


def test_attempt_precedes_parse_failure_and_refusal_is_terminal(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reduced = tmp_path / "reduced.json"
    reduced.write_text('{"value": NaN}\n')
    acquisition = tmp_path / "development-attempt.json"
    acquisition.write_text('{"status":"TERMINAL_DEVELOPMENT_ATTEMPT_STARTED"}\n')
    monkeypatch.setattr(subject, "INPUT", reduced)
    monkeypatch.setattr(subject, "DEVELOPMENT_ATTEMPT", acquisition)
    monkeypatch.setattr(subject, "EVALUATION_ATTEMPT", tmp_path / "eval-attempt.json")
    monkeypatch.setattr(subject, "EVALUATION_REFUSAL", tmp_path / "refusal.json")
    monkeypatch.setattr(subject, "OUTPUT", tmp_path / "result.json")
    monkeypatch.setattr(subject, "_validate_family_policy", lambda: {})
    monkeypatch.setattr(subject, "_artifact_bindings", lambda: {"binding": "a" * 64})
    with pytest.raises(ValueError, match="nonfinite JSON"):
        subject.run_development(workers=1)
    assert subject.EVALUATION_ATTEMPT.is_file()
    refusal = json.loads(subject.EVALUATION_REFUSAL.read_text())
    assert refusal["status"] == "TERMINAL_DEVELOPMENT_EVALUATION_REFUSAL"
    assert refusal["held_h5ad_members_opened"] == 0
    assert refusal["rerun_permitted"] is False
    assert not subject.OUTPUT.exists()
    with pytest.raises(FileExistsError, match="evaluation artifact"):
        subject.run_development(workers=1)


def test_unavailable_required_family_refuses_but_optional_family_does_not() -> None:
    assert subject._completed_development_status(True, [], None) == "DEVELOPMENT_PASS"
    assert subject._completed_development_status(False, [], None) == "DEVELOPMENT_FAIL"
    with pytest.raises(subject.DevelopmentEvaluationRefusal) as refusal:
        subject._completed_development_status(False, ["primary"], None)
    assert refusal.value.detail["unavailable_required_candidate_families"] == [
        "primary"
    ]
    assert "label_permuted_graph" not in subject.REQUIRED_FAMILIES


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


def test_pass_result_contains_required_losses_gates_and_frozen_model(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    donors = list(subject.DEVELOPMENT_DONORS)
    books = subject._candidate_books(donors)
    for family_index, book in enumerate(books.values()):
        config = book.configs[0]
        loss = 0.8 if family_index == 0 else 1.0 + family_index / 100.0
        for fold in range(10):
            book.record(config, fold, loss, {"certified": True})
    evaluated = {
        "books": books,
        "independence": np.full(10, 1.2),
        "fold_graph_audit": [
            {"omitted_donor": donor, "status": "OK"} for donor in donors
        ],
    }
    input_path = tmp_path / "reduced.json"
    attempt_path = tmp_path / "acquisition.json"
    evaluation_path = tmp_path / "evaluation.json"
    output_path = tmp_path / "result.json"
    for path in (input_path, attempt_path, evaluation_path):
        path.write_text("{}\n")
    monkeypatch.setattr(subject, "INPUT", input_path)
    monkeypatch.setattr(subject, "DEVELOPMENT_ATTEMPT", attempt_path)
    monkeypatch.setattr(subject, "EVALUATION_ATTEMPT", evaluation_path)
    monkeypatch.setattr(subject, "OUTPUT", output_path)
    monkeypatch.setattr(subject, "_cross_validate", lambda data, workers: evaluated)
    monkeypatch.setattr(
        subject,
        "_fit_frozen_models",
        lambda data, selections: {
            "methods": {
                family: {"kind": "test"} for family in subject.REQUIRED_FAMILIES
            }
        },
    )
    data = {
        "donors": donors,
        "source_manifest_sha256": "a" * 64,
    }
    result = subject._run_development_after_attempt(
        data, 1, {"protocol_sha256": "b" * 64}
    )
    assert result["status"] == "DEVELOPMENT_PASS"
    assert set(result["gate"]["comparisons"]) == set(subject.GATE_COMPARATORS)
    assert all(row["passes_all"] for row in result["gate"]["comparisons"].values())
    assert set(subject.REQUIRED_FAMILIES).issubset(
        result["frozen_source_model"]["methods"]
    )
    assert set(result["development_losses"]) == {
        *books,
        "independence",
    }
    assert output_path.is_file()


def test_evaluator_has_no_held_or_raw_h5ad_access_implementation() -> None:
    source = subject.Path(subject.__file__).read_text()
    for forbidden in (
        "h5py",
        "requests",
        "urllib",
        "urlopen",
        "read_h5ad",
        "_matrix_columns",
    ):
        assert forbidden not in source
