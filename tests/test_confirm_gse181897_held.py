from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from experiments import confirm_gse181897_held as subject
from experiments import reduce_gse181897_source as reducer


def _plan(donors: int = 2) -> reducer.SourcePlan:
    selected = np.arange(donors * subject.CELL_BUDGET, dtype=np.int64).reshape(
        donors, subject.CELL_BUDGET
    )
    authorized = np.zeros(400, dtype=bool)
    authorized[selected.ravel()] = True
    batches = tuple(8 if index == 0 else 9 for index in range(donors))
    return reducer.SourcePlan(
        donor_axis=tuple(str(index) for index in range(donors)),
        free_id_axis=tuple(str(index + 100) for index in range(donors)),
        batch_axis=batches,
        selected_rows=selected,
        selected_barcodes=np.asarray(
            [[f"cell-{row}" for row in rows] for rows in selected], dtype=str
        ),
        authorized_rows=authorized,
        donor_audit=tuple(
            {
                "selected_cell_axis_sha256": subject._axis_sha256(
                    [f"cell-{row}" for row in rows]
                )
            }
            for rows in selected
        ),
    )


def test_internal_and_confirmation_axes_are_exact_and_disjoint() -> None:
    assert subject._expected_donor_axis("internal") == (
        "3",
        "16",
        "19",
        "48",
        "50",
        "0",
        "2",
        "17",
        "33",
    )
    assert subject._expected_donor_axis("confirmation") == (
        "8",
        "21",
        "28",
        "41",
        "53",
        "56",
        "6",
        "20",
        "26",
        "40",
        "46",
        "54",
    )
    assert set(subject._expected_donor_axis("internal")).isdisjoint(
        subject._expected_donor_axis("confirmation")
    )
    assert subject._expected_batch_axis("internal") == (8,) * 5 + (9,) * 4
    assert subject._expected_batch_axis("confirmation") == (10,) * 6 + (11,) * 6


def test_csr_firewall_scans_indices_but_never_decodes_out_of_panel_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan()
    row_count = 400
    indptr = np.zeros(row_count + 1, dtype=np.int32)
    indices: list[int] = []
    data: list[float] = []
    for row in range(row_count):
        if row < 256:
            indices.extend((2, 99))
            data.extend((1.0, np.nan))
        elif row == 300:
            indices.append(2)
            data.append(np.nan)
        indptr[row + 1] = len(indices)
    path = tmp_path / "tiny.h5"
    with h5py.File(path, "w") as handle:
        matrix = handle.create_group("X")
        matrix.create_dataset("data", data=np.asarray(data, dtype=np.float32))
        matrix.create_dataset("indices", data=np.asarray(indices, dtype=np.int32))
        matrix.create_dataset("indptr", data=indptr)
    monkeypatch.setattr(reducer, "_matrix_metadata", lambda matrix: {})
    monkeypatch.setattr(reducer, "EXPECTED_X_SHAPE", (row_count, 20_399))
    monkeypatch.setattr(reducer, "EXPECTED_X_DATA_LENGTH", len(indices))
    with h5py.File(path, "r") as handle:
        states, audit = subject._read_authorized_csr_states(
            handle["X"], plan, (2,), "internal"
        )
    assert states.shape == (256, 1)
    assert states.all()
    assert audit["csr_index_entries_scanned"] == 512
    assert audit["requested_X_data_entries_decoded"] == 256
    assert audit["out_of_panel_index_positions_scanned"] == 256
    assert audit["out_of_panel_X_data_entries_decoded"] == 0
    assert audit["unauthorized_held_batch_rows_scanned"] == 0


def test_margin_reader_keeps_modalities_separate_and_retains_no_binary_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    first = np.ones((256, 17), dtype=bool)
    second = np.zeros((256, 17), dtype=bool)
    calls: list[tuple[int, ...]] = []

    def fake_read(matrix, observed_plan, columns, stage):
        calls.append(columns)
        return (first if len(calls) == 1 else second), {
            "csr_index_entries_scanned": 10,
            "requested_X_data_entries_decoded": 5,
        }

    monkeypatch.setattr(subject, "_read_authorized_csr_states", fake_read)
    rna, adt, audit = subject._read_margin_counts(
        object(), plan, tuple(range(17)), tuple(range(17, 34)), "internal"
    )
    assert calls == [tuple(range(17)), tuple(range(17, 34))]
    assert np.array_equal(rna, np.full((2, 17), 128))
    assert np.array_equal(adt, np.zeros((2, 17)))
    assert not first.any() and not second.any()
    assert audit["same_cell_joint_tables_constructed"] == 0
    assert audit["same_cell_binary_rows_retained"] == 0


def _selection_artifact() -> dict:
    base = {
        "heterogeneity_penalty": 0.1,
        "ridge_penalty": 0.01,
        "transport_multiplier": 1.0,
    }
    primary = {
        "graph_neighbors": 2,
        **base,
        "graph_penalty": 0.01,
    }
    residuals = {"pearson": 1.0, "root_deviance": 0.75}
    models = {
        "primary": {"configuration": primary},
        "matched_graph_zero": {"configuration": {**primary, "graph_penalty": 0.0}},
        "common_effect_cmle": {"configuration": {"transport_multiplier": 1.0}},
        "pooled_saturated_poisson": {"configuration": {"transport_multiplier": 0.75}},
        "pearson_residual": {
            "configuration": {"family": "pearson", "transport_multiplier": 1.0}
        },
        "root_deviance_residual": {
            "configuration": {
                "family": "root_deviance",
                "transport_multiplier": 0.75,
            }
        },
        "destroyed_link": {"configuration": {**primary, "transport_multiplier": 1.25}},
    }
    return {
        "development": {
            "source_go_gate": {
                "passes": True,
                "checks": {key: True for key in subject.SOURCE_GATE_CHECKS},
            },
            "final_all_source_selection": {
                "stage_a_graph_zero": {"selected_configuration": base},
                "stage_b_nonzero_hypergraph": {
                    "fixed_from_stage_a": base,
                    "selected_configuration": primary,
                },
                "comparators": {
                    "selected_common_transport": 1.0,
                    "selected_poisson_transport": 0.75,
                    "selected_residual": {
                        "family": "pearson",
                        "transport_multiplier": 1.0,
                    },
                    "selected_residual_transports": residuals,
                    "selected_destroyed_transport": 1.25,
                },
            },
        },
        "candidate": {"canonical_configuration": primary, "models": models},
    }


def test_source_candidate_configuration_is_deeply_bound() -> None:
    artifact = _selection_artifact()
    subject._validate_source_selection_contract(artifact)
    drift = copy.deepcopy(artifact)
    drift["candidate"]["models"]["matched_graph_zero"]["configuration"][
        "transport_multiplier"
    ] = 1.25
    with pytest.raises(PermissionError, match="graph-zero"):
        subject._validate_source_selection_contract(drift)
    failed = copy.deepcopy(artifact)
    failed["development"]["source_go_gate"]["checks"][
        "comparison_mask_floor_passes"
    ] = False
    with pytest.raises(PermissionError, match="statistical or support"):
        subject._validate_source_selection_contract(failed)
    incomplete = copy.deepcopy(artifact)
    incomplete["development"]["source_go_gate"]["checks"].pop(
        "all_mandatory_estimators_complete"
    )
    with pytest.raises(PermissionError, match="statistical or support"):
        subject._validate_source_selection_contract(incomplete)


def test_v1_or_nonunique_axis_preflight_cannot_authorize_numeric_access() -> None:
    observed = SimpleNamespace(payload={})
    with pytest.raises(PermissionError, match="frozen unread"):
        subject._validate_axis_preflight(
            {"schema": "gse181897-axis-preflight/1.0"}, observed
        )
    invalid = {
        "schema": "gse181897-axis-preflight/1.1",
        "status": "AXES_FROZEN_UNIQUE_X_NUMERIC_UNREAD",
        "numeric_access": {"decoded_X_entries": 0, "matrix_datasets_indexed": []},
        "hdf5": {
            "obs": {"unique_rows": 136_142, "index_is_unique": True},
            "var": {"unique_rows": 20_399, "index_is_unique": False},
        },
    }
    with pytest.raises(PermissionError, match="uniqueness"):
        subject._validate_axis_preflight(invalid, observed)


def test_authorization_chain_uses_exact_tags_ancestry_and_public_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authorization = tmp_path / "authorization.json"
    authorization.write_text("{}\n")
    action = "internal_score"
    expected_tags = subject.BASE_FREEZE_TAGS + subject.UPSTREAM_TAGS[action]
    records = [
        {
            "tag": tag,
            "annotated_tag_object": f"{index + 1:040x}",
            "peeled_commit": f"{index + 101:040x}",
        }
        for index, tag in enumerate(expected_tags)
    ]
    action_record = {
        "tag": subject.ACTION_TAGS[action],
        "annotated_tag_object": "f" * 40,
        "peeled_commit": "e" * 40,
    }
    observed_paths: list[tuple[str, Path]] = []
    ancestry: list[tuple[str, str]] = []
    snapshots = {record["tag"]: record for record in records}
    snapshots[action_record["tag"]] = action_record
    monkeypatch.setattr(subject, "_tag_snapshot", lambda tag: snapshots[tag])
    monkeypatch.setattr(subject, "_tagged_paths", lambda action: {})
    monkeypatch.setattr(
        subject,
        "_published_path_matches",
        lambda tag, path: observed_paths.append((tag, path)),
    )
    monkeypatch.setattr(
        subject,
        "_require_ancestor",
        lambda first, second: ancestry.append((first, second)),
    )
    result = subject._verify_public_authorization(
        action, authorization, {"verified_freeze_chain": records}
    )
    assert result["tag"] == subject.ACTION_TAGS[action]
    assert observed_paths == [(subject.ACTION_TAGS[action], authorization)]
    assert len(ancestry) == len(expected_tags)
    broken = copy.deepcopy(records)
    broken[-1]["peeled_commit"] = "0" * 40
    with pytest.raises(PermissionError, match="wrong"):
        subject._verify_public_authorization(
            action, authorization, {"verified_freeze_chain": broken}
        )


def test_attempt_path_is_exclusive_and_exact_authorization_path_is_canonical(
    tmp_path: Path,
) -> None:
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_text("{}\n")
    attempt_path = tmp_path / "attempt.json"
    subject._claim_attempt(
        attempt_path,
        "internal_prepare",
        authorization_path,
        {},
        {"tag": "t", "tag_object": "o", "commit": "c"},
    )
    with pytest.raises(FileExistsError):
        subject._claim_attempt(
            attempt_path,
            "internal_prepare",
            authorization_path,
            {},
            {"tag": "t", "tag_object": "o", "commit": "c"},
        )
    with pytest.raises(PermissionError, match="not canonical"):
        subject._validate_authorization(
            authorization_path,
            "internal_prepare",
            subject._action_paths("internal_prepare"),
        )


def test_exact_gate_thresholds_and_zero_tie_rule() -> None:
    losses = np.ones((9, len(subject.METHOD_AXIS)), dtype=float)
    index = {name: position for position, name in enumerate(subject.METHOD_AXIS)}
    losses[:, index["primary"]] = 0.8
    losses[:, index["matched_graph_zero"]] = 1.0
    losses[:, index["common_effect_cmle"]] = 0.95
    losses[:, index["pooled_saturated_poisson"]] = 0.94
    losses[:, index["primary_classical_residual"]] = 0.93
    losses[:, index["destroyed_link"]] = 0.92
    gate = subject._held_gate("internal", losses, np.asarray((8,) * 5 + (9,) * 4))
    assert gate["passes"] is True
    assert gate["topology"]["empirical_p"] == 1 / 64
    tied = subject._exact_sign_test(np.asarray((-1.0, -1.0, 0.0, 1.0)))
    assert tied["nonzero_donors"] == 3
    assert tied["exact_ties"] == 1
    assert tied["favorable_donors"] == 2
    assert tied["one_sided_p"] == 0.5


def test_h5ad_post_read_hash_detects_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "input.h5ad"
    path.write_bytes(b"frozen")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(reducer, "SOURCE_H5AD_BYTES", 6)
    monkeypatch.setattr(reducer, "SOURCE_H5AD_SHA256", expected)
    assert subject._verify_h5ad_after_numeric_read(path) == expected
    path.write_bytes(b"changed")
    with pytest.raises(PermissionError, match="changed during numeric access"):
        subject._verify_h5ad_after_numeric_read(path)


def test_score_interruption_writes_terminal_refusal_and_blocks_second_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = {
        key: tmp_path / f"{key}.dat" for key in subject._action_paths("internal_score")
    }
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_text("{}\n")
    paths["attempt"] = tmp_path / "attempt.json"
    paths["score_terminal"] = tmp_path / "score-terminal.json"
    paths["result"] = tmp_path / "result.json"
    plan = SimpleNamespace(donor_axis=(), batch_axis=())
    monkeypatch.setattr(subject, "_require_runtime", lambda: None)
    monkeypatch.setattr(subject, "_action_paths", lambda action: paths)
    monkeypatch.setattr(
        subject, "_validate_authorization", lambda *args: ({}, {"tag": "freeze"})
    )
    monkeypatch.setattr(subject, "_load_bound_candidate", lambda path: object())
    monkeypatch.setattr(
        subject,
        "_load_frozen_prepare_artifacts",
        lambda *args: ({}, {}, {"selection": {}}),
    )
    monkeypatch.setattr(subject, "_inspect_held_axes", lambda *args: (object(), plan))
    monkeypatch.setattr(subject, "_validate_prepare_axes", lambda *args: None)
    monkeypatch.setattr(subject, "_selection_certificate", lambda plan: {})
    monkeypatch.setattr(subject, "_implementation_snapshot", lambda: {})
    monkeypatch.setattr(subject, "_execution_snapshot", lambda *args: {})
    monkeypatch.setattr(
        subject,
        "_score_claimed",
        lambda *args: (_ for _ in ()).throw(RuntimeError("stop")),
    )
    with pytest.raises(RuntimeError, match="stop"):
        subject.score("internal", authorization_path)
    terminal = json.loads(paths["score_terminal"].read_text())
    assert terminal["status"] == "TERMINAL_HELD_SCORE_REFUSAL"
    assert terminal["rerun_forbidden"] is True
    assert paths["attempt"].is_file()
    with pytest.raises(FileExistsError, match="terminal"):
        subject.score("internal", authorization_path)


def test_confirmation_requires_published_complete_internal_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = tmp_path / "internal-result.json"
    result.write_text(
        json.dumps(
            {
                "schema": "gse181897-held-score/1.0",
                "stage": "internal",
                "status": "INTERNAL_VALIDATION_FAILED",
                "gate": {"passes": False},
            }
        )
    )
    terminal = tmp_path / "internal-terminal.json"
    terminal.write_text("{}\n")
    original = subject._stage_paths

    def paths(stage: str):
        values = original(stage)
        if stage == "internal":
            values["score_terminal"] = terminal
        return values

    monkeypatch.setattr(subject, "_stage_paths", paths)
    with pytest.raises(PermissionError, match="did not pass"):
        subject._validate_internal_pass(result)


def test_authorization_templates_are_closed_and_freeze_exact_lineage() -> None:
    directory = subject.CONFIRMATION_DIRECTORY
    expected = {
        "internal_prepare": subject.BASE_FREEZE_TAGS,
        "internal_score": subject.BASE_FREEZE_TAGS
        + subject.UPSTREAM_TAGS["internal_score"],
        "confirmation_prepare": subject.BASE_FREEZE_TAGS
        + subject.UPSTREAM_TAGS["confirmation_prepare"],
        "confirmation_score": subject.BASE_FREEZE_TAGS
        + subject.UPSTREAM_TAGS["confirmation_score"],
    }
    for action, tags in expected.items():
        template = json.loads(
            (directory / f"{action}_authorization_template_v1.json").read_text()
        )
        assert template["stage"] == action
        assert template["status"].startswith("TEMPLATE_CLOSED")
        assert subject._contains_pending(template)
        assert (
            tuple(record["tag"] for record in template["verified_freeze_chain"]) == tags
        )
