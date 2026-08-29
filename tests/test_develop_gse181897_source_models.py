from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import develop_gse181897_source_models as subject
from experiments import reduce_gse181897_source as reducer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _source_arrays() -> dict[str, np.ndarray]:
    generator = np.random.default_rng(181897)
    rna = generator.binomial(
        1,
        0.5,
        size=(
            subject.EXPECTED_DONOR_COUNT,
            subject.CELL_BUDGET,
            subject.MARKER_COUNT,
        ),
    ).astype(np.int32)
    adt = generator.binomial(1, 0.5, size=rna.shape).astype(np.int32)
    tables = np.asarray(
        [subject._tables(rna[index], adt[index]) for index in range(len(rna))],
        dtype=np.int16,
    )
    denominator = [item.adt_feature for item in subject.PANEL]
    denominator.extend(f"control-{index}" for index in range(96 - len(denominator)))
    coordinate_axis = [
        f"{rna_item.rna_gene}|{adt_item.adt_feature}"
        for rna_item in subject.PANEL
        for adt_item in subject.PANEL
    ]
    arrays = {
        "donor_axis": np.asarray(subject.EXPECTED_DONORS),
        "free_id_axis": np.asarray(
            [f"free-{index}" for index in range(subject.EXPECTED_DONOR_COUNT)]
        ),
        "batch_axis": np.asarray(subject.EXPECTED_BATCH_AXIS, dtype=np.int8),
        "condition_axis": np.asarray(
            [subject.CONTROL_CONDITION] * subject.EXPECTED_DONOR_COUNT
        ),
        "rna_gene_axis": np.asarray([item.rna_gene for item in subject.PANEL]),
        "rna_feature_id_axis": np.asarray(
            [item.rna_feature_id for item in subject.PANEL]
        ),
        "adt_protein_axis": np.asarray([item.protein_label for item in subject.PANEL]),
        "adt_feature_axis": np.asarray([item.adt_feature for item in subject.PANEL]),
        "protein_denominator_axis": np.asarray(denominator),
        "coordinate_axis": np.asarray(coordinate_axis),
        "source_comparison_mask": np.ones(subject.COORDINATE_COUNT, dtype=np.uint8),
        "selected_barcodes": np.asarray(
            [
                [f"cell-{donor}-{cell}" for cell in range(subject.CELL_BUDGET)]
                for donor in range(subject.EXPECTED_DONOR_COUNT)
            ]
        ),
        "selected_row_indices": np.arange(
            subject.EXPECTED_DONOR_COUNT * subject.CELL_BUDGET, dtype=np.int64
        ).reshape(subject.EXPECTED_DONOR_COUNT, subject.CELL_BUDGET),
        "rna_counts": rna,
        "adt_counts": adt,
        "adt_graph_profile": generator.normal(
            size=(subject.EXPECTED_DONOR_COUNT, subject.MARKER_COUNT)
        ).astype(np.float64),
        "tables": tables.reshape(subject.EXPECTED_DONOR_COUNT, -1, 4),
    }
    records = {
        donor: {
            "tables": tables[index].astype(np.int64),
            "subject_support": np.ones(
                (subject.MARKER_COUNT, subject.MARKER_COUNT), dtype=bool
            ),
        }
        for index, donor in enumerate(subject.EXPECTED_DONORS)
    }
    mask, _ = subject._training_mask(records, list(subject.EXPECTED_DONORS))
    arrays["source_comparison_mask"] = mask.astype(np.uint8).ravel()
    return arrays


def _write_source_fixture(
    directory: Path,
    monkeypatch: pytest.MonkeyPatch,
    arrays: dict[str, np.ndarray] | None = None,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    values = _source_arrays() if arrays is None else arrays
    source_path = directory / "source.npz"
    preflight = directory / "axis_preflight_v2.json"
    preflight.write_text(
        json.dumps(
            {
                "schema": "gse181897-axis-preflight/1.1",
                "status": "AXES_FROZEN_UNIQUE_X_NUMERIC_UNREAD",
                "reducer_sha256": "a" * 64,
            }
        )
        + "\n"
    )
    authorization_path = directory / "authorization.json"
    attempt_path = directory / "attempt.json"
    reduction_terminal_path = directory / "reduction_terminal.json"
    model_output_path = directory / "model.json"
    model_terminal_path = directory / "model_terminal.json"
    monkeypatch.setattr(reducer, "DEFAULT_PREFLIGHT", preflight)
    monkeypatch.setattr(reducer, "DEFAULT_SOURCE_AUTHORIZATION", authorization_path)
    monkeypatch.setattr(reducer, "DEFAULT_SOURCE_ATTEMPT", attempt_path)
    implementation_hashes = {
        "fixture": "b" * 64,
        "experiments/reduce_gse181897_source.py": "a" * 64,
    }
    monkeypatch.setattr(
        reducer,
        "_campaign_implementation_hashes",
        lambda: implementation_hashes,
    )
    monkeypatch.setattr(reducer, "_validate_public_freeze_chain", lambda *_args: None)
    freeze = {
        "tag_object": "1" * 40,
        "peeled_commit": "2" * 40,
        "remote_tag_and_commit_match": True,
    }
    authorization = {
        "schema": reducer.SOURCE_AUTHORIZATION_SCHEMA,
        "status": reducer.SOURCE_AUTHORIZATION_STATUS,
        "accession": "GSE181897",
        "stage": "source_development",
        "candidate_freeze": {"tag": reducer.CANDIDATE_TAG, **freeze},
        "implementation_freeze": {
            "tag": reducer.IMPLEMENTATION_TAG,
            **freeze,
            "files_sha256": implementation_hashes,
        },
        "axis_freeze": {
            "tag": reducer.AXIS_PREFLIGHT_TAG,
            **freeze,
            "preflight_path": reducer._display_path(preflight),
            "preflight_sha256": _sha256(preflight),
            "axis_reducer_sha256": "a" * 64,
        },
        "source_input": {
            "archive_bytes": reducer.SOURCE_ARCHIVE_BYTES,
            "archive_sha256": reducer.SOURCE_ARCHIVE_SHA256,
            "h5ad_bytes": reducer.SOURCE_H5AD_BYTES,
            "h5ad_sha256": reducer.SOURCE_H5AD_SHA256,
        },
        "attempt_path": reducer._display_path(attempt_path),
        "outputs": {
            "source_reduction": reducer._display_path(source_path),
            "source_reduction_manifest": reducer._display_path(
                directory / "manifest.json"
            ),
            "source_model": reducer._display_path(model_output_path),
            "source_model_terminal": reducer._display_path(model_terminal_path),
            "source_reduction_terminal": reducer._display_path(reduction_terminal_path),
        },
        "runtime": reducer._campaign_runtime(),
        "one_shot_policy": {
            "exclusive_attempt_before_numeric_x": True,
            "rerun_after_claim_forbidden": True,
            "failure_is_terminal": True,
            "model_must_bind_same_attempt": True,
        },
    }
    authorization_path.write_text(json.dumps(authorization) + "\n")
    reducer.claim_source_campaign(
        authorization_path,
        attempt_path,
        preflight,
        source_path,
        directory / "manifest.json",
        model_output_path,
        model_terminal_path,
        reduction_terminal_path,
    )
    np.savez_compressed(source_path, **values)
    stored_mask = np.asarray(values["source_comparison_mask"], dtype=np.uint8).reshape(
        subject.MARKER_COUNT, subject.MARKER_COUNT
    )
    manifest = {
        "schema": "gse181897-source-reduction/1.0",
        "status": "SOURCE_REDUCTION_COMPLETE",
        "accession": "GSE181897",
        "stage": "source_development",
        "numeric_batches_processed": list(subject.SOURCE_BATCHES),
        "numeric_condition_processed": subject.CONTROL_CONDITION,
        "held_batches_unopened": list(subject.FORBIDDEN_BATCHES),
        "non_control_rows_unopened": True,
        "condition_0_mispool_rows_unopened": 455,
        "excluded_development_donors": {"23": 6, "62": 2},
        "donor_count": subject.EXPECTED_DONOR_COUNT,
        "cell_budget_per_donor": subject.CELL_BUDGET,
        "panel": [
            {
                **item,
                "rna_column_zero_based": index,
                "adt_column_zero_based": 100 + index,
            }
            for index, item in enumerate(subject._expected_panel())
        ],
        "adt_graph_profile": {
            "denominator_feature_count": 96,
            "denominator_rule": "all 96 var/genome=BD99AbSeq features",
            "denominator_feature_axis": [
                str(value) for value in values["protein_denominator_axis"]
            ],
            "cognate_feature_axis": [item.adt_feature for item in subject.PANEL],
        },
        "axis_preflight": {
            "path": str(preflight),
            "sha256": _sha256(preflight),
            "schema": "gse181897-axis-preflight/1.1",
            "status": "AXES_FROZEN_UNIQUE_X_NUMERIC_UNREAD",
        },
        "source_campaign_attempt": {
            "path": reducer._display_path(attempt_path),
            "sha256": _sha256(attempt_path),
            "schema": reducer.SOURCE_ATTEMPT_SCHEMA,
            "status": reducer.SOURCE_ATTEMPT_STATUS,
        },
        "numeric_access": {
            "numeric_rows_decoded": (
                subject.EXPECTED_DONOR_COUNT * subject.CELL_BUDGET
            ),
            "matrix_datasets_indexed": ["/X/indptr", "/X/indices", "/X/data"],
            "numeric_feature_columns_decoded": 113,
            "stored_numeric_values_decoded": 1000,
            "csr_index_entries_scanned": 2000,
            "requested_stored_data_entries_decoded": 1000,
            "unrequested_stored_data_entries_decoded": 0,
            "out_of_panel_index_positions_scanned": 1000,
            "out_of_panel_indices_used_only_for_membership_filtering": True,
            "out_of_panel_featurewise_statistics_retained": 0,
            "out_of_panel_feature_signal_entering_model_outputs": 0,
            "held_batch_rows_decoded": 0,
            "non_control_rows_decoded": 0,
            "unselected_authorized_rows_decoded": 0,
        },
        "availability_diagnostics": {
            "status": "SOURCE_JOINT_TABLES_REDUCED_NO_MODEL_OR_LOSS_INSPECTED",
            "final_source_mask": {
                "mask_sha256": subject._array_sha256(stored_mask),
                "mask_coordinates": int(np.count_nonzero(stored_mask)),
                "mask_at_least_232_coordinates": True,
                "every_source_donor_at_least_232_coordinates": True,
            },
        },
        "output": {
            "path": str(source_path),
            "bytes": source_path.stat().st_size,
            "sha256": _sha256(source_path),
            "members": sorted(values),
        },
        "reducer_sha256": _sha256(
            subject.ROOT / "experiments/reduce_gse181897_source.py"
        ),
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest) + "\n")
    reduction_terminal_path.write_text(
        json.dumps(
            {
                "schema": "gse181897-source-reduction-terminal/1.0",
                "status": "SOURCE_REDUCTION_COMPLETE_MODEL_PENDING",
                "attempt_path": reducer._display_path(attempt_path),
                "attempt_sha256": _sha256(attempt_path),
                "source_reduction_path": reducer._display_path(source_path),
                "source_reduction_sha256": _sha256(source_path),
                "source_manifest_path": reducer._display_path(manifest_path),
                "source_manifest_sha256": _sha256(manifest_path),
                "model_output_path": reducer._display_path(model_output_path),
                "model_terminal_path": reducer._display_path(model_terminal_path),
                "model_numeric_access_authorized": False,
            }
        )
        + "\n"
    )
    return (
        source_path,
        manifest_path,
        attempt_path,
        reduction_terminal_path,
        model_output_path,
        model_terminal_path,
    )


def test_load_source_accepts_exact_source_only_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path, manifest_path, attempt, terminal, model, model_terminal = (
        _write_source_fixture(tmp_path, monkeypatch)
    )
    loaded = subject._load_source(
        source_path, manifest_path, attempt, model, model_terminal, terminal
    )
    assert loaded.donors == list(subject.EXPECTED_DONORS)
    assert loaded.batches == list(subject.EXPECTED_BATCH_AXIS)
    assert loaded.tables.shape == (39, 17, 17, 2, 2)
    assert set(loaded.batches).isdisjoint(subject.FORBIDDEN_BATCHES)


@pytest.mark.parametrize("failure", ["held_batch", "noncontrol", "extra_member"])
def test_load_source_rejects_firewall_or_schema_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    arrays = _source_arrays()
    if failure == "held_batch":
        arrays["batch_axis"] = arrays["batch_axis"].copy()
        arrays["batch_axis"][0] = 8
    elif failure == "noncontrol":
        arrays["condition_axis"] = arrays["condition_axis"].copy()
        arrays["condition_axis"][0] = "R"
    else:
        arrays["forbidden_held_values"] = np.asarray([1])
    source_path, manifest_path, attempt, terminal, model, model_terminal = (
        _write_source_fixture(tmp_path, monkeypatch, arrays)
    )
    with pytest.raises((PermissionError, ValueError), match="batch|condition|member"):
        subject._load_source(
            source_path, manifest_path, attempt, model, model_terminal, terminal
        )


def test_load_source_rejects_a_joint_table_not_linked_to_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arrays = _source_arrays()
    arrays["tables"] = arrays["tables"].copy()
    arrays["tables"][0, 0, 0] += 1
    source_path, manifest_path, attempt, terminal, model, model_terminal = (
        _write_source_fixture(tmp_path, monkeypatch, arrays)
    )
    with pytest.raises(ValueError, match="stored source table"):
        subject._load_source(
            source_path, manifest_path, attempt, model, model_terminal, terminal
        )


def _full_support_records() -> dict[str, dict[str, object]]:
    arrays = _source_arrays()
    tables = arrays["tables"].reshape(39, 17, 17, 2, 2).astype(np.int64)
    records = {}
    for index, donor in enumerate(subject.EXPECTED_DONORS):
        records[donor] = {
            "batch": subject.EXPECTED_BATCH_AXIS[index],
            "tables": tables[index],
            "destroyed_tables": tables[index].copy(),
            "subject_support": np.ones((17, 17), dtype=bool),
            "rna_profile": np.asarray(
                arrays["rna_counts"][index].mean(axis=0), dtype=float
            ),
            "adt_profile": arrays["adt_graph_profile"][index],
        }
    return records


def test_training_mask_enforces_support_interior_and_four_cell_positivity() -> None:
    records = _full_support_records()
    mask, certificate = subject._training_mask(records, list(subject.EXPECTED_DONORS))
    assert mask.shape == (17, 17)
    assert np.count_nonzero(mask) == 289
    assert all(certificate["checks"].values())
    assert certificate["strict_pooled_fixed_margin_interior"] is True


def test_training_mask_rejects_boundary_even_when_pooled_cells_are_positive() -> None:
    donors = list(subject.EXPECTED_DONORS)
    first = np.asarray([[0, 20], [30, 78]], dtype=np.int64)
    second = np.asarray([[32, 68], [28, 0]], dtype=np.int64)
    records = {
        donor: {
            "tables": np.broadcast_to(
                first if index % 2 == 0 else second,
                (17, 17, 2, 2),
            ).copy(),
            "subject_support": np.ones((17, 17), dtype=bool),
        }
        for index, donor in enumerate(donors)
    }
    pooled = sum(record["tables"] for record in records.values())
    assert np.all(pooled > 0)
    with pytest.raises(subject.SourceGoRefusal, match="comparison mask"):
        subject._training_mask(records, donors)


def test_within_pool_normalization_refuses_zero_scale() -> None:
    profiles = np.ones((39, 17))
    with pytest.raises(subject.SourceGoRefusal, match="zero or nonfinite"):
        subject._within_batch_normalize(profiles, list(subject.EXPECTED_BATCH_AXIS))


def test_destroyed_link_is_exact_deterministic_cyclic_row_shift() -> None:
    generator = np.random.default_rng(44)
    states = generator.binomial(1, 0.4, size=(128, 17)).astype(np.uint8)
    barcodes = [f"cell-{index:03d}" for index in range(128)]
    donor = "34"
    order = sorted(
        range(128),
        key=lambda index: (
            hashlib.sha256(
                f"{subject.DESTROYED_LINK_SALT}|{donor}|{barcodes[index]}".encode()
            ).hexdigest(),
            barcodes[index],
        ),
    )
    order_array = np.asarray(order)
    assert np.all(order_array != np.roll(order_array, 1))
    expected = np.empty_like(states)
    expected[order_array] = states[np.roll(order_array, 1)]
    first = subject._destroyed_adt(states, barcodes, donor)
    second = subject._destroyed_adt(states, barcodes, donor)
    np.testing.assert_array_equal(first, expected)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first.sum(axis=0), states.sum(axis=0))
    assert sorted(map(tuple, first.tolist())) == sorted(map(tuple, states.tolist()))


def test_equal_distance_hypergraph_ties_use_lower_marker_index() -> None:
    incidence = subject._marker_hyperedges(np.zeros((5, 17)), 2)
    memberships = {
        tuple(np.flatnonzero(incidence[:, column]).tolist())
        for column in range(incidence.shape[1])
    }
    assert (0, 1, 2) in memberships
    assert (0, 1, 16) in memberships


def test_training_design_is_unchanged_when_nontraining_values_change() -> None:
    records = _full_support_records()
    training = [
        donor for donor in subject.EXPECTED_DONORS if records[donor]["batch"] != 0
    ]
    first = subject._training_design(records, training, 2)
    held = [donor for donor in subject.EXPECTED_DONORS if donor not in training]
    for donor in held:
        records[donor]["rna_profile"] = np.full(17, 1e9)
        records[donor]["adt_profile"] = np.full(17, -1e9)
        records[donor]["tables"] = np.zeros((17, 17, 2, 2), dtype=int)
        records[donor]["subject_support"] = np.zeros((17, 17), dtype=bool)
    second = subject._training_design(records, training, 2)
    assert first["mask_record"]["mask_sha256"] == second["mask_record"]["mask_sha256"]
    assert first["rna_incidence_sha256"] == second["rna_incidence_sha256"]
    assert first["adt_incidence_sha256"] == second["adt_incidence_sha256"]


def test_nested_axes_never_include_outer_pool_in_inner_selection() -> None:
    records = _full_support_records()
    donors = list(subject.EXPECTED_DONORS)
    for outer_batch, outer_training, outer_validation in subject._batch_splits(
        records, donors
    ):
        assert {records[donor]["batch"] for donor in outer_validation} == {outer_batch}
        assert all(records[donor]["batch"] != outer_batch for donor in outer_training)
        for _, inner_training, inner_validation in subject._batch_splits(
            records, outer_training
        ):
            assert all(
                records[donor]["batch"] != outer_batch
                for donor in inner_training + inner_validation
            )


def test_stage_a_completes_without_constructing_a_profile_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = _full_support_records()
    donors = ["34", "35", "18", "22"]
    graph_zero_fits = []

    monkeypatch.setattr(
        subject,
        "_training_mask",
        lambda _records, training: (
            np.ones((17, 17), dtype=bool),
            {"training_donors": training},
        ),
    )

    def fit_graph_zero(*_args: object, **_kwargs: object) -> dict[str, object]:
        graph_zero_fits.append(True)
        return {"population_log_odds": np.zeros((17, 17))}

    monkeypatch.setattr(subject, "_fit_structured", fit_graph_zero)
    monkeypatch.setattr(
        subject,
        "_fit_common",
        lambda *_args: {"population_log_odds": np.zeros((17, 17))},
    )
    monkeypatch.setattr(
        subject,
        "_fit_pooled_poisson",
        lambda *_args: {"population_log_odds": np.zeros((17, 17))},
    )
    monkeypatch.setattr(
        subject,
        "_fit_residual",
        lambda *_args: {"pooled_coordinate": np.zeros((17, 17))},
    )
    monkeypatch.setattr(subject, "_population_loss", lambda *_args: 1.0)
    monkeypatch.setattr(subject, "_residual_loss", lambda *_args: 1.0)
    monkeypatch.setattr(subject, "_independence_loss", lambda *_args: 1.0)

    class GraphConstructionReached(RuntimeError):
        pass

    def reject_graph(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise GraphConstructionReached

    monkeypatch.setattr(subject, "_training_design", reject_graph)
    with pytest.raises(GraphConstructionReached):
        subject._select_models(records, donors)
    assert len(graph_zero_fits) == 2 * len(subject.HETEROGENEITY_GRID) * len(
        subject.RIDGE_GRID
    )


def test_selection_tie_breaks_are_deterministic_and_favor_weaker_graph() -> None:
    donors = ["a", "b", "c", "d"]
    batches = [0, 0, 1, 1]
    values = np.ones(4)
    base_a = subject.BaseConfig(0.1, 0.01, 0.5)
    base_b = subject.BaseConfig(1.0, 0.01, 0.5)
    assert (
        subject._select_base(
            {base_b: values.copy(), base_a: values.copy()}, donors, batches
        )
        == base_a
    )
    structured_a = subject.StructuredConfig(3, 0.1, 0.01, 0.01, 0.5)
    structured_b = subject.StructuredConfig(2, 0.1, 0.01, 0.03, 0.5)
    assert (
        subject._select_structured(
            {structured_b: values.copy(), structured_a: values.copy()}, donors, batches
        )
        == structured_a
    )
    residual_losses = {
        subject.ResidualConfig("pearson", 1.0): values.copy(),
        subject.ResidualConfig("root_deviance", 1.0): values.copy(),
    }
    assert (
        subject._select_residual(
            residual_losses,
            {"pearson": 1.0, "root_deviance": 1.0},
            donors,
            batches,
        ).family
        == "pearson"
    )


def test_pooled_poisson_matches_saturated_loglinear_interaction() -> None:
    tables = np.zeros((3, 17, 17, 2, 2), dtype=np.int64)
    support = np.zeros((3, 17, 17), dtype=bool)
    component = np.asarray(
        [
            [[40, 24], [20, 44]],
            [[36, 20], [28, 44]],
            [[32, 28], [24, 44]],
        ],
        dtype=np.int64,
    )
    tables[:, 0, 0] = component
    support[:, 0, 0] = True
    mask = np.zeros((17, 17), dtype=bool)
    mask[0, 0] = True
    fit = subject._fit_pooled_poisson(tables, support, mask)
    pooled = component.sum(axis=0).astype(float)
    design = np.asarray(
        [[1.0, row, column, row * column] for row in (0, 1) for column in (0, 1)]
    )
    beta = np.linalg.solve(design, np.log(pooled.ravel()))
    expected_log_odds = math.log(
        pooled[0, 0] * pooled[1, 1] / pooled[0, 1] / pooled[1, 0]
    )
    assert beta[3] == pytest.approx(expected_log_odds, abs=1e-12)
    assert fit["population_log_odds"][0, 0] == pytest.approx(
        expected_log_odds, abs=1e-12
    )
    assert fit["fit_certificate"]["maximum_normalized_reconstruction_error"] <= 1e-8
    assert fit["fit_certificate"]["same_supported_donor_coordinate_tables_as_primary"]
    assert fit["fit_certificate"]["pooled_tables_sha256"] == subject._array_sha256(
        component.sum(axis=0, keepdims=True)
    )


def test_pooled_poisson_uses_the_same_supported_donor_tables() -> None:
    tables = np.broadcast_to(
        np.asarray([[32, 32], [32, 32]], dtype=np.int64),
        (2, 17, 17, 2, 2),
    ).copy()
    tables[0, 0, 0] = np.asarray([[40, 24], [20, 44]])
    tables[1, 0, 0] = np.asarray([[100, 1], [1, 26]])
    support = np.zeros((2, 17, 17), dtype=bool)
    support[0, 0, 0] = True
    mask = np.zeros((17, 17), dtype=bool)
    mask[0, 0] = True
    fit = subject._fit_pooled_poisson(tables, support, mask)
    expected = math.log(40 * 44 / 24 / 20)
    assert fit["population_log_odds"][0, 0] == pytest.approx(expected)
    assert fit["fit_certificate"]["coordinate_count"] == 1
    primary = subject._fit_structured(
        tables,
        support,
        np.eye(17),
        np.eye(17),
        subject.StructuredConfig(2, 0.1, 0.01, 0.0, 1.0),
    )
    expected_support = np.zeros((17, 17), dtype=np.int64)
    expected_support[0, 0] = 1
    assert primary["fit_certificate"]["support_count_sha256"] == (
        subject._array_sha256(expected_support)
    )
    assert fit["fit_certificate"]["pooled_tables_sha256"] == (
        subject._array_sha256(tables[:1, 0, 0])
    )


def test_all_topology_null_pairs_are_unique_and_nonidentity() -> None:
    subject._validate_topology_permutations()
    identity = np.arange(subject.MARKER_COUNT)
    pairs = []
    for control in range(subject.TOPOLOGY_NULL_COUNT):
        rna = subject._permutation(control, "rna")
        adt = subject._permutation(control, "adt")
        assert not np.array_equal(rna, identity)
        assert not np.array_equal(adt, identity)
        pairs.append((tuple(rna), tuple(adt)))
    assert len(set(pairs)) == 63


def test_source_gate_uses_7_pool_27_donor_and_bootstrap_contract() -> None:
    batches = list(subject.EXPECTED_BATCH_AXIS)
    primary = np.full(39, 0.80)
    zero = np.ones(39)
    classical = np.full(39, 0.95)
    batch_array = np.asarray(batches)
    primary[batch_array == 7] = 1.01
    losses = {
        name: np.full(39, 1.1)
        for name in (
            "common_effect_cmle",
            "pooled_saturated_poisson",
            "pearson_residual",
            "root_deviance_residual",
            "destroyed_link",
            "independence",
        )
    }
    losses.update(
        {
            "primary": primary,
            "matched_graph_zero": zero,
            "primary_classical_residual": classical,
        }
    )
    nested = {
        "donor_axis": list(subject.EXPECTED_DONORS),
        "batch_axis": batches,
        "losses": losses,
    }
    selection = SimpleNamespace(
        selected_primary=subject.StructuredConfig(2, 0.1, 0.01, 0.01, 1.0)
    )
    support_coverage = {
        "comparison_mask_floor_passes": True,
        "every_source_donor_support_floor_passes": True,
    }
    gate = subject._source_gate(nested, selection, support_coverage)
    assert gate["passes"] is True
    assert gate["improved_outer_pool_means"] == 7
    assert gate["favorable_source_donors"] == 35
    assert gate["exact_sign_reference"][
        "at_least_7_of_8_pool_signs_one_sided_p"
    ] == pytest.approx(9 / 256)
    assert gate["checks"]["comparison_mask_floor_passes"] is True
    assert gate["checks"]["every_source_donor_support_floor_passes"] is True

    support_coverage["comparison_mask_floor_passes"] = False
    assert subject._source_gate(nested, selection, support_coverage)["passes"] is False


def test_within_pool_bootstrap_matches_manual_rng_for_unequal_pools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "SOURCE_BOOTSTRAPS", 101)
    primary = np.asarray([0.8, 1.2, 0.7, 0.9, 1.0])
    comparator = np.ones(5)
    batches = [0, 0, 1, 1, 1]
    name = "manual-unequal-pools"
    observed = subject._stratified_bootstrap(primary, comparator, batches, name)
    differences = primary - comparator
    generator = np.random.default_rng(subject._bootstrap_seed(name))
    manual = []
    batch_axis = np.asarray(batches)
    for batch in (0, 1):
        values = differences[batch_axis == batch]
        draws = generator.integers(0, len(values), size=(101, len(values)))
        manual.append(values[draws].mean(axis=1))
    replicate = np.asarray(manual).mean(axis=0)
    expected = np.quantile(replicate, (0.025, 0.975), method="linear")
    np.testing.assert_allclose(observed["interval_95_percent"], expected)
    assert observed["observed_equal_batch_mean_difference"] == pytest.approx(
        np.mean([differences[:2].mean(), differences[2:].mean()])
    )


def test_support_coverage_consolidates_nested_final_and_all_source_floors() -> None:
    mask = {
        "coordinate_count": 232,
        "mask_sha256": "a" * 64,
        "training_donor_supported_coordinate_counts": {"training": 232},
        "checks": {
            "at_least_232_coordinates": True,
            "every_training_donor_has_at_least_232_coordinates": True,
        },
    }
    fold = {
        "comparison_mask": dict(mask),
        "validation_supported_coordinate_counts": {"donor": 232},
    }
    nested = {
        "outer_folds": {
            "0": {
                "inner_selection": {"folds": {"1": fold}},
                "outer_training_design": {"comparison_mask": dict(mask)},
                "validation_supported_coordinate_counts": {"outer": 232},
            }
        }
    }
    selection = SimpleNamespace(fold_records={2: fold})
    all_source = {**mask, "training_donor_supported_coordinate_counts": {"all": 232}}
    certificate = subject._source_support_coverage(nested, selection, all_source)
    assert certificate["comparison_mask_floor_passes"] is True
    assert certificate["every_source_donor_support_floor_passes"] is True
    assert certificate["mask_certificate_count"] == 4
    assert certificate["donor_support_certificate_count"] == 7

    nested["outer_folds"]["0"]["outer_training_design"]["comparison_mask"][
        "coordinate_count"
    ] = 231
    assert (
        subject._source_support_coverage(nested, selection, all_source)[
            "comparison_mask_floor_passes"
        ]
        is False
    )
    nested["outer_folds"]["0"]["outer_training_design"]["comparison_mask"][
        "coordinate_count"
    ] = 232
    nested["outer_folds"]["0"]["inner_selection"]["folds"]["1"]["comparison_mask"][
        "checks"
    ]["every_training_donor_has_at_least_232_coordinates"] = False
    assert (
        subject._source_support_coverage(nested, selection, all_source)[
            "comparison_mask_floor_passes"
        ]
        is False
    )


def test_write_json_is_exclusive(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    subject._write_json_exclusive(path, {"status": "first"})
    with pytest.raises(FileExistsError):
        subject._write_json_exclusive(path, {"status": "second"})


def test_source_paths_do_not_name_held_numeric_artifacts() -> None:
    assert "development/gse181897_source" in str(subject.DEFAULT_INPUT)
    assert "confirmation/gse181897" not in str(subject.DEFAULT_INPUT)
    assert subject.FORBIDDEN_BATCHES == (8, 9, 10, 11)


def test_model_interruption_is_terminal_and_blocks_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = tmp_path / "attempt.json"
    attempt.write_text("{}\n")
    terminal = tmp_path / "model-terminal.json"
    args = SimpleNamespace(
        model_terminal=terminal,
        attempt=attempt,
        input=tmp_path / "source.npz",
        manifest=tmp_path / "manifest.json",
        output=tmp_path / "model.json",
        reduction_terminal=tmp_path / "reduction-terminal.json",
    )
    monkeypatch.setattr(subject, "_require_runtime", lambda: None)
    monkeypatch.setattr(
        subject,
        "_load_source",
        lambda *_args: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    result = subject._run_model_one_shot(args)
    assert result["status"] == "TERMINAL_SOURCE_MODEL_REFUSAL"
    assert result["reason_code"] == "KeyboardInterrupt"
    with pytest.raises(FileExistsError):
        subject._run_model_one_shot(args)
