import gzip
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import pytest

from experiments import confirm_gse185381_aml as confirmation


def _markers(count: int = 9):
    return [
        {
            "rna_symbol": f"G{index}",
            "adt_aliases": [f"CD{index}", f"ADT-CD{index}-G{index}"],
        }
        for index in range(count)
    ]


def _run_synthetic_score(monkeypatch, outcomes, mutation=None):
    monkeypatch.setattr(confirmation, "MINIMUM_VALID_RNA_AXES", 1)
    monkeypatch.setattr(confirmation, "MINIMUM_VALID_ADT_AXES", 1)
    monkeypatch.setattr(confirmation, "MINIMUM_VALID_ORDERED_PAIRS", 1)
    donors = [
        {
            "donor_id": f"AML{index:02d}",
            "role": "held",
            "acquisition_cluster": f"B{index % 13}",
        }
        for index in range(39)
    ]
    locked = "common_effect_stratified_cmle"
    methods = [
        "primary",
        "graph_zero_ablation",
        "destroyed_link",
        confirmation.RAW_RESIDUAL_METHOD,
        locked,
        confirmation.INDEPENDENCE_METHOD,
    ]
    values = {
        "primary": 1.0,
        locked: 2.0,
        confirmation.RAW_RESIDUAL_METHOD: 3.0,
        confirmation.INDEPENDENCE_METHOD: 4.0,
        "graph_zero_ablation": 5.0,
        "destroyed_link": 6.0,
    }
    predicted = {
        method: [[[[value, 0.0], [0.0, 0.0]]]]
        for method, value in values.items()
    }
    samples = [
        {
            "donor_id": row["donor_id"],
            "row_margins": [[[384, 0]]],
            "column_margins": [[[384, 0]]],
            "valid_ordered_pair_mask": [[True]],
            "valid_ordered_pair_mask_sha256": "h",
            "valid_rna_axes": 1,
            "required_valid_rna_axes": 1,
            "valid_adt_axes": 1,
            "required_valid_adt_axes": 1,
            "valid_ordered_pairs": 1,
            "required_valid_ordered_pairs": 1,
            "predicted_tables": predicted,
            "prediction_sha256": {method: "h" for method in methods},
        }
        for row in donors
    ]
    prediction = {
        "samples": samples,
        "methods": methods,
        "locked_classical_method": locked,
    }
    source = {
        "model": {
            "locked_classical_method": locked,
            "primary_selection": {
                "selected_configuration": {"graph_penalty": 0.1}
            },
        }
    }
    rna_public = {
        "rna_states": {"sha256": "r" * 64, "bytes": 1},
        "rna_axis_quality": {
            row["donor_id"]: {"axis_valid": [True]} for row in donors
        },
    }
    adt_public = {
        "adt_states": {"sha256": "a" * 64, "bytes": 1},
        "adt_axis_quality": {
            row["donor_id"]: {"axis_valid": [True]} for row in donors
        },
    }
    states = {row["donor_id"]: [[0] for _ in range(384)] for row in donors}
    rna_private = {"schema": "gse185381-private-rna-states/1.0", "states": states}
    adt_private = {"schema": "gse185381-private-adt-states/1.0", "states": states}
    if mutation is not None:
        mutation(prediction, rna_public, adt_public)
    monkeypatch.setattr(confirmation, "_candidate", lambda: {"donors": donors})
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
        lambda path, certificate: (
            rna_private if Path(path).name.startswith("rna") else adt_private
        ),
    )
    monkeypatch.setattr(confirmation, "_array_sha256", lambda value: "h")
    monkeypatch.setattr(confirmation, "_sha256", lambda path: "s" * 64)
    monkeypatch.setattr(
        confirmation,
        "_donor_loss",
        lambda truth, estimate, mask: float(np.asarray(estimate).flat[0]),
    )

    def comparison(donor_axis, clusters, primary, comparator):
        marker = float(np.asarray(comparator)[0])
        return {"passes": outcomes[marker], "comparator_marker": marker}

    monkeypatch.setattr(confirmation, "_comparison", comparison)
    return confirmation._score_stage_body(Path("rna-private.json"), Path("adt-private.json"))


def test_matrix_parser_preserves_requested_header_order_and_exact_aliases(
    tmp_path: Path,
):
    path = tmp_path / "matrix.csv.gz"
    with gzip.open(path, "wt", newline="") as handle:
        handle.write(
            ",pool:cell-b,pool:cell-a,unused\n"
            "CD0RA,99,99,99\n"
            "ADT-CD0-G0,3,7,0\n"
            "CD1,5,11,0\n"
        )
    observed = confirmation._read_matrix_rows(
        path,
        ["pool:cell-a", "pool:cell-b"],
        _markers(2),
        "adt",
    )
    np.testing.assert_array_equal(observed, [[7, 11], [3, 5]])


def test_matrix_parser_refuses_duplicate_exact_alias_hits(tmp_path: Path):
    path = tmp_path / "matrix.csv.gz"
    with gzip.open(path, "wt", newline="") as handle:
        handle.write(",cell-a\nCD0,1\nADT-CD0-G0,2\n")
    with pytest.raises(confirmation.ConfirmationRefusal, match="AMBIGUOUS"):
        confirmation._read_matrix_rows(path, ["cell-a"], _markers(1), "adt")


def test_matrix_parser_refuses_shifted_or_truncated_rows(tmp_path: Path):
    path = tmp_path / "matrix.csv.gz"
    with gzip.open(path, "wt", newline="") as handle:
        handle.write(",cell-a,cell-b\nG0,1\n")
    with pytest.raises(confirmation.ConfirmationRefusal, match="ALIGNMENT"):
        confirmation._read_matrix_rows(path, ["cell-a"], _markers(1), "rna")


def test_cell_selection_is_salted_deterministic_and_exact_budget():
    donor = "Control1"
    assignments = {
        f"pool:cell-{index:04d}": {"donor": donor, "pool_id": "pool"}
        for index in range(500)
    }
    first = confirmation._selected_cells(
        assignments, [donor], {donor: "pool"}
    )[donor]
    second = confirmation._selected_cells(
        dict(reversed(list(assignments.items()))), [donor], {donor: "pool"}
    )[donor]
    assert first == second
    assert len(first) == confirmation.CELL_BUDGET
    assert len(set(first)) == confirmation.CELL_BUDGET
    expected = sorted(
        assignments,
        key=lambda cell: (
            hashlib.sha256(
                f"{confirmation.CELL_SELECTION_SALT}|{donor}|{cell}".encode()
            ).hexdigest(),
            cell,
        ),
    )[: confirmation.CELL_BUDGET]
    assert first == expected


def test_cell_selection_never_aggregates_across_pools():
    donor = "Control1"
    assignments = {
        **{
            f"pool-a:cell-{index:04d}": {"donor": donor, "pool_id": "pool-a"}
            for index in range(400)
        },
        **{
            f"pool-b:cell-{index:04d}": {"donor": donor, "pool_id": "pool-b"}
            for index in range(500)
        },
    }
    selected = confirmation._selected_cells(
        assignments, [donor], {donor: "pool-a"}
    )[donor]
    assert len(selected) == confirmation.CELL_BUDGET
    assert all(cell.startswith("pool-a:") for cell in selected)


def test_source_pool_projection_never_selects_held_columns():
    selected = {"Control1": ["pool:control-a", "pool:control-b"]}
    assignments = {
        "pool:control-a": {"donor": "Control1", "pool_id": "pool"},
        "pool:control-b": {"donor": "Control1", "pool_id": "pool"},
        "pool:aml-a": {"donor": "AML1", "pool_id": "pool"},
    }
    cells, destinations = confirmation._pool_selected(selected, assignments, "pool")
    assert cells == ["pool:control-a", "pool:control-b"]
    assert destinations == [("Control1", 0), ("Control1", 1)]


def test_all_zero_adt_axis_is_excluded_despite_deterministic_midrank():
    cells = [f"cell-{index:03d}" for index in range(confirmation.CELL_BUDGET)]
    counts = np.zeros((confirmation.CELL_BUDGET, 9))
    first = confirmation._adt_states(counts, cells, "AML1")
    second = confirmation._adt_states(counts, cells, "AML1")
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(
        first.sum(axis=0), np.full(9, confirmation.CELL_BUDGET // 2)
    )
    quality = confirmation._adt_axis_quality(counts)
    assert not quality["valid"].any()
    np.testing.assert_array_equal(quality["distinct_values"], np.ones(9))
    np.testing.assert_array_equal(
        quality["largest_equal_value_fraction"], np.ones(9)
    )


def test_poisson_independence_signed_deviance_includes_zero_cells():
    table = np.asarray([[0, 12], [8, 20]], dtype=np.int64)
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()
    positive = table > 0
    deviance = 2.0 * np.sum(
        table[positive] * np.log(table[positive] / expected[positive])
    )
    determinant = table[0, 0] * table[1, 1] - table[0, 1] * table[1, 0]
    manual = math.copysign(math.sqrt(deviance), determinant)
    assert confirmation._poisson_signed_deviance(table) == pytest.approx(manual)
    assert np.isfinite(confirmation._poisson_signed_deviance(table))


def test_hierarchical_and_poisson_predictions_use_identical_margins_and_masks():
    rng = np.random.default_rng(20260829)
    size = 9
    positives = rng.integers(40, 340, size=size)
    rows = np.repeat(
        np.stack([confirmation.CELL_BUDGET - positives, positives], axis=1)[:, None, :],
        size,
        axis=1,
    )
    columns = np.broadcast_to(
        np.asarray([confirmation.CELL_BUDGET // 2] * 2), (size, size, 2)
    ).copy()
    primary = confirmation._predict_log_odds(
        rng.normal(size=(size, size)), rows, columns, 0.75
    )
    residual = confirmation._predict_residual(
        rng.normal(scale=0.1, size=(size, size)), rows, columns, 1.0
    )
    for estimate in (primary, residual):
        np.testing.assert_allclose(estimate.sum(axis=-1), rows)
        np.testing.assert_allclose(estimate.sum(axis=-2), columns)
    expected_mask = np.all(rows > 0, axis=-1) & np.all(columns > 0, axis=-1)
    assert expected_mask.all()
    assert confirmation._informative(primary).all()
    assert confirmation._informative(residual).all()


def test_prediction_stage_never_forms_or_opens_target_joint_states(monkeypatch):
    markers = _markers(16)
    donors = [
        {
            "donor_id": f"AML{index:02d}",
            "role": "held",
            "acquisition_cluster": f"B{index % 13}",
            "selected_pool_id": f"P{index:02d}",
        }
        for index in range(39)
    ]
    candidate = {"donors": donors, "markers": markers}
    source = {
        "model": {
            "models": {
                "primary": {
                    "family": "primary",
                    "population_log_odds": np.zeros((16, 16)).tolist(),
                    "transport_multiplier": 1.0,
                },
                "graph_zero_ablation": {
                    "family": "graph-zero",
                    "population_log_odds": np.zeros((16, 16)).tolist(),
                    "transport_multiplier": 1.0,
                },
                "destroyed_link": {
                    "family": "destroyed",
                    "population_log_odds": np.zeros((16, 16)).tolist(),
                    "transport_multiplier": 1.0,
                },
                confirmation.RAW_RESIDUAL_METHOD: {
                    "family": "residual",
                    "pooled_coordinate": np.zeros((16, 16)).tolist(),
                    "transport_multiplier": 1.0,
                },
                confirmation.INDEPENDENCE_METHOD: {
                    "family": "target_margin_independence"
                },
            },
            "available_methods": [
                "primary",
                "graph_zero_ablation",
                "destroyed_link",
                confirmation.RAW_RESIDUAL_METHOD,
                confirmation.INDEPENDENCE_METHOD,
            ],
                "locked_classical_method": confirmation.RAW_RESIDUAL_METHOD,
                "primary_selection": {
                    "selected_configuration": {"graph_penalty": 0.1}
                },
            }
    }
    row_margin = {marker["rna_symbol"]: [300, 84] for marker in markers}
    column_margin = {marker["rna_symbol"]: [192, 192] for marker in markers}
    rna = {
        "held_donors": [row["donor_id"] for row in donors],
        "selected_axis_sha256": {row["donor_id"]: "r" * 64 for row in donors},
        "row_margins": {row["donor_id"]: row_margin for row in donors},
        "rna_axis_quality": {
            row["donor_id"]: {"axis_valid": [True] * 16} for row in donors
        },
    }
    adt = {
        "held_donors": [row["donor_id"] for row in donors],
        "selected_axis_sha256": rna["selected_axis_sha256"],
        "column_margins": {row["donor_id"]: column_margin for row in donors},
        "adt_axis_quality": {
            row["donor_id"]: {"axis_valid": [True] * 16} for row in donors
        },
    }

    monkeypatch.setattr(confirmation, "_candidate", lambda: candidate)
    monkeypatch.setattr(
        confirmation,
        "_require_completed",
        lambda stage: {"source": source, "rna": rna, "adt": adt}[stage],
    )
    monkeypatch.setattr(confirmation, "_sha256", lambda path: "a" * 64)
    monkeypatch.setattr(
        confirmation,
        "_binary_tables",
        lambda *args: pytest.fail("prediction formed target joint tables"),
    )
    monkeypatch.setattr(
        confirmation,
        "_private_artifact",
        lambda *args: pytest.fail("prediction opened private target states"),
    )

    payload = confirmation._prediction_stage_body()
    assert payload["status"] == "HELD_PREDICTIONS_FROZEN_BEFORE_RNA_ADT_PAIRING"
    assert payload["held_private_state_artifacts_opened"] == 0
    assert payload["held_joint_tables_formed"] == 0
    assert len(payload["samples"]) == 39
    assert payload["locked_classical_method"] == confirmation.RAW_RESIDUAL_METHOD
    assert all(row["valid_rna_axes"] == 16 for row in payload["samples"])
    assert all(row["valid_adt_axes"] == 16 for row in payload["samples"])
    assert all(row["valid_ordered_pairs"] == 256 for row in payload["samples"])


@pytest.mark.parametrize(
    ("valid_rna", "valid_adt"),
    [(8, 16), (16, 8), (9, 14)],
)
def test_prediction_stage_refuses_held_donors_below_any_support_floor(
    monkeypatch, valid_rna, valid_adt
):
    markers = _markers(16)
    donors = [
        {
            "donor_id": f"AML{index:02d}",
            "role": "held",
            "acquisition_cluster": f"B{index % 13}",
            "selected_pool_id": f"P{index:02d}",
        }
        for index in range(39)
    ]
    methods = [
        "primary",
        "graph_zero_ablation",
        "destroyed_link",
        confirmation.RAW_RESIDUAL_METHOD,
        confirmation.INDEPENDENCE_METHOD,
    ]
    source = {
        "model": {
            "models": {method: {} for method in methods},
            "available_methods": methods,
            "locked_classical_method": confirmation.RAW_RESIDUAL_METHOD,
            "primary_selection": {
                "selected_configuration": {"graph_penalty": 0.1}
            },
        }
    }
    row_margin = {marker["rna_symbol"]: [300, 84] for marker in markers}
    column_margin = {marker["rna_symbol"]: [192, 192] for marker in markers}
    rna = {
        "held_donors": [row["donor_id"] for row in donors],
        "selected_axis_sha256": {row["donor_id"]: "r" * 64 for row in donors},
        "row_margins": {row["donor_id"]: row_margin for row in donors},
        "rna_axis_quality": {
            row["donor_id"]: {
                "axis_valid": [True] * valid_rna + [False] * (16 - valid_rna)
            }
            for row in donors
        },
    }
    adt = {
        "held_donors": [row["donor_id"] for row in donors],
        "selected_axis_sha256": rna["selected_axis_sha256"],
        "column_margins": {row["donor_id"]: column_margin for row in donors},
        "adt_axis_quality": {
            row["donor_id"]: {
                "axis_valid": [True] * valid_adt + [False] * (16 - valid_adt)
            }
            for row in donors
        },
    }
    monkeypatch.setattr(
        confirmation,
        "_candidate",
        lambda: {"donors": donors, "markers": markers},
    )
    monkeypatch.setattr(
        confirmation,
        "_require_completed",
        lambda stage: {"source": source, "rna": rna, "adt": adt}[stage],
    )
    monkeypatch.setattr(
        confirmation,
        "_predict_model",
        lambda *args: pytest.fail("prediction ran below the support floor"),
    )
    with pytest.raises(
        confirmation.ConfirmationRefusal,
        match="HELD_DONOR_FAILS_FROZEN_SUPPORT_CONTRACT",
    ):
        confirmation._prediction_stage_body()


def test_confirmation_gate_passes_only_when_every_frozen_requirement_holds(
    monkeypatch,
):
    monkeypatch.setattr(confirmation, "BOOTSTRAPS", 2_000)
    donors = [f"AML{index:02d}" for index in range(39)]
    clusters = [f"B{index % 13}" for index in range(39)]
    comparator = np.ones(39)
    decisive = confirmation._comparison(
        donors, clusters, np.full(39, 0.80), comparator
    )
    assert decisive["passes"]
    assert decisive["relative_loss_reduction"] == pytest.approx(0.20)
    assert decisive["exact_donor_sign_test"]["one_sided_p"] <= 0.0125
    assert (
        decisive["exact_acquisition_cluster_sign_flip"]["one_sided_p"]
        <= 0.0125
    )
    assert decisive["acquisition_cluster_bootstrap"]["draws"] == 2_000
    assert all(decisive["checks"].values())

    insufficient = confirmation._comparison(
        donors, clusters, np.full(39, 0.96), comparator
    )
    assert not insufficient["passes"]
    assert not insufficient["checks"]["relative_loss_reduction_at_least_five_percent"]


def test_failed_estimator_gate_still_reports_raw_poisson_and_independence(monkeypatch):
    result = _run_synthetic_score(
        monkeypatch,
        {2.0: False, 3.0: True, 4.0: True},
    )
    assert result["status"] == (
        "COMPLETED_ESTIMATOR_GATE_FAIL_PRIMARY_VS_LOCKED_CLASSICAL"
    )
    assert result["primary_vs_untuned_poisson_residual"]["comparator_marker"] == 3.0
    assert result["target_margin_independence_head_to_head"][
        "comparator_marker"
    ] == 4.0
    assert result["graph_zero_serial_secondary"]["status"] == (
        "NOT_EVALUATED_ESTIMATOR_GATE_FAILED"
    )
    assert result["destroyed_link_serial_secondary"]["status"] == (
        "NOT_EVALUATED_ESTIMATOR_GATE_FAILED"
    )
    assert not result["estimator_validation_supported"]
    assert not result["structured_coupling_field_supported"]
    assert all(
        "graph_zero_ablation" not in sample["losses"]
        and "destroyed_link" not in sample["losses"]
        for sample in result["samples"]
    )


def test_graph_gate_failure_stops_destroyed_link_gate(monkeypatch):
    result = _run_synthetic_score(
        monkeypatch,
        {2.0: True, 3.0: True, 4.0: True, 5.0: False},
    )
    assert result["status"] == "COMPLETED_ESTIMATOR_GATE_PASS"
    assert result["graph_zero_serial_secondary"]["status"] == (
        "EVALUATED_AFTER_ESTIMATOR_GATE_PASS"
    )
    assert result["destroyed_link_serial_secondary"]["status"] == (
        "NOT_EVALUATED_GRAPH_STRUCTURE_GATE_FAILED"
    )
    assert not result["graph_structure_supported"]
    assert not result["coupling_link_supported"]
    assert all("destroyed_link" not in sample["losses"] for sample in result["samples"])


def test_serial_chain_distinguishes_estimator_graph_and_link_support(monkeypatch):
    result = _run_synthetic_score(
        monkeypatch,
        {2.0: True, 3.0: True, 4.0: True, 5.0: True, 6.0: True},
    )
    assert result["confirmatory_gate_order"] == [
        "primary_vs_locked_classical",
        "primary_vs_graph_zero_ablation",
        "primary_vs_destroyed_link",
    ]
    assert result["estimator_validation_supported"]
    assert result["graph_structure_supported"]
    assert result["coupling_link_supported"]
    assert result["structured_coupling_field_supported"]
    assert result["broad_classical_descriptive_gain"]
    assert result["destroyed_link_serial_secondary"]["status"] == (
        "EVALUATED_AFTER_GRAPH_STRUCTURE_GATE_PASS"
    )


@pytest.mark.parametrize("mutation_kind", ["count", "mask"])
def test_score_stage_rejects_frozen_support_that_differs_from_public_qc(
    monkeypatch, mutation_kind
):
    def mutate(prediction, _rna_public, _adt_public):
        sample = prediction["samples"][0]
        if mutation_kind == "count":
            sample["valid_rna_axes"] = 0
        else:
            sample["valid_ordered_pair_mask"] = [[False]]

    message = (
        "frozen support contract differs"
        if mutation_kind == "count"
        else "frozen valid mask differs from separate axis QC"
    )
    with pytest.raises(PermissionError, match=message):
        _run_synthetic_score(
            monkeypatch,
            {2.0: True, 3.0: True, 4.0: True},
            mutation=mutate,
        )


def test_donor_loss_enforces_the_128_pair_backstop():
    table = np.asarray([[96, 96], [96, 96]], dtype=float)
    truth = np.tile(table, (16, 16, 1, 1))
    mask = np.zeros((16, 16), dtype=bool)
    mask.flat[: confirmation.MINIMUM_VALID_ORDERED_PAIRS] = True
    assert confirmation._donor_loss(truth, truth.copy(), mask) == pytest.approx(0.0)
    mask.flat[confirmation.MINIMUM_VALID_ORDERED_PAIRS - 1] = False
    with pytest.raises(
        confirmation.ConfirmationRefusal,
        match="FEWER_THAN_MINIMUM_VALID_ORDERED_PAIRS",
    ):
        confirmation._donor_loss(truth, truth.copy(), mask)


def test_claim_and_run_are_one_shot_and_require_attempt_tag(
    tmp_path: Path, monkeypatch
):
    stage_paths = {"source": tmp_path / "source.json"}
    attempt_paths = {"source": tmp_path / "source.jsonl"}
    execution_paths = {"source": tmp_path / "execution.json"}
    access_paths = {"source": tmp_path / "access.jsonl"}
    observed_tags = []
    monkeypatch.setattr(confirmation, "STAGE_PATHS", stage_paths)
    monkeypatch.setattr(confirmation, "ATTEMPT_PATHS", attempt_paths)
    monkeypatch.setattr(confirmation, "EXECUTION_CLAIM_PATHS", execution_paths)
    monkeypatch.setattr(confirmation, "ACCESS_JOURNAL_PATHS", access_paths)
    monkeypatch.setattr(confirmation, "ATTEMPT_TAGS", {"source": "attempt-tag"})
    monkeypatch.setattr(confirmation, "_relative", lambda path: path.name)
    monkeypatch.setattr(
        confirmation,
        "_stage_prerequisites",
        lambda stage: {"protocol_tag": "public-protocol"},
    )
    runtime = {"runtime": "frozen"}
    monkeypatch.setattr(
        confirmation, "_require_runtime_environment", lambda: runtime
    )
    monkeypatch.setattr(
        confirmation, "_protocol_paths", lambda *paths: tuple(map(str, paths))
    )

    def require_tag(tag, paths):
        observed_tags.append((tag, tuple(paths)))
        return "c" * 40

    monkeypatch.setattr(confirmation, "_require_public_tag", require_tag)

    started = confirmation.claim_stage("source")
    assert started["status"] == "STARTED"
    with pytest.raises(PermissionError):
        confirmation.claim_stage("source")

    result = confirmation._run_claimed_stage(
        "source",
        lambda: {"schema": "synthetic/1.0", "status": "SOURCE_MODEL_FROZEN"},
    )
    assert result["status"] == "SOURCE_MODEL_FROZEN"
    assert any(tag == "attempt-tag" for tag, _ in observed_tags)
    rows = [
        json.loads(line) for line in attempt_paths["source"].read_text().splitlines()
    ]
    assert [row["status"] for row in rows] == [
        "STARTED",
        "EXECUTING_CONSUMED",
        "FINISHED",
    ]
    assert json.loads(execution_paths["source"].read_text()) == rows[1]
    with pytest.raises(PermissionError):
        confirmation._run_claimed_stage(
            "source", lambda: pytest.fail("one-shot body ran twice")
        )


def test_unexpected_exception_is_terminal_without_exception_text(
    tmp_path: Path, monkeypatch
):
    stage_paths = {"source": tmp_path / "source.json"}
    attempt_paths = {"source": tmp_path / "source.jsonl"}
    execution_paths = {"source": tmp_path / "execution.json"}
    access_paths = {"source": tmp_path / "access.jsonl"}
    runtime = {"runtime": "frozen"}
    monkeypatch.setattr(confirmation, "STAGE_PATHS", stage_paths)
    monkeypatch.setattr(confirmation, "ATTEMPT_PATHS", attempt_paths)
    monkeypatch.setattr(confirmation, "EXECUTION_CLAIM_PATHS", execution_paths)
    monkeypatch.setattr(confirmation, "ACCESS_JOURNAL_PATHS", access_paths)
    monkeypatch.setattr(confirmation, "ATTEMPT_TAGS", {"source": "attempt-tag"})
    monkeypatch.setattr(confirmation, "_relative", lambda path: path.name)
    monkeypatch.setattr(
        confirmation,
        "_stage_prerequisites",
        lambda stage: {"protocol_tag": "public-protocol"},
    )
    monkeypatch.setattr(
        confirmation, "_require_runtime_environment", lambda: runtime
    )
    monkeypatch.setattr(
        confirmation, "_require_public_tag", lambda tag, paths: "c" * 40
    )
    monkeypatch.setattr(
        confirmation,
        "_candidate",
        lambda: {
            "donors": [
                {
                    "donor_id": "Control1",
                    "role": "source",
                    "selected_pool_id": "pool-1",
                }
            ],
            "pools": [
                {
                    "pool_id": "pool-1",
                    "rna_url": "https://example.org/assay.csv.gz",
                    "rna_file": {
                        "bytes": 5,
                        "sha256": hashlib.sha256(b"assay").hexdigest(),
                    },
                }
            ],
        },
    )
    confirmation.claim_stage("source")

    observed = tmp_path / "assay.csv.gz"
    observed.write_bytes(b"assay")

    def fail_after_access():
        confirmation._append_assay_access(
            "source",
            "pool-1",
            "rna",
            "https://example.org/assay.csv.gz",
            observed,
            {"bytes": 5, "sha256": hashlib.sha256(b"assay").hexdigest()},
        )
        raise RuntimeError("private detail")

    result = confirmation._run_claimed_stage("source", fail_after_access)
    assert result["status"] == "TERMINAL_SOURCE_REFUSAL"
    assert result["refusal_code"] == "UNEXPECTED_RUNTIMEERROR"
    assert len(result["assay_access"]["downloaded_and_hashed_files"]) == 1
    assert result["assay_access"]["downloaded_and_hashed_files"][0][
        "observed_sha256"
    ] == hashlib.sha256(b"assay").hexdigest()
    assert "private detail" not in stage_paths["source"].read_text()
    with pytest.raises(PermissionError):
        confirmation._run_claimed_stage("source", lambda: {})


def test_frozen_candidate_and_manifest_enforce_single_pool_components():
    candidate = confirmation._candidate()
    source = [row for row in candidate["donors"] if row["role"] == "source"]
    held = [row for row in candidate["donors"] if row["role"] == "held"]
    assert len(source) == 10
    assert len(held) == 39
    components = candidate["source_split"]["components"]
    assert candidate["source_split"]["pool_disjoint"] is True
    assert len({row["selected_pool_id"] for row in components}) == len(components)
    assert sum(len(row["donors"]) for row in components) == 10
    for component in components:
        assert all(
            next(
                row["selected_pool_id"]
                for row in source
                if row["donor_id"] == donor
            )
            == component["selected_pool_id"]
            for donor in component["donors"]
        )
    manifest = confirmation._manifest(candidate)
    assert len(manifest["pool_files"]) == len(candidate["pools"])


def test_component_equal_loss_does_not_overweight_multi_donor_pool():
    components = [["a", "b", "c"], ["d"], ["e"]]
    losses = {"a": 3.0, "b": 3.0, "c": 3.0, "d": 0.0, "e": 0.0}
    assert confirmation._component_equal_mean(losses, components) == pytest.approx(
        1.0
    )
    assert np.mean(list(losses.values())) == pytest.approx(1.8)


def test_knn_incidence_is_deterministic_and_graph_has_no_singletons():
    rng = np.random.default_rng(8)
    profiles = rng.normal(size=(5, 9))
    first = confirmation._knn_incidence(profiles)
    second = confirmation._knn_incidence(profiles.copy())
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first.sum(axis=0), np.full(first.shape[1], 2.0))
    assert np.all(first.sum(axis=1) >= confirmation.GRAPH_NEIGHBORS)


def test_primary_graph_grid_excludes_the_graph_zero_ablation():
    protocol = json.loads(confirmation.DEFAULT_PROTOCOL.read_text())
    assert confirmation.GRAPH_GRID == (0.1, 1.0)
    assert protocol["primary_estimator"]["grid"]["graph_penalty"] == [0.1, 1.0]
    assert all(
        config.graph_penalty > 0.0
        for config in confirmation._primary_configs(confirmation.GRAPH_GRID)
    )
    assert {
        "mapreg/__init__.py",
        "mapreg/classical_residuals.py",
        "mapreg/coupling_fields.py",
        "mapreg/factorial_coupling.py",
    } <= set(confirmation.PROTOCOL_BINDINGS)


@pytest.mark.parametrize(
    ("valid_rna", "valid_adt"),
    [(8, 16), (16, 8), (9, 14)],
)
def test_source_record_construction_enforces_every_support_floor(
    valid_rna, valid_adt
):
    cells = [f"cell-{index:03d}" for index in range(confirmation.CELL_BUDGET)]
    rna = np.tile(np.arange(confirmation.CELL_BUDGET)[:, None] % 2, (1, 16))
    adt = np.tile(np.arange(confirmation.CELL_BUDGET)[:, None], (1, 16))
    rna[:, valid_rna:] = 0
    adt[:, valid_adt:] = 0
    with pytest.raises(
        confirmation.ConfirmationRefusal,
        match="SOURCE_DONOR_FAILS_FROZEN_SUPPORT_CONTRACT",
    ):
        confirmation._source_records(
            {"markers": _markers(16)},
            {"Control1": cells},
            {"Control1": rna.astype(float)},
            {"Control1": adt.astype(float)},
        )


def test_source_record_construction_accepts_the_smallest_supported_submap():
    cells = [f"cell-{index:03d}" for index in range(confirmation.CELL_BUDGET)]
    rna = np.tile(np.arange(confirmation.CELL_BUDGET)[:, None] % 2, (1, 16))
    adt = np.tile(np.arange(confirmation.CELL_BUDGET)[:, None], (1, 16))
    rna[:, 9:] = 0
    adt[:, 15:] = 0
    record = confirmation._source_records(
        {"markers": _markers(16)},
        {"Control1": cells},
        {"Control1": rna.astype(float)},
        {"Control1": adt.astype(float)},
    )["Control1"]
    assert record["valid_rna_axes"] == 9
    assert record["valid_adt_axes"] == 15
    assert record["valid_ordered_pairs"] == 135


def test_protocol_support_contract_matches_the_runner_constants():
    protocol = json.loads(confirmation.DEFAULT_PROTOCOL.read_text())
    support = protocol["cell_and_feature_contract"]
    assert support["minimum_valid_rna_axes_per_donor"] == (
        confirmation.MINIMUM_VALID_RNA_AXES
    )
    assert support["minimum_valid_adt_axes_per_donor"] == (
        confirmation.MINIMUM_VALID_ADT_AXES
    )
    assert support["minimum_valid_ordered_pairs_per_donor"] == (
        confirmation.MINIMUM_VALID_ORDERED_PAIRS
    )


def test_target_margin_independence_and_paule_mandel_preserve_margins():
    tables = np.empty((4, 2, 2, 2, 2), dtype=np.int64)
    for donor in range(4):
        for index in np.ndindex((2, 2)):
            table = np.asarray([[90, 70], [80, 144]], dtype=np.int64)
            table[0, 0] += donor + sum(index)
            table[0, 1] -= donor + sum(index)
            tables[(donor, *index)] = table
    model = confirmation._fit_paule_mandel_log_odds(tables)
    rows, columns = confirmation._margins(tables[0])
    random_effects = confirmation._predict_model(model, rows, columns)
    independence = confirmation._predict_model(
        {"family": "target_margin_independence"}, rows, columns
    )
    for estimate in (random_effects, independence):
        np.testing.assert_allclose(estimate.sum(axis=-1), rows)
        np.testing.assert_allclose(estimate.sum(axis=-2), columns)
    assert np.isfinite(model["tau_squared"]).all()


def test_completed_public_tag_must_bind_output_ledger_and_execution_claim(
    tmp_path: Path, monkeypatch
):
    output = tmp_path / "source.json"
    attempt = tmp_path / "source.jsonl"
    execution = tmp_path / "execution.json"
    access = tmp_path / "access.jsonl"
    runtime = {"runtime": "frozen"}
    protocol_commit = "p" * 40
    monkeypatch.setattr(confirmation, "STAGE_PATHS", {"source": output})
    monkeypatch.setattr(confirmation, "ATTEMPT_PATHS", {"source": attempt})
    monkeypatch.setattr(
        confirmation, "EXECUTION_CLAIM_PATHS", {"source": execution}
    )
    monkeypatch.setattr(confirmation, "ACCESS_JOURNAL_PATHS", {"source": access})
    monkeypatch.setattr(confirmation, "ATTEMPT_TAGS", {"source": "source-attempt"})
    monkeypatch.setattr(confirmation, "_relative", lambda path: path.name)
    monkeypatch.setattr(
        confirmation, "_require_runtime_environment", lambda: runtime
    )
    prerequisites = {"protocol_tag": "public-protocol"}
    monkeypatch.setattr(
        confirmation, "_stage_prerequisites", lambda stage: prerequisites
    )
    monkeypatch.setattr(
        confirmation, "_require_public_attempt_prefix", lambda stage, row: "a" * 40
    )
    observed = []

    def require_tag(tag, paths):
        observed.append((tag, tuple(paths)))
        return protocol_commit

    monkeypatch.setattr(confirmation, "_require_public_tag", require_tag)
    started = {
        "schema": "gse185381-aml-stage-attempt/1.0",
        "stage": "source",
        "status": "STARTED",
        "created_at_utc": "2026-08-29T00:00:00Z",
        "attempt_tag_required_before_assay_access": "source-attempt",
        "prerequisites": prerequisites,
        "protocol_commit": protocol_commit,
        "runtime_environment": runtime,
        "one_shot": True,
    }
    consumed = {
        "schema": "gse185381-aml-stage-attempt/1.0",
        "stage": "source",
        "status": "EXECUTING_CONSUMED",
        "created_at_utc": "2026-08-29T00:00:01Z",
        "protocol_commit": protocol_commit,
        "runtime_environment": runtime,
        "interruption_consumes_stage": True,
    }
    payload = {
        "schema": "synthetic/1.0",
        "status": "SOURCE_READY",
        "protocol_commit": protocol_commit,
        "runtime_environment": runtime,
    }
    access_header = {
        "schema": "gse185381-aml-assay-access/1.0",
        "stage": "source",
        "status": "OPENED_BEFORE_ASSAY_ACCESS",
        "created_at_utc": "2026-08-29T00:00:00Z",
        "protocol_commit": protocol_commit,
        "runtime_environment": runtime,
    }
    confirmation._append_jsonl(access, access_header, create=True)
    payload["assay_access"] = confirmation._access_summary("source")
    confirmation._write_json_x(execution, consumed)
    confirmation._write_json_x(output, payload)
    finished = {
        "schema": "gse185381-aml-stage-attempt/1.0",
        "stage": "source",
        "status": "FINISHED",
        "created_at_utc": "2026-08-29T00:00:02Z",
        "terminal_status": "SOURCE_READY",
        "output_sha256": confirmation._sha256(output),
        "protocol_commit": protocol_commit,
        "runtime_environment": runtime,
    }
    for index, row in enumerate((started, consumed, finished)):
        confirmation._append_jsonl(attempt, row, create=index == 0)
    observed_payload, _ = confirmation._require_completed_stage_artifact(
        "source-result", output, "source", require_success=True
    )
    assert observed_payload == payload
    assert observed[0] == (
        "source-result",
        (output.name, attempt.name, execution.name, access.name),
    )


def test_started_attempt_validation_rejects_tampered_prerequisites(monkeypatch):
    runtime = {"runtime": "frozen"}
    protocol_commit = "p" * 40
    monkeypatch.setattr(confirmation, "ATTEMPT_TAGS", {"source": "source-attempt"})
    monkeypatch.setattr(
        confirmation,
        "_stage_prerequisites",
        lambda stage: {"protocol_tag": "public-protocol"},
    )
    started = {
        "schema": "gse185381-aml-stage-attempt/1.0",
        "stage": "source",
        "status": "STARTED",
        "created_at_utc": "2026-08-29T00:00:00Z",
        "attempt_tag_required_before_assay_access": "source-attempt",
        "prerequisites": {"protocol_tag": "tampered"},
        "protocol_commit": protocol_commit,
        "runtime_environment": runtime,
        "one_shot": True,
    }
    with pytest.raises(PermissionError, match="STARTED event"):
        confirmation._validate_started_attempt(
            "source", started, runtime, protocol_commit
        )


def test_access_header_validation_rejects_protocol_or_runtime_tampering():
    records = [
        {
            "schema": "gse185381-aml-assay-access/1.0",
            "stage": "source",
            "status": "OPENED_BEFORE_ASSAY_ACCESS",
            "created_at_utc": "2026-08-29T00:00:00Z",
            "protocol_commit": "wrong",
            "runtime_environment": {"runtime": "wrong"},
        }
    ]
    with pytest.raises(PermissionError, match="access header"):
        confirmation._validate_access_header(
            "source",
            records,
            "p" * 40,
            {"runtime": "frozen"},
            "2026-08-29T00:00:00Z",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pool_id", "wrong-pool"),
        ("url", "https://example.org/wrong.csv.gz"),
        ("modality", "adt"),
    ],
)
def test_access_journal_rejects_records_outside_frozen_stage_contract(
    field, value, tmp_path: Path, monkeypatch
):
    access = tmp_path / "access.jsonl"
    monkeypatch.setattr(confirmation, "ACCESS_JOURNAL_PATHS", {"rna": access})
    monkeypatch.setattr(
        confirmation,
        "_candidate",
        lambda: {
            "donors": [
                {
                    "donor_id": "AML1",
                    "role": "held",
                    "selected_pool_id": "pool-1",
                }
            ],
            "pools": [
                {
                    "pool_id": "pool-1",
                    "rna_url": "https://example.org/rna.csv.gz",
                    "rna_file": {"bytes": 3, "sha256": None},
                    "adt_url": "https://example.org/adt.csv.gz",
                    "adt_file": {"bytes": 4, "sha256": None},
                }
            ],
        },
    )
    confirmation._append_jsonl(
        access,
        {
            "schema": "gse185381-aml-assay-access/1.0",
            "stage": "rna",
            "status": "OPENED_BEFORE_ASSAY_ACCESS",
            "created_at_utc": "2026-08-29T00:00:00Z",
            "protocol_commit": "p" * 40,
            "runtime_environment": {"runtime": "frozen"},
        },
        create=True,
    )
    record = {
        "schema": "gse185381-aml-assay-access/1.0",
        "stage": "rna",
        "status": "DOWNLOADED_AND_HASHED",
        "created_at_utc": "2026-08-29T00:00:01Z",
        "pool_id": "pool-1",
        "modality": "rna",
        "url": "https://example.org/rna.csv.gz",
        "filename": "rna.csv.gz",
        "expected_bytes": 3,
        "expected_sha256": None,
        "observed_bytes": 3,
        "observed_sha256": hashlib.sha256(b"rna").hexdigest(),
    }
    record[field] = value
    confirmation._append_jsonl(access, record, create=False)
    with pytest.raises(PermissionError, match="frozen pool"):
        confirmation._access_records("rna")


def test_assay_download_is_journaled_before_size_refusal(tmp_path: Path, monkeypatch):
    access = tmp_path / "access.jsonl"
    destination = tmp_path / "assay.csv.gz"
    monkeypatch.setattr(confirmation, "ACCESS_JOURNAL_PATHS", {"source": access})
    monkeypatch.setattr(confirmation, "_relative", lambda path: path.name)
    monkeypatch.setattr(
        confirmation,
        "_candidate",
        lambda: {
            "donors": [
                {
                    "donor_id": "Control1",
                    "role": "source",
                    "selected_pool_id": "pool-1",
                }
            ],
            "pools": [
                {
                    "pool_id": "pool-1",
                    "rna_url": "https://example.org/assay.csv.gz",
                    "rna_file": {"bytes": 4, "sha256": None},
                }
            ],
        },
    )
    confirmation._append_jsonl(
        access,
        {
            "schema": "gse185381-aml-assay-access/1.0",
            "stage": "source",
            "status": "OPENED_BEFORE_ASSAY_ACCESS",
            "created_at_utc": "2026-08-29T00:00:00Z",
            "protocol_commit": "p" * 40,
            "runtime_environment": {"runtime": "frozen"},
        },
        create=True,
    )
    monkeypatch.setattr(confirmation, "urlopen", lambda url: io.BytesIO(b"abc"))
    with pytest.raises(
        confirmation.ConfirmationRefusal, match="DOWNLOADED_FILE_SIZE_DIFFERS"
    ):
        confirmation._fetch(
            "https://example.org/assay.csv.gz",
            destination,
            {
                "bytes": 4,
                "sha256": None,
                "sha256_policy": confirmation.ASSAY_SHA256_POLICIES["rna"],
            },
            authorized_modality="rna",
            authorized_stage="source",
            access_stage="source",
            pool_id="pool-1",
        )
    records = confirmation._access_records("source")
    assert records[1]["observed_bytes"] == 3
    assert records[1]["observed_sha256"] == hashlib.sha256(b"abc").hexdigest()


def test_classical_refusal_codes_preserve_method_and_reason():
    first = confirmation._classical_refusal_record(
        "pooled_saturated_poisson",
        confirmation.CouplingEstimationRefusal("zero cell"),
    )
    second = confirmation._classical_refusal_record(
        "pooled_saturated_poisson",
        confirmation.CouplingEstimationRefusal("nonfinite interaction"),
    )
    assert first["code"] == "POOLED_SATURATED_POISSON__ZERO_CELL"
    assert second["code"] == (
        "POOLED_SATURATED_POISSON__NONFINITE_INTERACTION"
    )
    assert first != second


def test_public_payload_rejects_private_states_and_local_paths():
    confirmation._validate_public_payload({"sha256": "a" * 64})
    with pytest.raises(PermissionError, match="private key"):
        confirmation._validate_public_payload({"states": [[0, 1]]})
    with pytest.raises(PermissionError, match="local path"):
        confirmation._validate_public_payload({"artifact": "/tmp/private.json"})


def test_private_paths_inside_repository_are_refused(tmp_path: Path):
    with pytest.raises(PermissionError, match="outside"):
        confirmation._private_path(confirmation.ROOT / "private.json")
    assert (
        confirmation._private_path(tmp_path / "private.json")
        == (tmp_path / "private.json").resolve()
    )
