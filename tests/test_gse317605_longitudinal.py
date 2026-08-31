from __future__ import annotations

import copy
import gzip
import hashlib
import io
from itertools import product
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import confirm_gse317605_longitudinal as runner
from experiments import gse317605_longitudinal_core as core
from mapreg.streamed_gzip_matrix_market import reduce_gzip_matrix_market


def _manifest() -> dict[str, object]:
    return json.loads(runner.MANIFEST.read_text(encoding="utf-8"))


def _candidate() -> dict[str, object]:
    return json.loads(runner.CANDIDATE.read_text(encoding="utf-8"))


def _patch_stage_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    paths = {}
    for stage in runner.STAGES:
        paths[stage] = {
            "attempt": tmp_path / f"{stage}_attempt.json",
            "consumption": tmp_path / f"{stage}_consumption.json",
            "journal": tmp_path / f"{stage}_access.jsonl",
            "result": tmp_path / f"{stage}_result.json",
        }
    monkeypatch.setattr(runner, "STAGE_PATHS", paths)
    monkeypatch.setattr(runner, "STAGE_LOCK_DIRECTORY", tmp_path / "locks")


def _losses(patient_order: tuple[str, ...]) -> dict[str, dict[str, object]]:
    means = {
        "primary": 0.80,
        "classical_time_conditioned_ridge_poisson": 1.00,
        "fixed_margin_independence": 1.15,
        "destroyed_link": 1.10,
        "graph_zero_retuned_exact_coupling": 0.95,
    }
    return {
        method: {
            "mean": value,
            "by_patient": {
                patient: value + 0.001 * index
                for index, patient in enumerate(patient_order)
            },
            "by_timepoint": {
                timepoint: value + 0.001 * index
                for index, timepoint in enumerate(core.TIMEPOINTS)
            },
        }
        for method, value in means.items()
    }


def _matrix_payload(
    rows: int,
    columns: int,
    entries: list[str],
    *,
    banner: str = "%%MatrixMarket matrix coordinate integer general",
) -> bytes:
    text = "\n".join(
        [banner, "% fixture", f"{rows} {columns} {len(entries)}", *entries, ""]
    )
    return gzip.compress(text.encode("ascii"), mtime=0)


def test_complete_patient_hash_split_and_partial_axis_are_exact() -> None:
    candidate = _candidate()
    split = candidate["split"]
    complete = [
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "19",
        "22",
        "23",
        "24",
        "25",
        "26",
        "27",
    ]
    ordered = sorted(
        complete,
        key=lambda patient: hashlib.sha256(
            f"GSE317605-PATIENT-SPLIT-v1|COMPLETE|{patient}".encode()
        ).hexdigest(),
    )
    assert (
        ordered[:7]
        == split["calibration"]
        == list(runner.EXPECTED_STAGE_PATIENTS["calibration"])
    )
    assert (
        ordered[7:10]
        == split["pilot"]
        == list(runner.EXPECTED_STAGE_PATIENTS["pilot_gex"])
    )
    assert ordered[10:] == split["held_complete"]
    assert split["held_partial"] == {
        "18": ["T01", "T02", "T03"],
        "20": ["T01", "T02", "T04"],
        "21": ["T01", "T03"],
    }
    assert split["counts"] == {"calibration": 7, "pilot": 3, "held": 8}


def test_manifest_has_exact_patient_visit_and_replicate_universe() -> None:
    manifest = _manifest()
    assert {
        key: manifest["summary"][key]
        for key in (
            "physical_patients",
            "complete_patients",
            "partial_patients",
            "paired_modality_records",
            "matrix_files_designated",
        )
    } == {
        "physical_patients": 18,
        "complete_patients": 15,
        "partial_patients": 3,
        "paired_modality_records": 84,
        "matrix_files_designated": 168,
    }
    assert manifest["summary"]["patient_timepoint_visits"] == 68
    assert manifest["summary"]["pooled_visits_with_at_least_192_rows"] == 68
    assert (
        manifest["summary"][
            "replicate_axes_with_deposited_duplicate_barcode_strings"
        ]
        == 14
    )
    assert [record["patient_id"] for record in manifest["patients"]] == [
        *runner.EXPECTED_STAGE_PATIENTS["calibration"],
        *runner.EXPECTED_STAGE_PATIENTS["pilot_gex"],
        *runner.EXPECTED_STAGE_PATIENTS["held_gex"],
    ]
    for patient in manifest["patients"]:
        grouped: dict[str, set[str]] = {}
        for record in patient["replicates"]:
            grouped.setdefault(record["timepoint"], set()).update(record.keys())
            assert set(record) == {"timepoint", "replicate", "GEX", "ADT"}
            for modality in ("GEX", "ADT"):
                assert set(record[modality]["files"]) == {
                    "barcodes",
                    "features",
                    "matrix",
                }
        assert list(grouped) == patient["timepoints"]
    matrix_names = {
        record[modality]["files"]["matrix"]["name"]
        for patient in manifest["patients"]
        for record in patient["replicates"]
        for modality in ("GEX", "ADT")
    }
    assert len(matrix_names) == 168


def test_feature_axes_marker_coordinates_and_salts_are_frozen() -> None:
    candidate = _candidate()
    assert candidate["feature_axes"] == {
        "coordinate_convention": "one-based deposited Matrix Market row index",
        "GEX": {
            "rows": 33538,
            "sha256": "6bb91dd583b8ed7e4d6ea2efb6cb9b103b229573fec7ba2c1f7ba583994a21b1",
        },
        "ADT": {
            "rows": 99,
            "sha256": "95797e25f128965db196858b0abf9a56487894431c5b792e54bbb53ccddfa1da",
        },
    }
    assert tuple(row["rna_symbol"] for row in candidate["markers"]) == core.MARKERS
    assert tuple(row["adt_id"] for row in candidate["markers"]) == core.ADT_MARKERS
    assert runner.GEX_SELECTED_ROWS == tuple(row["rna_row"] for row in candidate["markers"])
    assert runner.ADT_SELECTED_ROWS == tuple(row["adt_row"] for row in candidate["markers"])
    assert runner.CELL_SALT == "GSE317605-CELL-v2"
    assert runner.ADT_TIE_SALT == "GSE317605-ADT-TIE-v2"
    assert candidate["cell_contract"]["cell_selection_salt"] == runner.CELL_SALT
    assert candidate["cell_contract"]["adt_tie_salt"] == runner.ADT_TIE_SALT
    assert runner.CELL_BUDGET == 192
    assert runner.ADT_HIGH_COUNT == 96


def test_fixed_overlapping_hypergraph_and_operators_are_exact() -> None:
    incidence = core.marker_incidence()
    assert incidence.shape == (16, 5)
    marker_index = {marker: index for index, marker in enumerate(core.MARKERS)}
    for edge, (_, members) in enumerate(core.HYPEREDGES):
        expected = {marker_index[marker] for marker in members}
        assert set(np.flatnonzero(incidence[:, edge])) == expected
    assert int(incidence[marker_index["CD27"]].sum()) == 3
    marker = core.marker_laplacian()
    pair = core.pair_laplacian()
    temporal = core.temporal_laplacian()
    assert np.allclose(marker, marker.T)
    assert np.linalg.eigvalsh(marker)[0] >= -1e-10
    expected_pair = np.kron(marker, np.eye(16)) + np.kron(np.eye(16), marker)
    np.testing.assert_allclose(pair, expected_pair)
    np.testing.assert_array_equal(
        temporal,
        [[1, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 2, -1], [0, 0, -1, 1]],
    )


def test_direct_coupling_fit_converges_and_recovers_temporal_direction() -> None:
    templates = np.asarray(
        [
            [[42, 8], [8, 42]],
            [[34, 16], [16, 34]],
            [[16, 34], [34, 16]],
            [[8, 42], [42, 8]],
        ],
        dtype=np.int64,
    )
    tables = np.stack(
        [
            np.broadcast_to(table, (len(core.MARKERS), len(core.MARKERS), 2, 2))
            for table in templates
        ]
    ).copy()
    configuration = core.CouplingConfig(10.0, 0.05, 0.05, 1.0)
    fitted = core.fit_coupling_field(
        tables,
        core.TIMEPOINTS,
        configuration,
        maximum_iterations=250,
        gradient_tolerance=1e-6,
    )
    time_means = fitted.population_log_odds.mean(axis=(1, 2))
    assert fitted.optimizer == "direct_lbfgsb_exact_conditional_sparse_penalty"
    assert fitted.iterations > 0
    assert fitted.gradient_norm <= 5e-6
    assert np.all(np.diff(time_means) < 0.0)
    assert time_means[0] > 0.0 > time_means[-1]


def test_poisson_comparator_never_uses_conditional_reconstruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert core.MANDATORY_METHODS[1] == "classical_time_conditioned_ridge_poisson"
    assert "matched_time_conditioned_ridge_poisson" not in core.MANDATORY_METHODS
    candidate = _candidate()
    protocol = json.loads(runner.PROTOCOL.read_text(encoding="utf-8"))
    assert (
        candidate["estimators"]["mandatory_comparators"][0]
        == "classical_time_conditioned_ridge_poisson"
    )
    assert (
        "classical_time_conditioned_ridge_poisson"
        in protocol["decision_replay"]["prediction_estimands"]
    )
    table = np.broadcast_to(
        np.asarray([[12, 18], [8, 62]], dtype=np.int64),
        (1, 16, 16, 2, 2),
    ).copy()
    primary = np.full((4, 16, 16), 0.4)
    poisson = np.full((4, 16, 16), 1.7)
    conditional_predict = core.predict_tables_at_observed_margins
    poisson_predict = core.predict_poisson_tables_at_observed_margins
    poisson_calls = 0

    def guarded_conditional(observed: np.ndarray, field: np.ndarray) -> np.ndarray:
        assert not np.array_equal(field, poisson[0]), (
            "Poisson comparator reached the conditional estimand"
        )
        return conditional_predict(observed, field)

    def tracked_poisson(
        observed: np.ndarray,
        field: np.ndarray,
        *,
        transport_scale: float = 1.0,
    ) -> np.ndarray:
        nonlocal poisson_calls
        poisson_calls += 1
        assert np.array_equal(field, poisson[0])
        return poisson_predict(observed, field, transport_scale=transport_scale)

    monkeypatch.setattr(core, "predict_tables_at_observed_margins", guarded_conditional)
    monkeypatch.setattr(
        core, "predict_poisson_tables_at_observed_margins", tracked_poisson
    )
    losses = core.losses_from_fields(
        table,
        ["23"],
        ["T01"],
        {
            "primary": primary,
            "classical_time_conditioned_ridge_poisson": poisson,
        },
    )
    assert poisson_calls == 1
    explicit = core.deviance_per_observation(
        table[0], poisson_predict(table[0], poisson[0])
    )
    assert losses["classical_time_conditioned_ridge_poisson"]["mean"] == pytest.approx(
        explicit
    )
    assert not np.allclose(
        poisson_predict(table[0], poisson[0]),
        conditional_predict(table[0], poisson[0]),
    )


def test_poisson_lopo_crosses_ridge_and_transport_grids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tables = np.broadcast_to(
        np.asarray([[12, 18], [8, 62]], dtype=np.int64),
        (28, 16, 16, 2, 2),
    ).copy()
    patients = tuple(str(index) for index in range(7) for _ in core.TIMEPOINTS)
    timepoints = core.TIMEPOINTS * 7
    field_by_ridge = {
        ridge: np.full((4, 16, 16), index + 1.0)
        for index, ridge in enumerate(core.POISSON_RIDGE_GRID)
    }
    fit_calls: list[float] = []
    reconstruction_calls: list[tuple[float, float]] = []

    def fake_fit(
        values: np.ndarray,
        labels: tuple[str, ...] | np.ndarray,
        *args: object,
        **kwargs: object,
    ) -> tuple[np.ndarray, dict[str, object]]:
        ridge = float(kwargs.get("ridge_penalty", args[0] if args else np.nan))
        assert ridge in core.POISSON_RIDGE_GRID
        fit_calls.append(ridge)
        return field_by_ridge[ridge], {"ridge_penalty": ridge}

    monkeypatch.setattr(core, "fit_time_conditioned_ridge_poisson", fake_fit)
    monkeypatch.setattr(
        core,
        "_visit_losses_for_field",
        lambda values, labels, fitted: np.full(len(values), 99.0),
    )

    def poisson_losses(
        values: np.ndarray,
        labels: tuple[str, ...] | np.ndarray,
        fitted: np.ndarray,
        *,
        transport_scale: float = 1.0,
    ) -> np.ndarray:
        ridge = next(
            value for value, field in field_by_ridge.items() if np.array_equal(fitted, field)
        )
        reconstruction_calls.append((ridge, transport_scale))
        return np.full(len(values), transport_scale + 1.0)

    monkeypatch.setattr(core, "_visit_losses_for_poisson", poisson_losses)
    classical, conditional = core._cv_poisson_candidates(
        tables, patients, timepoints
    )
    assert sorted(fit_calls) == sorted(list(core.POISSON_RIDGE_GRID) * 7)
    assert sorted(reconstruction_calls) == sorted(
        list(product(core.POISSON_RIDGE_GRID, core.TRANSPORT_GRID)) * 7
    )
    expected = set(product(core.POISSON_RIDGE_GRID, core.TRANSPORT_GRID))
    assert set(classical) == expected
    assert set(conditional) == expected
    assert all(row["mean"] == 99.0 for row in conditional.values())


def test_poisson_ridge_and_transport_selection_replay_exactly() -> None:
    exact: dict[core.CouplingConfig, dict[str, float]] = {}
    exact_ledger = []
    for base in core._base_configurations():
        for alpha in core.TRANSPORT_GRID:
            config = core._with_transport(base, alpha)
            loss = float(
                config.deviation_penalty
                + config.hypergraph_penalty
                + config.temporal_penalty
                + config.transport_multiplier
            )
            exact[config] = {"mean": loss}
            exact_ledger.append(
                {
                    "configuration": {
                        "deviation_penalty": config.deviation_penalty,
                        "hypergraph_penalty": config.hypergraph_penalty,
                        "temporal_penalty": config.temporal_penalty,
                        "transport_multiplier": config.transport_multiplier,
                    },
                    "losses": {"mean": loss},
                }
            )

    selected_primary, _ = core._selection_record(exact)
    selected_graph_zero, _ = core._selection_record(
        {config: row for config, row in exact.items() if config.hypergraph_penalty == 0}
    )
    selected_temporal_zero, _ = core._selection_record(
        {config: row for config, row in exact.items() if config.temporal_penalty == 0}
    )
    selected_structure_zero, _ = core._selection_record(
        {
            config: row
            for config, row in exact.items()
            if config.hypergraph_penalty == config.temporal_penalty == 0
        }
    )

    target = (core.POISSON_RIDGE_GRID[-1], core.TRANSPORT_GRID[-1])
    poisson_ledger = []
    for ridge, alpha in product(core.POISSON_RIDGE_GRID, core.TRANSPORT_GRID):
        poisson_ledger.append(
            {
                "ridge_penalty": ridge,
                "transport_multiplier": alpha,
                "losses": {"mean": 0.0 if (ridge, alpha) == target else 1.0},
                "conditional_reconstruction_losses": {"mean": 2.0},
            }
        )

    def config_dict(config: core.CouplingConfig) -> dict[str, float]:
        return {
            "deviation_penalty": config.deviation_penalty,
            "hypergraph_penalty": config.hypergraph_penalty,
            "temporal_penalty": config.temporal_penalty,
            "transport_multiplier": config.transport_multiplier,
        }

    selection = {
        "exact_cv_ledger": exact_ledger,
        "poisson_cv_ledger": poisson_ledger,
        "selected_primary": config_dict(selected_primary),
        "selected_graph_zero": config_dict(selected_graph_zero),
        "selected_temporal_zero": config_dict(selected_temporal_zero),
        "selected_structure_zero": config_dict(selected_structure_zero),
        "selected_poisson_ridge": target[0],
        "selected_poisson_transport": target[1],
    }
    replay = core.replay_calibration_selection(selection)
    assert replay["selected_poisson_ridge"] == target[0]
    assert replay["selected_poisson_transport"] == target[1]
    tampered = copy.deepcopy(selection)
    tampered["selected_poisson_ridge"] = core.POISSON_RIDGE_GRID[0]
    with pytest.raises(ValueError, match="differs from the CV ledgers"):
        core.replay_calibration_selection(tampered)


def test_frozen_refit_applies_transport_once_and_reuses_selected_poisson_ridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fitted_transports: list[float] = []
    poisson_ridges: list[float] = []
    base_field = np.ones((len(core.TIMEPOINTS), len(core.MARKERS), len(core.MARKERS)))

    def fake_coupling(
        tables: np.ndarray,
        timepoints: tuple[str, ...],
        configuration: core.CouplingConfig,
    ) -> core.CouplingFit:
        fitted_transports.append(configuration.transport_multiplier)
        return core.CouplingFit(
            population_log_odds=base_field.copy(),
            donor_timepoint_deviation=np.zeros(
                (len(tables), len(core.MARKERS), len(core.MARKERS))
            ),
            objective=1.0,
            gradient_norm=1e-9,
            iterations=2,
            configuration=configuration,
            optimizer="fixture",
        )

    def fake_poisson(
        tables: np.ndarray,
        timepoints: tuple[str, ...],
        *,
        ridge_penalty: float,
    ) -> tuple[np.ndarray, dict[str, float]]:
        poisson_ridges.append(ridge_penalty)
        return 3.0 * base_field, {"ridge_penalty": ridge_penalty}

    monkeypatch.setattr(core, "fit_coupling_field", fake_coupling)
    monkeypatch.setattr(core, "fit_time_conditioned_ridge_poisson", fake_poisson)
    primary = {
        "deviation_penalty": 0.1,
        "hypergraph_penalty": 0.05,
        "temporal_penalty": 0.05,
        "transport_multiplier": 1.25,
    }
    graph_zero = {**primary, "hypergraph_penalty": 0.0, "transport_multiplier": 0.5}
    temporal_zero = {**primary, "temporal_penalty": 0.0, "transport_multiplier": 0.75}
    structure_zero = {
        **primary,
        "hypergraph_penalty": 0.0,
        "temporal_penalty": 0.0,
        "transport_multiplier": 1.0,
    }
    tables = np.zeros((4, len(core.MARKERS), len(core.MARKERS), 2, 2), dtype=int)
    models = core.fit_frozen_models(
        tables,
        tables,
        core.TIMEPOINTS,
        primary,
        graph_zero,
        temporal_zero,
        structure_zero,
        0.1,
        0.75,
    )
    assert fitted_transports == [1.0] * 5
    assert poisson_ridges == [0.1]
    np.testing.assert_allclose(models["fields"]["primary"], 1.25)
    np.testing.assert_allclose(
        models["fields"]["classical_time_conditioned_ridge_poisson"], 2.25
    )
    assert models["selected_poisson_ridge"] == 0.1
    assert models["selected_poisson_transport"] == 0.75


def test_promoted_source_refit_forwards_the_complete_frozen_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_stage_paths(monkeypatch, tmp_path)
    configuration = {
        "deviation_penalty": 0.1,
        "hypergraph_penalty": 0.05,
        "temporal_penalty": 0.05,
        "transport_multiplier": 1.0,
    }
    selection = {
        "selected_primary": configuration,
        "selected_graph_zero": {**configuration, "hypergraph_penalty": 0.0},
        "selected_temporal_zero": {**configuration, "temporal_penalty": 0.0},
        "selected_structure_zero": {
            **configuration,
            "hypergraph_penalty": 0.0,
            "temporal_penalty": 0.0,
        },
        "selected_poisson_ridge": 0.1,
        "selected_poisson_transport": 0.75,
    }
    table = np.broadcast_to(
        np.asarray([[12, 18], [8, 62]], dtype=np.int64),
        (1, len(core.MARKERS), len(core.MARKERS), 2, 2),
    ).copy()
    runner._write_json_x(
        runner.STAGE_PATHS["calibration"]["result"],
        {
            "tables": table.tolist(),
            "destroyed_tables": table.tolist(),
            "visit_timepoint_axis": ["T01"],
            "selection": selection,
        },
    )
    monkeypatch.setattr(runner, "_load_gex_bridge", lambda *args: [])
    monkeypatch.setattr(runner, "_acquire_modality", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        runner,
        "_joint_panels",
        lambda *args: (table, table, ["13"], ["T01"], []),
    )
    monkeypatch.setattr(runner, "_fields", lambda models: {})
    monkeypatch.setattr(runner, "_descriptive_fields", lambda models: {})
    monkeypatch.setattr(core, "losses_from_fields", lambda *args: _losses(("13",)))
    monkeypatch.setattr(core, "pilot_gate", lambda *args: {"passes": True})
    refit_calls: list[tuple[object, ...]] = []

    def frozen_refit(*args: object) -> dict[str, object]:
        refit_calls.append(args)
        return {"status": "fixture"}

    monkeypatch.setattr(core, "fit_frozen_models", frozen_refit)
    result = runner._run_pilot_adt(tmp_path / "scratch", {}, {"models": {}})
    assert result["status"] == "PILOT_PASS"
    assert len(refit_calls) == 1
    assert refit_calls[0][3:] == (
        selection["selected_primary"],
        selection["selected_graph_zero"],
        selection["selected_temporal_zero"],
        selection["selected_structure_zero"],
        selection["selected_poisson_ridge"],
        selection["selected_poisson_transport"],
    )


def test_partial_patient_visits_are_averaged_before_equal_patient_aggregation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tables = np.broadcast_to(
        np.asarray([[12, 18], [8, 62]], dtype=np.int64),
        (3, len(core.MARKERS), len(core.MARKERS), 2, 2),
    ).copy()
    monkeypatch.setattr(
        core,
        "_visit_losses_for_field",
        lambda *args, **kwargs: np.asarray([1.0, 3.0, 9.0]),
    )
    losses = core.losses_from_fields(
        tables,
        ("complete", "complete", "partial"),
        ("T01", "T02", "T01"),
        {"primary": np.zeros((4, len(core.MARKERS), len(core.MARKERS)))},
    )["primary"]
    assert losses["by_patient"] == {"complete": 2.0, "partial": 9.0}
    assert losses["mean"] == 5.5
    assert losses["mean"] != pytest.approx((1.0 + 3.0 + 9.0) / 3.0)


def test_matrix_market_reduction_uses_exact_frozen_coordinates() -> None:
    gex = _matrix_payload(
        runner.GEX_ROWS,
        3,
        [
            f"{runner.GEX_SELECTED_ROWS[0]} 3 7",
            f"{runner.GEX_SELECTED_ROWS[-1]} 1 11",
            "1 2 13",
        ],
    )
    block, audit = reduce_gzip_matrix_market(
        io.BytesIO(gex),
        expected_shape=(runner.GEX_ROWS, 3),
        selected_rows=(runner.GEX_SELECTED_ROWS[0], runner.GEX_SELECTED_ROWS[-1]),
        selected_columns=(1, 3),
        allow_integral_real=True,
    )
    np.testing.assert_array_equal(block, [[0, 7], [11, 0]])
    assert audit.parsed_nnz == audit.declared_nnz == 3
    assert audit.gzip_stream_exhausted

    adt = _matrix_payload(
        runner.ADT_ROWS,
        2,
        [f"{runner.ADT_SELECTED_ROWS[0]} 2 5.0"],
        banner="%%MatrixMarket matrix coordinate real general",
    )
    block, audit = reduce_gzip_matrix_market(
        io.BytesIO(adt),
        expected_shape=(runner.ADT_ROWS, 2),
        selected_rows=(runner.ADT_SELECTED_ROWS[0],),
        selected_columns=(2,),
        allow_integral_real=True,
    )
    assert block.tolist() == [[5]]
    assert audit.banner.endswith("real general")


def test_feature_axis_is_revalidated_before_matrix_reduction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rows = ("ENSG1\tA\tGene Expression", "ENSG2\tB\tGene Expression")
    expected = runner._feature_axis_sha256(rows)
    monkeypatch.setattr(runner, "GEX_ROWS", 2)
    monkeypatch.setattr(runner, "GEX_FEATURE_AXIS_SHA256", expected)
    path = tmp_path / "features.tsv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        stream.write("\n".join(rows) + "\n")
    assert runner._features(path, "GEX")["ordered_axis_sha256"] == expected

    with gzip.open(path, "wt", encoding="utf-8") as stream:
        stream.write("ENSG2\tB\tGene Expression\nENSG1\tA\tGene Expression\n")
    with pytest.raises(runner.ProtocolRefusal, match="FEATURE_AXIS_DIFFERS"):
        runner._features(path, "GEX")


def test_deposited_duplicate_barcodes_are_valid_axis_rows(tmp_path: Path) -> None:
    path = tmp_path / "barcodes.tsv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        stream.write("DUPLICATE\nDUPLICATE\nUNIQUE\n")
    values, audit = runner._barcodes(path)
    assert values == ("DUPLICATE", "DUPLICATE", "UNIQUE")
    assert audit["rows"] == 3
    assert audit["ordered_axis_sha256"] == runner._barcode_axis_sha256(values)


def test_bundled_archive_is_absent_and_download_refuses_it_before_get(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = _manifest()
    assert manifest["series_archive_permitted"] is False
    assert manifest["series_archive_name"] == runner.SERIES_ARCHIVE
    assert all(
        ".tar" not in file_record["name"].lower()
        and ".tar" not in file_record["url"].lower()
        for patient in manifest["patients"]
        for visit in patient["replicates"]
        for modality in ("GEX", "ADT")
        for file_record in visit[modality]["files"].values()
    )
    monkeypatch.setattr(
        runner,
        "_open_url",
        lambda request: (_ for _ in ()).throw(AssertionError("network reached")),
    )
    with pytest.raises(PermissionError, match="per-sample frozen contract"):
        runner._download(
            "calibration",
            {
                "name": runner.SERIES_ARCHIVE,
                "url": "https://ftp.ncbi.nlm.nih.gov/geo/GSE317605_RAW.tar",
                "bytes": 659916800,
            },
            tmp_path / "archive.tar",
            patient_id="23",
            timepoint="T01",
            replicate="1",
            modality="GEX",
            kind="matrix",
        )


def test_stage_file_sets_enforce_modality_and_patient_boundaries() -> None:
    manifest = _manifest()
    by_stage = {
        stage: runner._stage_file_records(stage, manifest) for stage in runner.STAGES
    }
    for stage, rows in by_stage.items():
        assert {row[0] for row in rows} == set(runner.EXPECTED_STAGE_PATIENTS[stage])
        assert {row[3] for row in rows} == set(runner.STAGE_MODALITIES[stage])
        assert all(row[4]["name"] != runner.SERIES_ARCHIVE for row in rows)
    pilot_gex = {row[4]["name"] for row in by_stage["pilot_gex"]}
    pilot_adt = {row[4]["name"] for row in by_stage["pilot_adt"]}
    held_gex = {row[4]["name"] for row in by_stage["held_gex"]}
    held_adt = {row[4]["name"] for row in by_stage["held_adt"]}
    assert pilot_gex.isdisjoint(pilot_adt | held_gex | held_adt)
    assert pilot_adt.isdisjoint(held_gex | held_adt)
    assert held_gex.isdisjoint(held_adt)
    with pytest.raises(PermissionError, match="ADT is forbidden"):
        runner._acquire_modality("pilot_gex", Path("/unused"), manifest, "ADT")
    with pytest.raises(PermissionError, match="GEX is forbidden"):
        runner._acquire_modality("held_adt", Path("/unused"), manifest, "GEX")


def test_cell_selection_uses_replicate_column_and_barcode_identity() -> None:
    axes = {
        "1": ("DUP", "DUP", *(f"A{index:03d}" for index in range(108))),
        "2": ("DUP", "DUP", *(f"B{index:03d}" for index in range(108))),
    }
    first = runner._selected_cells("23", "T01", axes)
    second = runner._selected_cells("23", "T01", axes)
    assert first == second
    identities = set(first)
    assert len(first) == len(identities) == 192
    assert sum(barcode == "DUP" for _, _, barcode in first) >= 2
    assert all(isinstance(column, int) and column >= 1 for _, column, _ in first)
    assert first == tuple(
        sorted(
            first,
            key=lambda value: (
                runner._cell_rank(
                    "23",
                    "T01",
                    value[0],
                    value[1],
                    value[2],
                ),
                value[0],
                value[1],
                value[2],
            ),
        )
    )
    counts = np.ones((192, 16), dtype=np.int64)
    visit = {
        "patient_id": "23",
        "timepoint": "T01",
        "selected_cells": [
            {"replicate": replicate, "column": column, "barcode": barcode}
            for replicate, column, barcode in first
        ],
        "counts": counts,
    }
    states = runner._adt_states(visit)
    assert states.shape == (192, 16)
    assert np.all(states.sum(axis=0) == 96)
    np.testing.assert_array_equal(states, runner._adt_states(visit))


def test_public_gex_artifact_excludes_identifiers_and_cell_states(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    selected = [
        {
            "replicate": "1",
            "column": index + 1,
            "barcode": f"CELL-{index:03d}",
        }
        for index in range(runner.CELL_BUDGET)
    ]
    raw = {
        "patient_id": "13",
        "timepoint": "T01",
        "selected_cells": selected,
        "selected_cell_axis_sha256": "a" * 64,
        "barcode_axes": {"1": "b" * 64},
        "feature_audit": {"1": {"rows": runner.GEX_ROWS}},
        "barcode_audit": {"1": {"rows": runner.CELL_BUDGET}},
        "matrix_audit": {"1": {"parsed_nnz": 10}},
        "counts": np.ones((runner.CELL_BUDGET, len(core.MARKERS)), dtype=np.int64),
        "counts_sha256": "c" * 64,
    }
    public = runner._gex_public_visit(raw)
    assert "selected_cells" not in public
    assert "rna_states" not in public
    assert public["rna_high_margins"] == [runner.CELL_BUDGET] * len(core.MARKERS)

    bridge_path = tmp_path / "private" / "pilot.json"
    monkeypatch.setattr(runner, "PRIVATE_GEX_BRIDGES", {"pilot_gex": bridge_path})
    binding = runner._write_gex_bridge("pilot_gex", [raw], "d" * 64)
    prediction = {
        "models_sha256": "d" * 64,
        "private_gex_bridge": binding,
    }
    loaded = runner._load_gex_bridge(prediction, "pilot_gex")
    assert loaded[0]["selected_cells"] == selected
    assert loaded[0]["rna_states"] == np.ones(
        (runner.CELL_BUDGET, len(core.MARKERS)), dtype=np.int8
    ).tolist()
    assert bridge_path.stat().st_mode & 0o777 == 0o600


def test_joint_panel_audit_retains_gex_and_adt_lineage() -> None:
    selected = [
        {"replicate": "1", "column": index + 1, "barcode": f"CELL-{index}"}
        for index in range(runner.CELL_BUDGET)
    ]
    states = np.zeros((runner.CELL_BUDGET, len(core.MARKERS)), dtype=np.int8)
    states[: runner.CELL_BUDGET // 2] = 1
    axis_sha256 = "a" * 64
    gex = {
        "patient_id": "23",
        "timepoint": "T01",
        "selected_cells": selected,
        "selected_cell_axis_sha256": axis_sha256,
        "rna_states": states.tolist(),
        "rna_states_sha256": "b" * 64,
        "counts_sha256": "c" * 64,
        "feature_audit": {"1": {"rows": runner.GEX_ROWS}},
        "barcode_audit": {"1": {"rows": runner.CELL_BUDGET}},
        "matrix_audit": {"1": {"parsed_nnz": 10}},
    }
    adt = {
        "patient_id": "23",
        "timepoint": "T01",
        "selected_cells": selected,
        "selected_cell_axis_sha256": axis_sha256,
        "counts": np.ones((runner.CELL_BUDGET, len(core.MARKERS)), dtype=np.int64),
        "counts_sha256": "d" * 64,
        "feature_audit": {"1": {"rows": runner.ADT_ROWS}},
        "barcode_audit": {"1": {"rows": runner.CELL_BUDGET}},
        "matrix_audit": {"1": {"parsed_nnz": 20}},
    }
    _, _, _, _, audits = runner._joint_panels([gex], [adt])
    audit = audits[0]
    assert audit["rna_states_sha256"] == gex["rna_states_sha256"]
    assert audit["gex_counts_sha256"] == gex["counts_sha256"]
    assert audit["gex_feature_audit"] == gex["feature_audit"]
    assert audit["gex_barcode_audit"] == gex["barcode_audit"]
    assert audit["gex_matrix_audit"] == gex["matrix_audit"]
    assert audit["adt_feature_audit"] == adt["feature_audit"]


def test_stage_attempt_binds_scratch_without_publishing_local_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_stage_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(
        runner,
        "_verify_implementation",
        lambda: {"public_commit": "a" * 40},
    )
    monkeypatch.setattr(runner, "_verify_dependency", lambda stage: None)
    token = tmp_path / "private" / "token.bin"
    scratch = tmp_path / "person-specific" / "scratch"
    attempt = runner.claim_stage("calibration", token, scratch)
    encoded = json.dumps(attempt)
    assert str(scratch.resolve()) not in encoded
    assert attempt["scratch_binding_sha256"] == runner._path_binding(scratch)


def test_implementation_freeze_refuses_preexisting_stage_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_stage_paths(monkeypatch, tmp_path)
    freeze = tmp_path / "implementation_freeze.json"
    monkeypatch.setattr(runner, "IMPLEMENTATION_FREEZE", freeze)
    monkeypatch.setattr(runner, "IMPLEMENTATION_BINDINGS", ())
    monkeypatch.setattr(runner, "PRIVATE_GEX_BRIDGES", {})
    monkeypatch.setattr(runner, "_validate_contract", lambda: ({}, {}, {}))
    monkeypatch.setattr(runner, "_require_public_tag", lambda *args: "a" * 40)
    monkeypatch.setattr(
        runner,
        "_git",
        lambda *args: SimpleNamespace(stdout="b" * 40 + "\n"),
    )
    runner.STAGE_PATHS["calibration"]["attempt"].write_text("{}\n", encoding="utf-8")
    with pytest.raises(PermissionError):
        runner.freeze_implementation()
    assert not freeze.exists()


def test_claim_rolls_back_token_attempt_and_journal_on_journal_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_stage_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(
        runner,
        "_verify_implementation",
        lambda: {"public_commit": "a" * 40},
    )
    monkeypatch.setattr(runner, "_verify_dependency", lambda stage: None)
    monkeypatch.setattr(
        runner,
        "_create_journal",
        lambda *args: (_ for _ in ()).throw(OSError("journal fsync failed")),
    )
    token = tmp_path / "private" / "token.bin"
    with pytest.raises(OSError, match="journal fsync failed"):
        runner.claim_stage("calibration", token, tmp_path / "scratch")
    assert not token.exists()
    assert all(
        not path.exists() for path in runner.STAGE_PATHS["calibration"].values()
    )


def test_public_download_failure_journal_has_no_message_or_local_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_stage_paths(monkeypatch, tmp_path)
    journal = runner.STAGE_PATHS["calibration"]["journal"]
    runner._create_journal(journal, "calibration", "2026-08-30T00:00:00Z")
    secret = tmp_path / "person-specific" / "secret.txt"
    monkeypatch.setattr(
        runner,
        "_open_url",
        lambda request: (_ for _ in ()).throw(RuntimeError(f"failed at {secret}")),
    )
    with pytest.raises(RuntimeError):
        runner._download(
            "calibration",
            {
                "name": "matrix.mtx.gz",
                "url": "https://ftp.ncbi.nlm.nih.gov/matrix.mtx.gz",
                "bytes": 10,
            },
            tmp_path / "scratch" / "matrix.mtx.gz",
            patient_id="23",
            timepoint="T01",
            replicate="1",
            modality="GEX",
            kind="matrix",
        )
    events = runner._read_jsonl(journal)
    failure = events[-1]
    assert failure["event"] == "FILE_GET_FAILED"
    assert "message" not in failure
    public_bytes = journal.read_text(encoding="utf-8")
    assert str(tmp_path.resolve()) not in public_bytes
    assert "person-specific" not in public_bytes


def test_stage_claim_and_consumption_are_exclusive_and_durable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_stage_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(
        runner,
        "_verify_implementation",
        lambda: {"public_commit": "a" * 40},
    )
    monkeypatch.setattr(runner, "_verify_dependency", lambda stage: None)
    token = tmp_path / "private" / "token.bin"
    scratch = tmp_path / "scratch"
    attempt = runner.claim_stage("calibration", token, scratch)
    assert attempt["status"] == "CLAIMED_BEFORE_FIRST_STAGE_FILE_GET"
    assert token.is_file() and token.stat().st_size == 32
    assert (
        runner._read_jsonl(runner.STAGE_PATHS["calibration"]["journal"])[0]["event"]
        == "OPENED_BEFORE_ASSAY_ACCESS"
    )
    with pytest.raises(FileExistsError, match="durable artifact"):
        runner.claim_stage("calibration", tmp_path / "second-token", scratch)

    monkeypatch.setattr(runner, "_require_public_tag", lambda tag, paths: "b" * 40)
    consumption = runner._consume_stage("calibration", token, scratch)
    assert consumption["consumed_before_first_file_get"] is True
    assert not token.exists()
    assert runner.STAGE_PATHS["calibration"]["consumption"].is_file()
    with pytest.raises(FileExistsError, match="already consumed"):
        runner._consume_stage("calibration", token, scratch)


def test_generic_failure_after_capability_consumption_is_terminal_and_sanitized(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_stage_paths(monkeypatch, tmp_path)
    implementation = {"public_commit": "a" * 40}
    monkeypatch.setattr(runner, "_verify_implementation", lambda: implementation)
    monkeypatch.setattr(runner, "_verify_dependency", lambda stage: None)
    monkeypatch.setattr(runner, "_require_public_tag", lambda *args: "b" * 40)
    monkeypatch.setattr(runner, "_validate_contract", lambda: ({}, {}, {}))
    token = tmp_path / "private" / "token.bin"
    scratch = tmp_path / "person-specific" / "scratch"
    runner.claim_stage("calibration", token, scratch)
    monkeypatch.setattr(
        runner,
        "_run_calibration",
        lambda *args: (_ for _ in ()).throw(
            RuntimeError(f"unexpected failure below {scratch}")
        ),
    )
    result = runner.run_stage("calibration", token, scratch)
    assert result["status"] == "TERMINAL_REFUSAL"
    assert result["rerun_permitted"] is False
    assert result["refusal_code"] == "UNEXPECTED_RuntimeError"
    assert not token.exists()
    assert runner.STAGE_PATHS["calibration"]["consumption"].is_file()
    assert runner.STAGE_PATHS["calibration"]["result"].is_file()
    encoded = json.dumps(result, sort_keys=True)
    assert str(tmp_path.resolve()) not in encoded
    assert "unexpected failure" not in encoded


def test_success_validation_reconciles_exact_file_journal_and_public_result_tag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_stage_paths(monkeypatch, tmp_path)
    stage = "pilot_gex"
    paths = runner.STAGE_PATHS[stage]
    runner._write_json_x(
        paths["attempt"],
        {"schema": "gse317605-stage-attempt/1.0", "stage": stage},
    )
    runner._write_json_x(
        paths["consumption"],
        {"schema": "gse317605-stage-consumption/1.0", "stage": stage},
    )
    runner._create_journal(paths["journal"], stage, "2026-08-30T00:00:00Z")
    identity = {
        "stage": stage,
        "patient_id": "13",
        "timepoint": "T01",
        "replicate": "1",
        "modality": "GEX",
        "kind": "matrix",
        "name": "matrix.mtx.gz",
        "url": "https://ftp.ncbi.nlm.nih.gov/matrix.mtx.gz",
    }
    for event in (
        {"stage": stage, "event": "CAPABILITY_CONSUMED"},
        {**identity, "event": "FILE_GET_STARTED"},
        {
            **identity,
            "event": "FILE_GET_FINISHED",
            "bytes": 10,
            "sha256": "a" * 64,
            "http_status": 200,
            "final_url": identity["url"],
        },
        {**identity, "event": "FILE_DELETED", "body_existed": True},
    ):
        runner._append_jsonl(
            paths["journal"], {**event, "created_at_utc": "2026-08-30T00:00:01Z"}
        )
    result = {
        "schema": "gse317605-gex-predictions/1.0",
        "stage": stage,
        "status": "PREDICTIONS_FROZEN_BEFORE_PILOT_ADT_ACCESS",
        "attempt_sha256": runner._sha256(paths["attempt"]),
        "consumption_sha256": runner._sha256(paths["consumption"]),
        "access_journal_sha256": runner._sha256(paths["journal"]),
        "access_ledger": {
            "expected_files": 1,
            "started_files": 1,
            "finished_files": 1,
            "failed_files": 0,
            "deleted_files": 1,
            "exact_manifest_reconciliation_passes": True,
        },
        "rerun_permitted": False,
    }
    runner._write_json_x(paths["result"], result)
    expected_key = (
        "13",
        "T01",
        "1",
        "GEX",
        "matrix",
        identity["name"],
    )
    monkeypatch.setattr(runner, "_validate_contract", lambda: ({}, {}, {}))
    monkeypatch.setattr(
        runner,
        "_verify_implementation",
        lambda: {"public_commit": "a" * 40},
    )
    monkeypatch.setattr(runner, "_verify_dependency", lambda requested_stage: None)
    monkeypatch.setattr(runner, "_require_ancestor", lambda ancestor, descendant: None)
    monkeypatch.setattr(
        runner,
        "_expected_stage_file_keys",
        lambda requested_stage, manifest: {expected_key},
    )
    tag_calls: list[tuple[str, tuple[Path, ...]]] = []

    def public_tag(tag: str, required_paths: tuple[Path, ...]) -> str:
        tag_calls.append((tag, tuple(required_paths)))
        return "b" * 40

    monkeypatch.setattr(runner, "_require_public_tag", public_tag)
    validated = runner.validate_stage(stage)
    assert validated["status"] == result["status"]
    assert any(
        tag == runner.RESULT_TAGS[stage] and paths["result"] in required
        for tag, required in tag_calls
    )

    runner._append_jsonl(
        paths["journal"],
        {**identity, "event": "FILE_GET_STARTED", "created_at_utc": "later"},
    )
    result["access_journal_sha256"] = runner._sha256(paths["journal"])
    paths["result"].write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(PermissionError, match="journal"):
        runner.validate_stage(stage)


def test_frozen_hashes_and_implementation_bindings_are_complete() -> None:
    assert runner._sha256(runner.CANDIDATE) == runner.CANDIDATE_SHA256
    assert runner._sha256(runner.PROTOCOL) == runner.PROTOCOL_SHA256
    assert runner._sha256(runner.MANIFEST) == runner.MANIFEST_SHA256
    assert runner._sha256(runner.ACCESS_HISTORY) == runner.ACCESS_HISTORY_SHA256
    assert len(runner.IMPLEMENTATION_BINDINGS) == len(
        set(runner.IMPLEMENTATION_BINDINGS)
    )
    assert all(
        (runner.ROOT / path).is_file() for path in runner.IMPLEMENTATION_BINDINGS
    )
    history = json.loads(runner.ACCESS_HISTORY.read_text(encoding="utf-8"))
    assert history["scope"]["matrix_market_headers_read"] == 0
    assert history["scope"]["matrix_market_entries_read"] == 0
    assert history["scope"]["matrix_bodies_downloaded"] == 0
    assert history["barcode_axis_audit"]["equal_pairs"] == 84
    assert history["barcode_axis_audit"]["unequal_pairs"] == 0


def test_held_gate_uses_exact_stratified_resamples_and_reports_reduction_ci() -> None:
    patients = runner.EXPECTED_STAGE_PATIENTS["held_adt"]
    completeness = {
        patient: "complete" if index < 5 else "partial"
        for index, patient in enumerate(patients)
    }
    losses = _losses(patients)
    primary = np.asarray([0.80, 0.90, 0.70, 1.00, 0.85, 0.60, 0.95, 0.75])
    comparator = np.asarray([1.10, 1.00, 1.05, 1.15, 0.95, 0.90, 1.10, 0.85])
    losses["primary"]["by_patient"] = dict(zip(patients, primary.tolist()))
    losses["primary"]["mean"] = float(primary.mean())
    losses["classical_time_conditioned_ridge_poisson"]["by_patient"] = dict(
        zip(patients, comparator.tolist())
    )
    losses["classical_time_conditioned_ridge_poisson"]["mean"] = float(
        comparator.mean()
    )

    gate = core.held_gate(losses, patients, completeness)
    assert gate == core.held_gate(losses, patients, completeness)
    assert gate["exact_stratified_resample_count"] == 84_375

    complete = tuple(range(5))
    partial = tuple(range(5, 8))
    resamples = np.asarray(
        [
            (*complete_draw, *partial_draw)
            for complete_draw in product(complete, repeat=5)
            for partial_draw in product(partial, repeat=3)
        ],
        dtype=int,
    )
    sampled_primary = primary[resamples].mean(axis=1)
    sampled_comparator = comparator[resamples].mean(axis=1)
    expected_difference = np.quantile(
        sampled_primary - sampled_comparator,
        (0.025, 0.975),
        method="linear",
    )
    expected_reduction = np.quantile(
        1.0 - sampled_primary / sampled_comparator,
        (0.025, 0.975),
        method="linear",
    )
    comparison = gate["comparisons"]["classical_time_conditioned_ridge_poisson"]
    np.testing.assert_allclose(
        comparison["additive_difference_95_interval"], expected_difference
    )
    np.testing.assert_allclose(
        comparison["relative_reduction_95_interval"], expected_reduction
    )


def test_all_three_frozen_decisions_replay_and_tampering_changes_outcome() -> None:
    calibration_losses = _losses(runner.EXPECTED_STAGE_PATIENTS["calibration"])
    calibration = core.calibration_gate(
        calibration_losses, runner.EXPECTED_STAGE_PATIENTS["calibration"]
    )
    assert calibration["passes"]
    calibration_value = {
        "status": "CALIBRATION_PASS",
        "selection": {"losses": calibration_losses, "gate": calibration},
    }
    assert runner._replay_decision("calibration", calibration_value) == calibration

    pilot_losses = _losses(runner.EXPECTED_STAGE_PATIENTS["pilot_adt"])
    pilot = core.pilot_gate(pilot_losses, runner.EXPECTED_STAGE_PATIENTS["pilot_adt"])
    assert pilot["passes"]
    assert (
        runner._replay_decision(
            "pilot_adt", {"status": "PILOT_PASS", "losses": pilot_losses}
        )
        == pilot
    )

    held_losses = _losses(runner.EXPECTED_STAGE_PATIENTS["held_adt"])
    completeness = {
        patient: (
            "complete" if patient in {"24", "27", "22", "25", "15"} else "partial"
        )
        for patient in runner.EXPECTED_STAGE_PATIENTS["held_adt"]
    }
    held = core.held_gate(
        held_losses, runner.EXPECTED_STAGE_PATIENTS["held_adt"], completeness
    )
    assert held["passes"]
    replayed = runner._replay_decision(
        "held_adt", {"status": "HELD_CONFIRMATION_PASS", "losses": held_losses}
    )
    assert replayed == held

    tampered = copy.deepcopy(held_losses)
    first_patient = runner.EXPECTED_STAGE_PATIENTS["held_adt"][0]
    tampered["primary"]["by_patient"][first_patient] = 2.0
    tampered_gate = core.held_gate(
        tampered, runner.EXPECTED_STAGE_PATIENTS["held_adt"], completeness
    )
    assert not tampered_gate["passes"]
