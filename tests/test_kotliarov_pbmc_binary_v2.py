import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments/develop_kotliarov_pbmc_binary_v2.py"
)
SPEC = importlib.util.spec_from_file_location("kotliarov_pbmc_binary_v2_tested", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
candidate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = candidate
SPEC.loader.exec_module(candidate)


def test_frozen_split_and_panel_match_public_designation() -> None:
    assert len(candidate.DEVELOPMENT) == 10
    assert len(candidate.HELD) == 9
    assert set(candidate.DEVELOPMENT).isdisjoint(candidate.HELD)
    assert set(candidate.EXCLUDED).isdisjoint(candidate.DEVELOPMENT + candidate.HELD)
    assert candidate.PANEL == (
        ("CD3D", "CD3"),
        ("CD4", "CD4"),
        ("CD8A", "CD8"),
        ("MS4A1", "CD20"),
        ("CD14", "CD14"),
        ("FCGR3A", "CD16"),
        ("NCAM1", "CD56"),
        ("HLA-DRA", "HLA-DR"),
        ("IL7R", "CD127"),
    )
    assert len(candidate.PANEL) ** 2 == 81


def test_import_does_not_mutate_shared_engine_contract() -> None:
    assert candidate.engine.MARKER_COUNT == candidate.ENGINE_DEFAULTS["MARKER_COUNT"]
    assert candidate.engine.CELL_BUDGET == candidate.ENGINE_DEFAULTS["CELL_BUDGET"]
    assert (
        candidate.engine.MINIMUM_INFORMATIVE_ENTITIES
        == candidate.ENGINE_DEFAULTS["MINIMUM_INFORMATIVE_ENTITIES"]
    )
    try:
        with candidate._engine_contract():
            assert candidate.engine.MARKER_COUNT == len(candidate.PANEL)
            assert candidate.engine.CELL_BUDGET == candidate.CELL_BUDGET
            raise RuntimeError("restore contract")
    except RuntimeError as error:
        assert str(error) == "restore contract"
    assert candidate.engine.MARKER_COUNT == candidate.ENGINE_DEFAULTS["MARKER_COUNT"]
    assert candidate.engine.CELL_BUDGET == candidate.ENGINE_DEFAULTS["CELL_BUDGET"]


def test_preflight_opens_no_matrix_file(monkeypatch) -> None:
    opened = []

    def forbidden(*args, **kwargs):
        opened.append((args, kwargs))
        raise AssertionError("preflight opened an HDF5 file")

    monkeypatch.setattr(candidate.reader.h5py, "File", forbidden)
    monkeypatch.setattr(
        candidate,
        "_immutable_public_bytes",
        lambda _commit, relative: (candidate.ROOT / relative).read_bytes(),
    )
    result = candidate.preflight()
    assert result["status"] == "SOURCE_IMPLEMENTATION_PREFLIGHT_PASS_NO_MATRIX_OPEN"
    assert result["matrix_files_opened"] == 0
    assert result["held_adt_values_authorized"] is False
    assert set(result["bindings"]) == set(candidate.LOCAL_BINDING_PATHS)
    assert result["runtime_contract"]["required_thread_environment"] == {
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }
    assert opened == []


def test_authorization_refuses_noncanonical_path(tmp_path: Path) -> None:
    try:
        candidate._validate_authorization(tmp_path / "authorization.json", "a" * 40)
    except PermissionError as error:
        assert "canonical repository path" in str(error)
    else:
        raise AssertionError("noncanonical source authorization path was accepted")


def test_authorization_refuses_dependency_binding_mutation(
    tmp_path: Path, monkeypatch
) -> None:
    authorization = tmp_path / "source_authorization_v2.json"
    monkeypatch.setattr(candidate, "AUTHORIZATION", authorization)
    binding = {
        "runner": {
            "path": candidate.LOCAL_BINDING_PATHS["runner"],
            "sha256": candidate._sha256(candidate.MODULE_PATH)
            if hasattr(candidate, "MODULE_PATH")
            else candidate._sha256(Path(candidate.__file__)),
        }
    }
    runtime = candidate._runtime_contract()
    payload = {
        "schema": "kotliarov-pbmc-binary-v2-source-authorization/2.0",
        "status": "SOURCE_PAIRED_DEVELOPMENT_ACCESS_AUTHORIZED",
        "public_source_freeze_commit": candidate.FROZEN_SOURCE_COMMIT,
        "public_implementation_commit": "b" * 40,
        "bindings": {"runner": {**binding["runner"], "sha256": "0" * 64}},
        "source_input_contract": candidate.EXPECTED_INPUTS,
        "runtime_contract": runtime,
        "held_adt_values_authorized": False,
        "source_attempt_path": candidate._relative(candidate.ATTEMPT),
        "source_output_path": candidate._relative(candidate.OUTPUT),
    }
    authorization.write_text(candidate.json.dumps(payload))
    monkeypatch.setattr(
        candidate,
        "preflight",
        lambda: {"bindings": binding},
    )
    monkeypatch.setattr(candidate, "_require_runtime_contract", lambda: runtime)
    try:
        candidate._validate_authorization(authorization, "c" * 40)
    except PermissionError as error:
        assert "not authorized" in str(error)
    else:
        raise AssertionError("mutated dependency binding was accepted")


def test_authorization_refuses_unordered_public_history(
    tmp_path: Path, monkeypatch
) -> None:
    authorization = tmp_path / "source_authorization_v2.json"
    monkeypatch.setattr(candidate, "AUTHORIZATION", authorization)
    runner_relative = candidate.LOCAL_BINDING_PATHS["runner"]
    binding = {
        "runner": {
            "path": runner_relative,
            "sha256": candidate._sha256(candidate.ROOT / runner_relative),
        }
    }
    runtime = candidate._runtime_contract()
    original_relative = candidate._relative
    auth_relative = "data/confirmation/kotliarov_pbmc_binary_v2/source_authorization_v2.json"
    monkeypatch.setattr(
        candidate,
        "_relative",
        lambda path: auth_relative
        if Path(path).resolve() == authorization.resolve()
        else original_relative(Path(path)),
    )
    payload = {
        "schema": "kotliarov-pbmc-binary-v2-source-authorization/2.0",
        "status": "SOURCE_PAIRED_DEVELOPMENT_ACCESS_AUTHORIZED",
        "public_source_freeze_commit": candidate.FROZEN_SOURCE_COMMIT,
        "public_implementation_commit": "b" * 40,
        "bindings": binding,
        "source_input_contract": candidate.EXPECTED_INPUTS,
        "runtime_contract": runtime,
        "held_adt_values_authorized": False,
        "source_attempt_path": original_relative(candidate.ATTEMPT),
        "source_output_path": original_relative(candidate.OUTPUT),
    }
    authorization.write_text(candidate.json.dumps(payload))
    monkeypatch.setattr(candidate, "preflight", lambda: {"bindings": binding})
    monkeypatch.setattr(candidate, "_require_runtime_contract", lambda: runtime)
    monkeypatch.setattr(
        candidate,
        "_immutable_public_bytes",
        lambda _commit, relative: authorization.read_bytes()
        if relative == auth_relative
        else (candidate.ROOT / relative).read_bytes(),
    )

    def refuse_history(_ancestor, _descendant):
        raise PermissionError("synthetic unordered history")

    monkeypatch.setattr(candidate, "_require_ancestor", refuse_history)
    try:
        candidate._validate_authorization(authorization, "c" * 40)
    except PermissionError as error:
        assert "unordered history" in str(error)
    else:
        raise AssertionError("unordered public history was accepted")


def test_source_cell_selection_is_count_blind_and_excludes_held(monkeypatch) -> None:
    donors = []
    batches = []
    identifiers = []
    for donor in candidate.DEVELOPMENT:
        donors.extend([donor] * 600)
        batches.extend(["1"] * 600)
        identifiers.extend([f"{donor}-cell-{index:04d}" for index in range(600)])
    for donor in candidate.HELD:
        donors.extend([donor] * 20)
        batches.extend(["2"] * 20)
        identifiers.extend([f"{donor}-held-{index:04d}" for index in range(20)])
    columns = {
        "batch": np.asarray(batches),
        "sampleid": np.asarray(donors),
        "joint_classification_global": np.asarray(["SNG_Singlet"] * len(donors)),
        "dmx_hto_match": np.asarray(["1"] * len(donors)),
        "timepoint": np.asarray(["d0"] * len(donors)),
    }
    monkeypatch.setattr(
        candidate.reader,
        "_read_dataframe_columns",
        lambda path, required: (np.asarray(identifiers), columns),
    )
    first = candidate._selected_source_cells(Path("unused.h5"))
    second = candidate._selected_source_cells(Path("unused.h5"))
    assert len(first) == 512 * len(candidate.DEVELOPMENT)
    assert first.groupby("donor").size().to_dict() == {
        donor: 512 for donor in candidate.DEVELOPMENT
    }
    assert set(first["donor"]) == set(candidate.DEVELOPMENT)
    assert not first["donor"].isin(candidate.HELD).any()
    assert first["cell_id"].tolist() == second["cell_id"].tolist()


def test_midrank_and_destroyed_controls_preserve_every_margin() -> None:
    cell_ids = [f"cell-{index:04d}" for index in range(candidate.CELL_BUDGET)]
    counts = np.column_stack(
        [
            np.arange(candidate.CELL_BUDGET) % (marker + 3)
            for marker in range(len(candidate.PANEL))
        ]
    )
    states = candidate._midrank_states(counts, cell_ids, "200")
    destroyed = candidate._destroyed_states(states, cell_ids, "200")
    np.testing.assert_array_equal(states.sum(axis=0), candidate.CELL_BUDGET // 2)
    np.testing.assert_array_equal(destroyed.sum(axis=0), states.sum(axis=0))
    assert not np.array_equal(destroyed, states)


def test_source_promotion_comparison_uses_donors_not_coordinates() -> None:
    primary = np.full(10, 0.80)
    comparator = np.full(10, 1.00)
    result = candidate._comparison(primary, comparator)
    assert result["donor_count"] == 10
    assert result["favorable_donors"] == 10
    assert result["relative_loss_reduction"] >= 0.05
    assert result["paired_donor_difference_95_ci"][1] < 0.0
    assert result["passes"] is True


def test_incomplete_configuration_cannot_be_selected() -> None:
    losses = {
        ("incomplete",): np.asarray([0.1, np.nan]),
        ("complete",): np.asarray([0.2, 0.2]),
    }
    selected, values = candidate._select_complete(losses)
    assert selected == ("complete",)
    np.testing.assert_array_equal(values, [0.2, 0.2])


def test_centered_residual_inversion_runs_for_the_full_panel() -> None:
    truth = np.broadcast_to(
        np.asarray([[160, 96], [96, 160]], dtype=np.int64),
        (len(candidate.PANEL), len(candidate.PANEL), 2, 2),
    ).copy()
    with candidate._engine_contract():
        for family in ("pearson", "deviance"):
            loss = candidate._classical_loss(
                truth,
                {
                    "family": family,
                    "centered": True,
                    "source_coordinate": [0.0] * (len(candidate.PANEL) ** 2),
                },
                1.0,
                np.ones((len(candidate.PANEL), len(candidate.PANEL)), dtype=bool),
            )
            assert np.isfinite(loss)


def test_classical_residual_alpha_one_reconstructs_repeated_source_panel() -> None:
    truth = np.broadcast_to(
        np.asarray([[180, 76], [76, 180]], dtype=np.int64),
        (len(candidate.PANEL), len(candidate.PANEL), 2, 2),
    ).copy()
    tables = np.stack([truth.copy() for _ in range(3)])
    support = np.ones((len(candidate.PANEL), len(candidate.PANEL)), dtype=bool)
    with candidate._engine_contract():
        for family in ("pearson", "deviance"):
            for centered in (False, True):
                model = candidate.classical._classical_model(
                    tables, family, centered
                )
                loss = candidate._classical_loss(truth, model, 1.0, support)
                assert abs(loss) < 1e-12


def test_develop_executes_every_frozen_candidate_and_serializes(
    monkeypatch,
) -> None:
    truth = np.broadcast_to(
        np.asarray([[180, 76], [76, 180]], dtype=np.int64),
        (len(candidate.PANEL), len(candidate.PANEL), 2, 2),
    ).copy()
    destroyed = np.broadcast_to(
        np.asarray([[128, 128], [128, 128]], dtype=np.int64),
        truth.shape,
    ).copy()
    support = np.ones((len(candidate.PANEL), len(candidate.PANEL)), dtype=bool)
    records = {
        donor: {
            "tables": truth.copy(),
            "destroyed_tables": destroyed.copy(),
            "support": support.copy(),
            "rna_profile": np.linspace(0.1, 0.9, len(candidate.PANEL)),
            "adt_profile": np.linspace(0.2, 1.0, len(candidate.PANEL)),
        }
        for donor in candidate.DEVELOPMENT
    }

    with candidate._engine_contract():
        exact_fit = candidate.engine._fit_common_effect(
            np.stack([truth, truth, truth]),
            support,
            np.stack([support, support, support]),
        )
    exact_conditional = np.asarray(exact_fit["population_log_odds"])
    exact_poisson = np.full(
        support.shape,
        2.0 * np.log(180.0 / 76.0),
        dtype=float,
    )
    residual_coordinates = {}
    for family in ("pearson", "deviance"):
        raw_function = (
            candidate.classical.signed_pearson_coordinate
            if family == "pearson"
            else candidate.classical.signed_deviance_coordinate
        )
        raw = raw_function(truth[0, 0])
        centered = candidate.classical.centered_classical_coordinate(
            truth[0, 0], statistic=family
        ).centered_coordinate
        residual_coordinates[(family, False)] = raw / np.sqrt(candidate.CELL_BUDGET)
        residual_coordinates[(family, True)] = centered / np.sqrt(
            candidate.CELL_BUDGET
        )

    def fake_primary(tables, *_args, **_kwargs):
        associated = float(np.asarray(tables)[..., 0, 0].mean()) > 128.0
        scale = 0.75 if associated else 0.0
        return {
            "population_log_odds": scale * exact_conditional,
            "fit_certificate": {"synthetic_deterministic_fit": True},
        }

    def fake_common(*_args, **_kwargs):
        return {
            "population_log_odds": 0.4 * exact_conditional,
            "fit_certificate": {"synthetic_deterministic_fit": True},
        }

    def fake_poisson(_tables):
        return {
            "population_log_odds": 0.4 * exact_poisson,
            "fit_certificate": {"synthetic_deterministic_fit": True},
        }

    def fake_classical(_tables, family, centered):
        coordinate = 0.4 * residual_coordinates[(family, centered)]
        return {
            "kind": "classical_residual",
            "family": family,
            "centered": centered,
            "source_coordinate": np.full(support.shape, coordinate).ravel().tolist(),
            "certificate": {"synthetic_deterministic_fit": True},
        }

    def fake_classical_loss(_truth, model, alpha, selected_support):
        assert selected_support.shape == support.shape
        family_offset = 0.01 if model["family"] == "pearson" else 0.02
        centering_offset = 0.01 if model["centered"] else 0.02
        return 0.2 + family_offset + centering_offset + (1.25 - alpha) ** 2

    monkeypatch.setattr(candidate.engine, "_fit_primary", fake_primary)
    monkeypatch.setattr(candidate.engine, "_fit_common_effect", fake_common)
    monkeypatch.setattr(candidate, "_fit_pooled_poisson", fake_poisson)
    monkeypatch.setattr(candidate.classical, "_classical_model", fake_classical)
    monkeypatch.setattr(candidate, "_classical_loss", fake_classical_loss)

    with candidate._engine_contract():
        result = candidate._develop(records)

    evaluations = result["candidate_evaluations"]
    assert len(evaluations["hierarchical"]) == 24
    assert len(evaluations["classical_residual"]) == 16
    assert len(evaluations["common_effect_cmle"]) == 4
    assert len(evaluations["pooled_saturated_poisson"]) == 4
    for family in evaluations.values():
        assert all(row["completed_folds"] == 10 for row in family)
        assert all(row["status"] == "COMPLETE" for row in family)
    assert {
        row["configuration"]["exact_null_centered"]
        for row in evaluations["classical_residual"]
    } == {False, True}
    assert len(result["support_diagnostics"]) == 10
    assert len(result["fold_losses"]["destroyed_link"]) == 10
    assert np.isfinite(result["fold_losses"]["destroyed_link"]).all()
    candidate.json.dumps(result, allow_nan=False)


def test_source_attempt_precedes_any_matrix_byte_access(
    tmp_path: Path, monkeypatch
) -> None:
    attempt = tmp_path / "attempt.json"
    output = tmp_path / "result.json"
    monkeypatch.setattr(candidate, "ATTEMPT", attempt)
    monkeypatch.setattr(candidate, "OUTPUT", output)
    monkeypatch.setattr(candidate, "_validate_authorization", lambda *_: {})

    def refuse_after_claim(_path: Path):
        assert attempt.is_file()
        raise RuntimeError("synthetic byte-access stop")

    monkeypatch.setattr(candidate, "_file_identity", refuse_after_claim)
    candidate.run_source(
        SimpleNamespace(
            authorization=str(tmp_path / "authorization.json"),
            authorization_commit="a" * 40,
            rna_matrix=str(tmp_path / "matrix.h5"),
            adt_matrix=str(tmp_path / "array.h5"),
            metadata_root=str(tmp_path / "metadata"),
        )
    )

    attempt_payload = candidate._read_json(attempt)
    result = candidate._read_json(output)
    assert attempt_payload["matrix_byte_access_begins_after_this_record"] is True
    assert result["status"] == "TERMINAL_SOURCE_EXECUTION_REFUSAL"
    assert result["reason"] == "synthetic byte-access stop"


def test_source_records_read_only_development_adt_columns_after_attempt(
    tmp_path: Path, monkeypatch
) -> None:
    attempt = tmp_path / "attempt.json"
    attempt.write_text("{}\n")
    monkeypatch.setattr(candidate, "ATTEMPT", attempt)
    donors = np.repeat(np.asarray(candidate.DEVELOPMENT), candidate.CELL_BUDGET)
    cells = np.asarray(
        [
            f"{donor}-cell-{index:04d}"
            for donor in candidate.DEVELOPMENT
            for index in range(candidate.CELL_BUDGET)
        ]
    )
    source_columns = np.arange(len(cells), dtype=np.int64)
    selected = candidate.pd.DataFrame(
        {"cell_id": cells, "source_column": source_columns, "donor": donors}
    )
    monkeypatch.setattr(candidate, "_selected_source_cells", lambda _path: selected)
    monkeypatch.setattr(
        candidate,
        "_sha256",
        lambda path: candidate.EXPECTED_METADATA[path.relative_to(tmp_path).as_posix()],
    )
    monkeypatch.setattr(
        candidate.reader,
        "_read_row_names",
        lambda _path: np.asarray([gene for gene, _ in candidate.PANEL]),
    )
    monkeypatch.setattr(
        candidate.reader,
        "_read_dataframe_columns",
        lambda _path, _required: (
            np.asarray([f"{protein}_PROT" for _, protein in candidate.PANEL]),
            {
                "target": np.asarray([protein for _, protein in candidate.PANEL]),
                "isotype": np.zeros(len(candidate.PANEL), dtype=np.int8),
            },
        ),
    )
    monkeypatch.setattr(
        candidate.reader,
        "_unique_indices",
        lambda _names, requested, _label: np.arange(len(requested), dtype=np.int64),
    )
    rna = np.tile(
        (np.arange(len(cells)) % 7 == 0).astype(np.int64),
        (len(candidate.PANEL), 1),
    )
    adt = np.vstack(
        [
            np.arange(len(cells), dtype=np.int64) % (marker + 3)
            for marker in range(len(candidate.PANEL))
        ]
    )
    monkeypatch.setattr(
        candidate,
        "_read_csc_exact_subset",
        lambda _path, _rows, columns, _shape: (
            rna[:, columns],
            {
                "selected_column_count": len(columns),
                "unselected_column_numeric_slices_read": 0,
            },
        ),
    )
    observed_columns = []

    def read_adt(_path, _rows, columns, _shape):
        assert attempt.is_file()
        observed_columns.extend(np.asarray(columns).tolist())
        return adt[:, columns]

    monkeypatch.setattr(candidate.reader, "_read_dense_subset", read_adt)
    with candidate._engine_contract():
        records, audit = candidate._source_records(
            tmp_path / "matrix.h5", tmp_path / "array.h5", tmp_path
        )
    assert set(records) == set(candidate.DEVELOPMENT)
    assert observed_columns == source_columns.tolist()
    assert audit["held_adt_columns_read"] == 0
    assert audit["held_adt_dataset_values_read"] == 0


def test_exact_csc_reader_touches_only_selected_column_slices(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "matrix.h5"
    with candidate.reader.h5py.File(path, "w") as handle:
        group = handle.create_group("compressed_sparse_matrix")
        group.attrs["layout"] = "CSC"
        group.create_dataset("shape", data=np.asarray([3, 5], dtype=np.uint32))
        group.create_dataset(
            "indptr", data=np.asarray([0, 1, 2, 3, 5, 6], dtype=np.uint64)
        )
        group.create_dataset(
            "indices", data=np.asarray([0, 1, 2, 0, 2, 1], dtype=np.uint16)
        )
        group.create_dataset(
            "data", data=np.asarray([10, 11, 12, 13, 14, 15], dtype=np.uint16)
        )
    calls = []
    original = candidate.reader.h5py.Dataset.__getitem__

    def tracked(dataset, key):
        calls.append((dataset.name, key))
        return original(dataset, key)

    monkeypatch.setattr(candidate.reader.h5py.Dataset, "__getitem__", tracked)
    observed, audit = candidate._read_csc_exact_subset(
        path,
        np.asarray([0, 2], dtype=np.int64),
        np.asarray([1, 3], dtype=np.int64),
        (3, 5),
    )
    np.testing.assert_array_equal(observed, np.asarray([[0, 13], [0, 14]]))
    indptr_keys = [key for name, key in calls if name.endswith("/indptr")]
    assert [int(key) for key in indptr_keys] == [1, 2, 3, 4]
    for dataset in ("indices", "data"):
        keys = [key for name, key in calls if name.endswith(f"/{dataset}")]
        assert [(key.start, key.stop, key.step) for key in keys] == [
            (1, 2, None),
            (3, 5, None),
        ]
    assert audit["indptr_full_vector_materialized"] is False
    assert audit["unselected_column_numeric_slices_read"] == 0
    assert [row["source_column"] for row in audit["column_reads"]] == [1, 3]


def test_integer_count_contract_refuses_fractional_and_nonfinite_values() -> None:
    for values in (
        np.asarray([[0.5, 1.0]]),
        np.asarray([[np.nan, 1.0]]),
        np.asarray([[-1.0, 1.0]]),
    ):
        try:
            candidate._integer_counts(values, "synthetic")
        except ValueError as error:
            assert "nonnegative integer counts" in str(error)
        else:
            raise AssertionError("invalid raw counts were accepted")
    constant = np.zeros((candidate.CELL_BUDGET, len(candidate.PANEL)), dtype=int)
    states = candidate._midrank_states(
        constant,
        [f"cell-{index}" for index in range(candidate.CELL_BUDGET)],
        "200",
    )
    np.testing.assert_array_equal(states.sum(axis=0), candidate.CELL_BUDGET // 2)


def test_poisson_reconstruction_and_positive_orientation() -> None:
    first = np.broadcast_to(
        np.asarray([[20, 10], [10, 20]], dtype=np.int64),
        (len(candidate.PANEL), len(candidate.PANEL), 2, 2),
    ).copy()
    second = np.broadcast_to(
        np.asarray([[30, 10], [10, 30]], dtype=np.int64),
        first.shape,
    ).copy()
    tables = np.stack([first, second])
    support = np.ones(tables.shape[:-2], dtype=bool)
    fit = candidate._fit_pooled_poisson(tables)
    pooled = tables.sum(axis=0)
    with candidate._engine_contract():
        rows, columns = candidate.engine._margins(pooled)
        reconstructed = candidate._poisson_prediction(
            fit["population_log_odds"], rows, columns, 1.0
        )
        common = candidate.engine._fit_common_effect(
            tables,
            np.ones((len(candidate.PANEL), len(candidate.PANEL)), dtype=bool),
            support,
        )
    np.testing.assert_allclose(reconstructed, pooled, rtol=0, atol=1e-8)
    assert np.all(fit["population_log_odds"] > 0)
    assert np.all(common["population_log_odds"] > 0)
    assert fit["fit_certificate"]["passes"] is True


def test_pooled_poisson_includes_degenerate_margin_donor_tables() -> None:
    informative = np.broadcast_to(
        np.asarray([[20, 10], [10, 20]], dtype=np.int64),
        (len(candidate.PANEL), len(candidate.PANEL), 2, 2),
    ).copy()
    degenerate = np.broadcast_to(
        np.asarray([[0, 0], [30, 30]], dtype=np.int64),
        informative.shape,
    ).copy()
    with_degenerate = candidate._fit_pooled_poisson(
        np.stack([informative, degenerate])
    )
    without_degenerate = candidate._fit_pooled_poisson(informative[None, ...])
    assert not np.allclose(
        with_degenerate["population_log_odds"],
        without_degenerate["population_log_odds"],
    )
    assert (
        with_degenerate["fit_certificate"][
            "includes_all_source_donor_tables_including_degenerate_margins"
        ]
        is True
    )


def test_loss_refuses_small_margin_drift() -> None:
    truth = np.broadcast_to(
        np.asarray([[20, 10], [10, 20]], dtype=np.int64),
        (len(candidate.PANEL), len(candidate.PANEL), 2, 2),
    ).copy()
    prediction = truth.astype(float)
    prediction[0, 0, 0, 0] += 1e-4
    try:
        candidate._loss(
            truth,
            prediction,
            np.ones((len(candidate.PANEL), len(candidate.PANEL)), dtype=bool),
        )
    except FloatingPointError as error:
        assert "target margin" in str(error)
    else:
        raise AssertionError("margin drift passed the loss certificate")


def test_loss_clips_negative_roundoff_to_zero() -> None:
    truth = np.broadcast_to(
        np.asarray([[20, 10], [10, 20]], dtype=np.int64),
        (len(candidate.PANEL), len(candidate.PANEL), 2, 2),
    ).copy()
    perturbation = np.asarray([[1e-8, -1e-8], [-1e-8, 1e-8]])
    prediction = truth.astype(float) + perturbation
    loss = candidate._loss(
        truth,
        prediction,
        np.ones((len(candidate.PANEL), len(candidate.PANEL)), dtype=bool),
    )
    assert loss == 0.0
