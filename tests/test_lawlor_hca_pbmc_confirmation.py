import json
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import experiments.confirm_lawlor_hca_pbmc as lawlor
from experiments.confirm_lawlor_hca_pbmc import (
    _field_table,
    _margin_stats,
    _prepare_locked_tables,
    _reducer_artifact_bundle,
    _require_locked_prediction_consistency,
    _require_locked_table_consistency,
    _require_reducer_artifact_bundle,
    _require_score_authorization,
    _require_score_release,
    _score_tables,
    _sha256,
    _residual_table,
    _state_thresholds,
    _table_stats,
    _write_refusal,
    summarize,
)


def _states(table):
    first, second = [], []
    for row, column in np.ndindex(table.shape):
        first.extend([row] * int(table[row, column]))
        second.extend([column] * int(table[row, column]))
    return np.asarray(first), np.asarray(second)


def test_version_two_lock_is_phase_consistent_and_has_no_reducer_bypass():
    designation = json.loads(lawlor.DESIGNATION.read_text())
    assert designation["schema"].endswith("/2.0")
    if designation["status"] == "OUTCOME_ACCESS_DISABLED":
        assert designation["outcome_access_authorized"] is False
        assert designation["public_freeze_commit"] is None
        assert designation["public_freeze_url"] is None
        with pytest.raises(PermissionError, match="not SEALED"):
            lawlor.preflight(require_sealed=True)
    else:
        assert designation["status"] == "SEALED"
        assert designation["outcome_access_authorized"] is True
        assert designation["public_freeze_commit"] is not None
        assert designation["public_freeze_url"] is not None
        lawlor.preflight(require_sealed=True)
    assert designation["score_authorization"] == (
        "data/confirmation/pbmc_citeseq_hca/score_authorization_v2.json"
    )
    assert designation["score_release"] == (
        "data/confirmation/pbmc_citeseq_hca/score_release_v2.json"
    )
    assert lawlor.AUTHORIZATION_TEMPLATE.exists()
    assert "skip_reducer" not in Path(lawlor.__file__).read_text()


def test_two_candidate_confirmation_family_is_fixed_before_outcomes():
    designation = json.loads(lawlor.DESIGNATION.read_text())
    family = designation["confirmation_family"]
    assert family["preflight_refusal_not_scoreable"] == "GSE143417 POKI-seq"
    assert family["untouched_scoreable_candidates"] == [
        "HCA:efea6426-510a-4b60-9a19-277e52bfa815 (Lawlor)",
        "GSE164378 (Hao)",
    ]
    assert "without stopping" in family["execution_rule"]
    assert "Bonferroni" in family["familywise_rule"]


def test_margin_null_reference_is_independent_of_pairing_order():
    first = np.repeat(np.arange(3), [20, 30, 40])
    second = np.repeat(np.arange(3), [25, 35, 30])
    generator = np.random.default_rng(9)
    original = _margin_stats(first, second, seed=44)
    permuted = _margin_stats(
        first[generator.permutation(len(first))],
        second[generator.permutation(len(second))],
        seed=44,
    )
    for family in ("field", "pearson", "deviance"):
        np.testing.assert_allclose(
            original[f"{family}_null"], permuted[f"{family}_null"]
        )


def test_field_reconstruction_includes_both_finite_table_null_means():
    baseline_table = np.array([[18, 3, 4], [5, 20, 5], [7, 6, 22]])
    stimulus_table = np.array([[15, 5, 5], [4, 22, 9], [6, 8, 26]])
    baseline = _table_stats(*_states(baseline_table), seed=3)
    stimulus = _table_stats(*_states(stimulus_table), seed=7)
    contrast = stimulus["field"] - baseline["field"]
    prediction = _field_table(
        baseline["field_raw"],
        baseline["field_null"],
        stimulus["field_null"],
        contrast,
        stimulus["rows"],
        stimulus["columns"],
    )
    np.testing.assert_allclose(prediction.sum(axis=1), stimulus_table.sum(axis=1))
    np.testing.assert_allclose(prediction.sum(axis=0), stimulus_table.sum(axis=0))
    raw = stimulus["field_raw"]
    reconstructed = _field_table(
        raw,
        np.zeros_like(raw),
        np.zeros_like(raw),
        np.zeros_like(raw),
        stimulus["rows"],
        stimulus["columns"],
    )
    np.testing.assert_allclose(prediction, reconstructed)


def test_full_residual_reconstruction_recovers_the_stimulus_table():
    baseline_table = np.array([[18, 3, 4], [5, 20, 5], [7, 6, 22]])
    stimulus_table = np.array([[15, 5, 5], [4, 22, 9], [6, 8, 26]])
    baseline = _table_stats(*_states(baseline_table), seed=11)
    stimulus = _table_stats(*_states(stimulus_table), seed=13)
    for family in ("pearson", "deviance"):
        contrast = stimulus[family] - baseline[family]
        prediction = _residual_table(
            baseline[f"{family}_raw"],
            baseline[f"{family}_null"],
            stimulus[f"{family}_null"],
            baseline["total"],
            stimulus["total"],
            contrast,
            stimulus["rows"],
            stimulus["columns"],
            family,
        )
        np.testing.assert_allclose(prediction, stimulus_table, atol=1e-7)


def test_destroyed_link_control_uses_the_primary_structured_fit(monkeypatch):
    donors = len(lawlor.DEVELOPMENT) + len(lawlor.HELD)
    entities = 12
    rng = np.random.default_rng(4)
    arrays = {
        "field": rng.normal(size=(donors, entities, 2, 2)),
        "field_variance": np.ones((donors, entities, 2, 2)),
        "field_destroyed": np.full((donors, entities, 2, 2), 7.0),
        "covariance": rng.normal(size=(donors, entities, 2, 2)),
        "pearson": rng.normal(size=(donors, entities, 3, 3)),
        "pearson_variance": np.ones((donors, entities, 3, 3)),
        "deviance": rng.normal(size=(donors, entities, 3, 3)),
        "deviance_variance": np.ones((donors, entities, 3, 3)),
        "endpoint": rng.normal(size=(donors, entities, 4)),
    }
    entity_frame = pd.DataFrame(
        {
            "marker_id": [f"marker-{index}" for index in range(entities)],
            "gene_symbol": [f"GENE{index}" for index in range(entities)],
            "contrast_id": ["CD3_CD28:B"] * entities,
            "lineage": ["B"] * entities,
        }
    )
    monkeypatch.setattr(
        lawlor,
        "_embedding_laplacian",
        lambda frame: (
            np.zeros((len(frame), len(frame))),
            np.zeros((len(frame), len(frame))),
            {},
        ),
    )
    monkeypatch.setattr(
        lawlor,
        "_structured",
        lambda values, variance, laplacian, **kwargs: values + 100.0,
    )
    predictions, _ = lawlor._fit_predictions(arrays, entity_frame)
    np.testing.assert_allclose(predictions["field_destroyed"], 107.0)


def test_locked_tables_must_be_reconstructible_from_locked_coordinates(monkeypatch):
    expected = {"field_primary": np.ones((4, 2, 3, 3))}
    monkeypatch.setattr(lawlor, "_predict_tables", lambda *args: expected)
    _require_locked_table_consistency({}, {}, expected, 2)
    tampered = {"field_primary": expected["field_primary"].copy()}
    tampered["field_primary"][0, 0, 0, 0] = 2.0
    with pytest.raises(PermissionError, match="inconsistent with coordinates"):
        _require_locked_table_consistency({}, {}, tampered, 2)


def test_locked_coordinates_must_exactly_match_the_sealed_development_refit(
    monkeypatch,
):
    expected = {
        "field_primary": np.arange(48, dtype=float).reshape(12, 4),
        "field_endpoint_ridge": np.arange(192, dtype=float).reshape(4, 12, 4),
    }
    monkeypatch.setattr(
        lawlor, "_fit_predictions", lambda arrays, entities: (expected, {})
    )
    _require_locked_prediction_consistency({}, pd.DataFrame(index=range(12)), expected)

    tampered = {name: value.copy() for name, value in expected.items()}
    tampered["field_primary"][0, 0] = np.nextafter(
        tampered["field_primary"][0, 0], np.inf
    )
    with pytest.raises(PermissionError, match="sealed development fit"):
        _require_locked_prediction_consistency(
            {}, pd.DataFrame(index=range(12)), tampered
        )


def test_coordinate_check_precedes_held_baseline_pairing(monkeypatch):
    calls = []
    held_rows = np.ones((len(lawlor.HELD), 2, 3))
    held_columns = np.ones((len(lawlor.HELD), 2, 3))

    def build(*args, open_held_baseline_pairing, open_held_stimulus_pairing):
        del args
        calls.append((open_held_baseline_pairing, open_held_stimulus_pairing))
        if open_held_baseline_pairing:
            raise AssertionError("held baseline pairing opened before coordinate check")
        prefix = np.zeros((len(lawlor.DEVELOPMENT), 2, 3))
        return {
            "stimulus_rows": np.concatenate((prefix, held_rows)),
            "stimulus_columns": np.concatenate((prefix, held_columns)),
        }

    monkeypatch.setattr(lawlor, "_build_fields", build)
    monkeypatch.setattr(
        lawlor,
        "_require_locked_prediction_consistency",
        lambda *args: (_ for _ in ()).throw(
            PermissionError("locked prediction coordinates differ")
        ),
    )
    record = {
        "held_stimulus_rows": held_rows.tolist(),
        "held_stimulus_columns": held_columns.tolist(),
    }
    with pytest.raises(PermissionError, match="coordinates differ"):
        _prepare_locked_tables(
            pd.DataFrame(),
            pd.DataFrame(index=range(2)),
            np.empty((2, 0)),
            np.empty((2, 0)),
            record,
            {},
        )
    assert calls == [(False, False)]


def test_refusal_scope_begins_after_procedural_gates_before_data_parsing():
    predict_source = inspect.getsource(lawlor.predict)
    assert predict_source.index("preflight(") < predict_source.index("try:")
    assert predict_source.index("try:") < predict_source.index("reduce_inputs(")
    source = inspect.getsource(lawlor.score)
    assert source.index("_require_score_release(") < source.index("try:")
    assert source.index("reduced = Path(args.reduced)") < source.index("try:")
    assert source.index("try:") < source.index("_locked_predictions(")
    assert source.index("try:") < source.index("_prepare_locked_tables(")


def test_public_gate_failures_do_not_create_terminal_refusals(monkeypatch, tmp_path):
    prediction = tmp_path / "prediction-refusal.json"
    score = tmp_path / "score-refusal.json"
    monkeypatch.setattr(lawlor, "PREDICTION_PATH", prediction)
    monkeypatch.setattr(lawlor, "OUTPUT", score)
    monkeypatch.setattr(
        lawlor,
        "preflight",
        lambda **kwargs: (_ for _ in ()).throw(PermissionError("not SEALED")),
    )

    with pytest.raises(PermissionError, match="not SEALED"):
        lawlor.predict(SimpleNamespace())
    with pytest.raises(PermissionError, match="not SEALED"):
        lawlor.score(SimpleNamespace())

    assert not prediction.exists()
    assert not score.exists()


def test_score_tables_returns_one_loss_per_held_donor_and_entity():
    entity_count = 3
    table = np.array([[12.0, 3.0, 2.0], [4.0, 15.0, 5.0], [3.0, 6.0, 18.0]])
    held = np.broadcast_to(
        table, (len(lawlor.HELD), entity_count, 3, 3)
    ).copy()
    arrays = {
        "stimulus_table": np.concatenate(
            (
                np.zeros((len(lawlor.DEVELOPMENT), entity_count, 3, 3)),
                held,
            )
        )
    }
    losses = _score_tables(arrays, {"field_primary": held.copy()})
    assert losses["field_primary"].shape == (len(lawlor.HELD), entity_count)
    np.testing.assert_allclose(losses["field_primary"], 0.0, atol=1e-14)

    with pytest.raises(ValueError, match="axes differ"):
        _score_tables(
            arrays,
            {"field_primary": held[:, : entity_count - 1].copy()},
        )


def test_execution_refusal_is_deterministic_and_sanitizes_home_path(tmp_path):
    output = tmp_path / "refusal.json"
    record = _write_refusal(
        output,
        stage="PREDICT",
        error=RuntimeError(f"failed under {Path.home()}/private-input"),
    )
    assert record["status"] == "REFUSE_EXECUTION"
    assert record["stage"] == "PREDICT"
    assert str(Path.home()) not in output.read_text()
    with pytest.raises(FileExistsError, match="already exists"):
        _write_refusal(output, stage="PREDICT", error=RuntimeError("again"))


def test_marker_aliases_are_explicit_and_one_to_one():
    aliases = pd.read_csv(lawlor.ALIASES, sep="\t")
    assert aliases["adt_token"].str.upper().is_unique
    assert aliases["gene_symbol"].str.upper().is_unique
    assert not {"CD45RA", "CD45RO"} & set(aliases["adt_token"])
    reducer = lawlor.REDUCER.read_text()
    assert "fallback <-" not in reducer
    assert "RNA gene symbols are ambiguous" in reducer


def test_marker_contrast_support_uses_only_separate_held_margins(monkeypatch):
    rows = []
    values = []
    for donor in lawlor.DEVELOPMENT + lawlor.HELD:
        for stimulus, lineage in lawlor.CONTRASTS:
            for condition in ("Baseline", stimulus):
                for index in range(30):
                    rows.append(
                        {
                            "donor": donor,
                            "condition": condition,
                            "cell_type": lineage,
                        }
                    )
                    values.append((index + 0.5) / 30.0)
    cells = pd.DataFrame(rows)
    markers = pd.DataFrame(
        {
            "marker_id": [f"ADT{index}::G{index}" for index in range(12)],
            "gene_symbol": [f"G{index}" for index in range(12)],
        }
    )
    rna = np.vstack([np.asarray(values) + index / 10 for index in range(12)])
    adt = rna.copy()
    bad = (
        (cells["donor"].astype(str).to_numpy() == lawlor.HELD[0])
        & (cells["condition"].astype(str).to_numpy() == lawlor.CONTRASTS[0][0])
        & (cells["cell_type"].astype(str).to_numpy() == lawlor.CONTRASTS[0][1])
    )
    adt[0, bad] = -10.0
    monkeypatch.setattr(
        lawlor,
        "_table",
        lambda *args: (_ for _ in ()).throw(AssertionError("joint table opened")),
    )

    entities, exclusions, rna_state, adt_state = _state_thresholds(
        cells, markers, rna, adt
    )

    assert len(entities) == 71
    assert entities["marker_id"].nunique() == 12
    assert rna_state.shape == adt_state.shape == (71, len(cells))
    assert exclusions[0]["entity_id"].endswith("CD3_CD28:B")
    assert exclusions[0]["reasons"] == [
        "HELD_PAIRING_INDEPENDENT_MARGINAL_SUPPORT"
    ]
    assert entities["development_minimum_rna_state_fraction"].min() >= 0.05
    assert entities["held_minimum_adt_state_fraction"].min() >= 0.05

    with pytest.raises(ValueError, match="12 unique marker clusters"):
        _state_thresholds(cells, markers.iloc[:3], rna[:3], adt[:3])


def test_entity_hypergraph_has_six_external_markers_and_typed_edges(
    tmp_path, monkeypatch
):
    marker_count = 8
    entities = pd.DataFrame(
        [
            {
                "marker_id": f"marker-{marker}",
                "gene_symbol": f"G{marker}",
                "contrast_id": contrast,
                "lineage": contrast.split(":", 1)[1],
            }
            for marker in range(marker_count)
            for contrast in ("CD3_CD28:B", "LPS:CD14_Mono")
        ]
    )
    embedding = tmp_path / "embedding.npz"
    np.savez(
        embedding,
        gene_names=np.asarray([f"G{index}" for index in range(marker_count)]),
        embedding=np.random.default_rng(7).normal(size=(marker_count, 5)),
    )
    monkeypatch.setattr(lawlor, "SCGPT", embedding)
    monkeypatch.setattr(lawlor, "ROOT", tmp_path)

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        laplacian, permuted_laplacian, metadata = lawlor._embedding_laplacian(
            entities
        )

    assert laplacian.shape == (len(entities), len(entities))
    assert np.isfinite(laplacian).all()
    assert permuted_laplacian.shape == laplacian.shape
    assert np.isfinite(permuted_laplacian).all()
    assert metadata["external_neighbors_per_covered_marker"] == 6
    assert all(
        len(neighbors) == 6 for neighbors in metadata["embedding_neighbors"].values()
    )
    assert metadata["contrast_hyperedges"] == ["CD3_CD28:B", "LPS:CD14_Mono"]
    assert metadata["lineage_hyperedges"] == ["B", "CD14_Mono"]
    assert metadata["membership_control_preserves_typed_incidence"] is True


def test_membership_control_permutes_marker_gene_rows_but_not_typed_edges():
    marker_order = [f"marker-{index}" for index in range(12)]
    marker_values = np.repeat(marker_order, 2)
    gene_incidence = np.repeat(np.eye(12), 2, axis=0)
    typed_incidence = np.tile(np.eye(2), (12, 1))

    permuted, permutation = lawlor._membership_permuted_incidence(
        gene_incidence, typed_incidence, marker_values, marker_order
    )

    np.testing.assert_array_equal(permuted[:, 12:], typed_incidence)
    assert permutation != list(range(12))
    assert not np.array_equal(permuted[:, :12], gene_incidence)
    for marker in marker_order:
        rows = permuted[marker_values == marker, :12]
        np.testing.assert_array_equal(rows[0], rows[1])


def test_held_stimulus_pairing_is_not_opened_before_prediction_lock(monkeypatch):
    rows = []
    for donor in lawlor.DEVELOPMENT + lawlor.HELD:
        for stimulus, lineage in lawlor.CONTRASTS:
            for condition in ("Baseline", stimulus):
                rows.extend(
                    {
                        "donor": donor,
                        "condition": condition,
                        "cell_type": lineage,
                    }
                    for _ in range(15)
                )
    cells = pd.DataFrame(rows)
    entities = pd.DataFrame(
        [
            {
                "marker_id": f"marker-{index}",
                "stimulus": stimulus,
                "lineage": lineage,
            }
            for index, (stimulus, lineage) in enumerate(lawlor.CONTRASTS)
        ]
    )
    state = np.tile(np.arange(3), len(cells) // 3)
    state = np.broadcast_to(state, (len(entities), len(cells)))
    calls = 0

    def margins(first, second, seed):
        del seed
        row = np.bincount(first, minlength=3).astype(float)
        column = np.bincount(second, minlength=3).astype(float)
        return {
            "rows": row,
            "columns": column,
            "total": float(len(first)),
            "endpoint": np.zeros(4),
            "field_null": np.zeros((2, 2)),
            "field_destroyed": np.zeros((2, 2)),
            "field_variance": np.ones((2, 2)),
            "pearson_null": np.zeros((3, 3)),
            "pearson_destroyed": np.zeros((3, 3)),
            "pearson_variance": np.ones((3, 3)),
            "deviance_null": np.zeros((3, 3)),
            "deviance_destroyed": np.zeros((3, 3)),
            "deviance_variance": np.ones((3, 3)),
        }

    def table_stats(first, second, seed):
        nonlocal calls
        calls += 1
        result = margins(first, second, seed)
        return {
            **result,
            "table": lawlor._table(first, second),
            "field": np.zeros((2, 2)),
            "field_raw": np.zeros((2, 2)),
            "covariance": np.zeros((2, 2)),
            "pearson": np.zeros((3, 3)),
            "pearson_raw": np.zeros((3, 3)),
            "deviance": np.zeros((3, 3)),
            "deviance_raw": np.zeros((3, 3)),
        }

    monkeypatch.setattr(lawlor, "_margin_stats", margins)
    monkeypatch.setattr(lawlor, "_table_stats", table_stats)
    result = lawlor._build_fields(
        cells,
        entities,
        state,
        state,
        open_held_baseline_pairing=False,
        open_held_stimulus_pairing=False,
    )
    assert calls == len(entities) * len(lawlor.DEVELOPMENT) * 2
    held = slice(len(lawlor.DEVELOPMENT), None)
    assert np.isnan(result["stimulus_table"][held]).all()
    assert np.isnan(result["field"][held]).all()
    assert np.isnan(result["baseline_field_raw"][held]).all()
    assert np.isfinite(result["stimulus_rows"][held]).all()


def test_prediction_is_invariant_to_within_block_held_adt_pairing(monkeypatch):
    rows = []
    values = []
    for donor in lawlor.DEVELOPMENT + lawlor.HELD:
        for stimulus, lineage in lawlor.CONTRASTS:
            for condition in ("Baseline", stimulus):
                for index in range(18):
                    rows.append(
                        {
                            "donor": donor,
                            "condition": condition,
                            "cell_type": lineage,
                        }
                    )
                    values.append((index + 0.5) / 18.0)
    cells = pd.DataFrame(rows)
    markers = pd.DataFrame(
        {
            "marker_id": [f"ADT{index}::G{index}" for index in range(12)],
            "gene_symbol": [f"G{index}" for index in range(12)],
        }
    )
    rna = np.vstack([np.asarray(values) + index / 10 for index in range(12)])
    adt = 1.1 * rna
    shuffled_adt = adt.copy()
    donor_values = cells["donor"].astype(str).to_numpy()
    conditions = cells["condition"].astype(str).to_numpy()
    lineages = cells["cell_type"].astype(str).to_numpy()
    generator = np.random.default_rng(19)
    for donor in lawlor.HELD:
        for stimulus, lineage in lawlor.CONTRASTS:
            for condition in ("Baseline", stimulus):
                index = np.flatnonzero(
                    (donor_values == donor)
                    & (conditions == condition)
                    & (lineages == lineage)
                )
                shuffled_adt[:, index] = adt[:, generator.permutation(index)]

    original = _state_thresholds(cells, markers, rna, adt)
    permuted = _state_thresholds(cells, markers, rna, shuffled_adt)
    original_entities, original_excluded, original_rna, original_adt = original
    permuted_entities, permuted_excluded, permuted_rna, permuted_adt = permuted
    pd.testing.assert_frame_equal(original_entities, permuted_entities)
    assert original_excluded == permuted_excluded == []
    np.testing.assert_array_equal(original_rna, permuted_rna)

    def margins(first, second, seed):
        del seed
        row = np.bincount(first, minlength=3).astype(float)
        column = np.bincount(second, minlength=3).astype(float)
        total = float(len(first))
        return {
            "rows": row,
            "columns": column,
            "total": total,
            "endpoint": np.r_[row[:2], column[:2]] / total,
            "field_null": np.zeros((2, 2)),
            "field_destroyed": np.zeros((2, 2)),
            "field_variance": np.ones((2, 2)),
            "pearson_null": np.zeros((3, 3)),
            "pearson_destroyed": np.zeros((3, 3)),
            "pearson_variance": np.ones((3, 3)),
            "deviance_null": np.zeros((3, 3)),
            "deviance_destroyed": np.zeros((3, 3)),
            "deviance_variance": np.ones((3, 3)),
        }

    def table_stats(first, second, seed):
        result = margins(first, second, seed)
        table = lawlor._table(first, second)
        residual = table / table.sum()
        field = residual[:2, :2]
        return {
            **result,
            "table": table,
            "field": field,
            "field_raw": field,
            "covariance": field,
            "pearson": residual,
            "pearson_raw": residual,
            "deviance": residual,
            "deviance_raw": residual,
        }

    monkeypatch.setattr(lawlor, "_margin_stats", margins)
    monkeypatch.setattr(lawlor, "_table_stats", table_stats)
    monkeypatch.setattr(
        lawlor,
        "_embedding_laplacian",
        lambda frame: (
            np.zeros((len(frame), len(frame))),
            np.zeros((len(frame), len(frame))),
            {},
        ),
    )
    monkeypatch.setattr(
        lawlor,
        "_structured",
        lambda values, variance, laplacian, **kwargs: values,
    )
    original_arrays = lawlor._build_fields(
        cells,
        original_entities,
        original_rna,
        original_adt,
        open_held_baseline_pairing=False,
        open_held_stimulus_pairing=False,
    )
    permuted_arrays = lawlor._build_fields(
        cells,
        permuted_entities,
        permuted_rna,
        permuted_adt,
        open_held_baseline_pairing=False,
        open_held_stimulus_pairing=False,
    )
    for name in original_arrays:
        np.testing.assert_allclose(
            original_arrays[name], permuted_arrays[name], equal_nan=True
        )
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        original_prediction, _ = lawlor._fit_predictions(
            original_arrays, original_entities
        )
        permuted_prediction, _ = lawlor._fit_predictions(
            permuted_arrays, permuted_entities
        )
    for name in original_prediction:
        np.testing.assert_allclose(
            original_prediction[name], permuted_prediction[name]
        )


def test_score_requires_the_exact_publicly_recorded_prediction_sha(tmp_path):
    prediction = tmp_path / "results" / "prediction.json"
    prediction.parent.mkdir()
    prediction.write_text('{"frozen": true}\n')
    authorization = tmp_path / "authorization.json"
    commit = "0123456789abcdef0123456789abcdef01234567"
    record = {
        "schema": "lawlor-hca-pbmc-score-authorization/2.0",
        "status": "SEALED",
        "outcome_access_authorized": True,
        "prediction_path": "results/prediction.json",
        "prediction_sha256": _sha256(prediction),
        "prediction_bytes": prediction.stat().st_size,
        "prediction_public_url": (
            "https://github.com/example/benchmark/blob/"
            f"{commit}/results/prediction.json"
        ),
        "prediction_public_commit": commit,
        "runner_sha256": _sha256(Path(lawlor.__file__)),
        "protocol_sha256": _sha256(lawlor.PROTOCOL),
    }
    authorization.write_text(json.dumps(record))
    original_root = lawlor.ROOT
    lawlor.ROOT = tmp_path
    try:
        assert _require_score_authorization(prediction, authorization) == record

        record["prediction_sha256"] = "0" * 64
        authorization.write_text(json.dumps(record))
        with pytest.raises(PermissionError, match="hash differs"):
            _require_score_authorization(prediction, authorization)
    finally:
        lawlor.ROOT = original_root


def test_score_release_binds_the_published_authorization(tmp_path):
    authorization = tmp_path / "data" / "authorization.json"
    authorization.parent.mkdir()
    authorization.write_text('{"sealed": true}\n')
    release = tmp_path / "release.json"
    commit = "89abcdef0123456789abcdef0123456789abcdef"
    record = {
        "schema": "lawlor-hca-pbmc-score-release/2.0",
        "status": "SEALED",
        "held_pairing_access_authorized": True,
        "authorization_path": "data/authorization.json",
        "authorization_sha256": _sha256(authorization),
        "authorization_bytes": authorization.stat().st_size,
        "authorization_public_url": (
            "https://github.com/example/benchmark/blob/"
            f"{commit}/data/authorization.json"
        ),
        "authorization_public_commit": commit,
        "runner_sha256": _sha256(Path(lawlor.__file__)),
        "protocol_sha256": _sha256(lawlor.PROTOCOL),
    }
    release.write_text(json.dumps(record))
    original_root = lawlor.ROOT
    lawlor.ROOT = tmp_path
    try:
        assert _require_score_release(authorization, release) == record
        authorization.write_text('{"sealed": false}\n')
        with pytest.raises(PermissionError, match="authorization hash differs"):
            _require_score_release(authorization, release)
    finally:
        lawlor.ROOT = original_root


@pytest.mark.parametrize(
    ("commit", "url", "message"),
    [
        (
            "0123456789abcdef",
            "https://github.com/example/benchmark/blob/0123456789abcdef/results/prediction.json",
            "40 lowercase",
        ),
        (
            "A" * 40,
            f"https://github.com/example/benchmark/blob/{'A' * 40}/results/prediction.json",
            "40 lowercase",
        ),
        (
            "0" * 40,
            f"https://github.com/example/benchmark/blob/{'1' * 40}/results/prediction.json",
            "authorized commit",
        ),
        (
            "0" * 40,
            f"https://github.com/example/benchmark/blob/{'0' * 40}/results/other.json",
            "exact prediction path",
        ),
        (
            "0" * 40,
            f"https://example.org/example/benchmark/blob/{'0' * 40}/results/prediction.json",
            "GitHub",
        ),
    ],
)
def test_score_rejects_nonexact_public_prediction_location(
    tmp_path, commit, url, message
):
    prediction = tmp_path / "results" / "prediction.json"
    prediction.parent.mkdir()
    prediction.write_text('{"frozen": true}\n')
    authorization = tmp_path / "authorization.json"
    record = {
        "schema": "lawlor-hca-pbmc-score-authorization/2.0",
        "status": "SEALED",
        "outcome_access_authorized": True,
        "prediction_path": "results/prediction.json",
        "prediction_sha256": _sha256(prediction),
        "prediction_bytes": prediction.stat().st_size,
        "prediction_public_url": url,
        "prediction_public_commit": commit,
        "runner_sha256": _sha256(Path(lawlor.__file__)),
        "protocol_sha256": _sha256(lawlor.PROTOCOL),
    }
    authorization.write_text(json.dumps(record))
    original_root = lawlor.ROOT
    lawlor.ROOT = tmp_path
    try:
        with pytest.raises(PermissionError, match=message):
            _require_score_authorization(prediction, authorization)
    finally:
        lawlor.ROOT = original_root


def _write_reducer_bundle(reduced: Path) -> None:
    reduced.mkdir()
    rows = []
    for index, relative in enumerate(lawlor.REDUCER_OUTPUTS):
        path = reduced / relative
        path.write_bytes(f"artifact-{index}\n".encode())
        rows.append(f"{relative}\t{path.stat().st_size}\t{_sha256(path)}")
    (reduced / lawlor.REDUCER_MANIFEST).write_text(
        "path\tbytes\tsha256\n" + "\n".join(rows) + "\n"
    )


def test_reducer_artifact_bundle_binds_outputs_and_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(lawlor, "ROOT", tmp_path)
    reduced = tmp_path / "data" / "reduced"
    reduced.parent.mkdir()
    _write_reducer_bundle(reduced)
    bundle = _reducer_artifact_bundle(reduced)
    assert bundle["reduced_root"] == "data/reduced"
    assert [record["path"] for record in bundle["artifacts"]] == [
        *lawlor.REDUCER_OUTPUTS,
        lawlor.REDUCER_MANIFEST,
    ]
    assert _require_reducer_artifact_bundle(reduced, bundle) == bundle

    (reduced / lawlor.REDUCER_MANIFEST).write_text(
        (reduced / lawlor.REDUCER_MANIFEST).read_text() + "\n"
    )
    with pytest.raises(PermissionError, match="paths, byte counts, or SHA-256"):
        _require_reducer_artifact_bundle(reduced, bundle)


def test_reducer_artifact_bundle_rejects_tampering_and_extra_files(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(lawlor, "ROOT", tmp_path)
    reduced = tmp_path / "reduced"
    _write_reducer_bundle(reduced)
    bundle = _reducer_artifact_bundle(reduced)

    (reduced / lawlor.REDUCER_OUTPUTS[0]).write_bytes(b"tampered\n")
    with pytest.raises(PermissionError, match="frozen prediction provenance"):
        _require_reducer_artifact_bundle(reduced, bundle)

    (reduced / lawlor.REDUCER_OUTPUTS[0]).write_bytes(b"artifact-0\n")
    (reduced / "unexpected.txt").write_text("not declared\n")
    with pytest.raises(PermissionError, match="frozen prediction provenance"):
        _require_reducer_artifact_bundle(reduced, bundle)


def test_promotion_gate_requires_beating_the_best_matched_field():
    entities = 12
    primary = np.linspace(-1.0, 1.0, entities * 4).reshape(entities, 4)
    arrays = {
        "field": np.zeros(
            (len(lawlor.DEVELOPMENT) + len(lawlor.HELD), entities, 2, 2)
        ),
        "pearson": np.zeros(
            (len(lawlor.DEVELOPMENT) + len(lawlor.HELD), entities, 3, 3)
        ),
        "deviance": np.zeros(
            (len(lawlor.DEVELOPMENT) + len(lawlor.HELD), entities, 3, 3)
        ),
    }
    arrays["field"][len(lawlor.DEVELOPMENT) :] = primary.reshape(entities, 2, 2)
    predictions = {
        "field_primary": primary,
        "field_direct": primary,
        "pearson_direct": np.zeros((entities, 9)),
        "pearson_structured": np.zeros((entities, 9)),
        "deviance_direct": np.zeros((entities, 9)),
        "deviance_structured": np.zeros((entities, 9)),
    }
    matched = (
        "field_direct",
        "field_zero",
        "field_scalar",
        "field_nuclear",
        "field_hypergraph",
        "field_endpoint_ridge",
        "covariance_direct",
        "field_membership_permuted",
    )
    losses = {
        "field_primary": np.full((4, entities), 0.2),
        "field_destroyed": np.full((4, entities), 0.5),
        "pearson_direct": np.full((4, entities), 0.5),
        "pearson_structured": np.full((4, entities), 0.5),
        "deviance_direct": np.full((4, entities), 0.5),
        "deviance_structured": np.full((4, entities), 0.5),
        **{name: np.full((4, entities), 0.4) for name in matched},
    }
    clusters = [f"marker-{index}" for index in range(entities)]
    summary, _ = summarize(
        arrays, predictions, losses, marker_clusters=clusters
    )
    assert summary["gate_passed"] is True

    losses["field_zero"][:] = 0.1
    summary, _ = summarize(
        arrays, predictions, losses, marker_clusters=clusters
    )
    assert summary["gate_passed"] is False
