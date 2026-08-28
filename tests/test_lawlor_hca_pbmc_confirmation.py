import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import experiments.confirm_lawlor_hca_pbmc as lawlor
from experiments.confirm_lawlor_hca_pbmc import (
    _field_table,
    _margin_stats,
    _require_score_authorization,
    _sha256,
    _residual_table,
    _table_stats,
    summarize,
)


def _states(table):
    first, second = [], []
    for row, column in np.ndindex(table.shape):
        first.extend([row] * int(table[row, column]))
        second.extend([column] * int(table[row, column]))
    return np.asarray(first), np.asarray(second)


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
    state = np.tile(np.arange(3), len(cells) // 3)[None, :]
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
        cells, ["marker"], state, state, open_held_pairing=False
    )
    assert calls == len(lawlor.CONTRASTS) * (
        len(lawlor.DEVELOPMENT) * 2 + len(lawlor.HELD)
    )
    held = slice(len(lawlor.DEVELOPMENT), None)
    assert np.isnan(result["stimulus_table"][held]).all()
    assert np.isnan(result["field"][held]).all()
    assert np.isfinite(result["stimulus_rows"][held]).all()


def test_score_requires_the_exact_publicly_recorded_prediction_sha(tmp_path):
    prediction = tmp_path / "prediction.json"
    prediction.write_text('{"frozen": true}\n')
    authorization = tmp_path / "authorization.json"
    record = {
        "status": "SEALED",
        "outcome_access_authorized": True,
        "prediction_sha256": _sha256(prediction),
        "prediction_bytes": prediction.stat().st_size,
        "prediction_public_url": "https://example.org/immutable/prediction.json",
        "prediction_public_commit": "0123456789abcdef",
        "runner_sha256": _sha256(Path(lawlor.__file__)),
    }
    authorization.write_text(json.dumps(record))
    assert _require_score_authorization(prediction, authorization) == record

    record["prediction_sha256"] = "0" * 64
    authorization.write_text(json.dumps(record))
    with pytest.raises(PermissionError, match="hash differs"):
        _require_score_authorization(prediction, authorization)


def test_promotion_gate_requires_beating_the_best_matched_field():
    markers = 12
    primary = np.linspace(-1.0, 1.0, markers * 24).reshape(markers, 24)
    arrays = {
        "field": np.zeros((len(lawlor.DEVELOPMENT) + len(lawlor.HELD), markers, 6, 2, 2)),
        "pearson": np.zeros((len(lawlor.DEVELOPMENT) + len(lawlor.HELD), markers, 6, 3, 3)),
        "deviance": np.zeros((len(lawlor.DEVELOPMENT) + len(lawlor.HELD), markers, 6, 3, 3)),
    }
    arrays["field"][len(lawlor.DEVELOPMENT) :] = primary.reshape(markers, 6, 2, 2)
    predictions = {
        "field_primary": primary,
        "field_direct": primary,
        "pearson_direct": np.zeros((markers, 54)),
        "pearson_structured": np.zeros((markers, 54)),
        "deviance_direct": np.zeros((markers, 54)),
        "deviance_structured": np.zeros((markers, 54)),
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
        "field_primary": np.full((4, markers, 6), 0.2),
        "field_destroyed": np.full((4, markers, 6), 0.5),
        "pearson_direct": np.full((4, markers, 6), 0.5),
        "pearson_structured": np.full((4, markers, 6), 0.5),
        "deviance_direct": np.full((4, markers, 6), 0.5),
        "deviance_structured": np.full((4, markers, 6), 0.5),
        **{name: np.full((4, markers, 6), 0.4) for name in matched},
    }
    summary, _ = summarize(arrays, predictions, losses)
    assert summary["gate_passed"] is True

    losses["field_zero"][:] = 0.1
    summary, _ = summarize(arrays, predictions, losses)
    assert summary["gate_passed"] is False
