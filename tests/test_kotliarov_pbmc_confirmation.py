import hashlib
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "experiments/confirm_kotliarov_pbmc.py"
SPEC = importlib.util.spec_from_file_location("confirm_kotliarov_pbmc_tested", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
kotliarov = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = kotliarov
SPEC.loader.exec_module(kotliarov)

_artifact_bundle = kotliarov._artifact_bundle
_adaptive_best_sign_flip_p = kotliarov._adaptive_best_sign_flip_p
_field_table = kotliarov._field_table
_margin_stats_from_counts = kotliarov._margin_stats_from_counts
_require_github_commit_url = kotliarov._require_github_commit_url
_residual_table = kotliarov._residual_table
_sign_flip_p = kotliarov._sign_flip_p
_table_stats = kotliarov._table_stats
_weighted_quantile = kotliarov._weighted_quantile
summarize = kotliarov.summarize


def _states(table):
    first, second = [], []
    for row, column in np.ndindex(table.shape):
        first.extend([row] * int(table[row, column]))
        second.extend([column] * int(table[row, column]))
    return np.asarray(first), np.asarray(second)


def test_declared_split_excludes_cross_batch_donor_and_uses_donors_as_replicates():
    assert len(kotliarov.DEVELOPMENT) == 10
    assert len(kotliarov.HELD) == 9
    assert set(kotliarov.DEVELOPMENT).isdisjoint(kotliarov.HELD)
    assert kotliarov.EXCLUDED == ("209",)
    assert "209" not in kotliarov.DEVELOPMENT + kotliarov.HELD
    assert set(kotliarov.HELD_HIGH + kotliarov.HELD_LOW) == set(kotliarov.HELD)
    assert len(kotliarov.HELD_HIGH) == 4
    assert len(kotliarov.HELD_LOW) == 5
    assert kotliarov.PERMUTATIONS == 64
    assert kotliarov.BOOTSTRAPS == 10_000


def test_disabled_preflight_uses_bound_embedding_manifest_when_binary_is_omitted(
    monkeypatch, tmp_path,
):
    missing = tmp_path / "missing.npz"
    original_repo_relative = kotliarov._repo_relative
    monkeypatch.setattr(kotliarov, "SCGPT", missing)
    monkeypatch.setattr(
        kotliarov,
        "_repo_relative",
        lambda path: (
            "data/scgpt_gene_embeddings.npz"
            if path == missing
            else original_repo_relative(path)
        ),
    )

    observed = kotliarov.preflight(require_sealed=False)
    manifest = json.loads(kotliarov.SCGPT_MANIFEST.read_text())
    assert observed["embedding_sha256"] == manifest["output"]["sha256"]

    designation = json.loads(kotliarov.DESIGNATION.read_text())
    designation.update(
        {
            "status": "SEALED",
            "outcome_access_authorized": True,
            "public_freeze_commit": "a" * 40,
            "public_freeze_url": (
                "https://github.com/sushaan-k/coupling-fields-benchmark/commit/"
                + "a" * 40
            ),
        }
    )
    designation_path = tmp_path / "designation.json"
    designation_path.write_text(json.dumps(designation))
    monkeypatch.setattr(kotliarov, "DESIGNATION", designation_path)
    with pytest.raises(FileNotFoundError, match="checksum-matched scGPT embedding"):
        kotliarov.preflight(require_sealed=True)


def test_conditional_field_and_full_classical_residuals_use_identical_margins():
    table = np.asarray([[28, 4, 3], [5, 31, 4], [2, 6, 37]])
    stats = _table_stats(*_states(table), seed=8)
    predicted = _field_table(
        stats["field"], stats["field_null"], stats["rows"], stats["columns"]
    )
    np.testing.assert_allclose(predicted.sum(axis=1), table.sum(axis=1), atol=1e-8)
    np.testing.assert_allclose(predicted.sum(axis=0), table.sum(axis=0), atol=1e-8)
    assert np.all(predicted > 0)
    for family in ("pearson", "deviance"):
        predicted = _residual_table(
            stats[family],
            stats[f"{family}_null"],
            stats["total"],
            stats["rows"],
            stats["columns"],
            family,
        )
        np.testing.assert_allclose(predicted, table, atol=1e-7)
        assert predicted.shape == (3, 3)
    assert np.asarray(stats["pearson"]).shape == (9,)
    assert np.asarray(stats["deviance"]).shape == (9,)


def test_margin_reference_depends_only_on_frozen_margins():
    rows = np.asarray([25, 31, 37])
    columns = np.asarray([29, 35, 29])
    first = _margin_stats_from_counts(rows, columns, seed=19)
    second = _margin_stats_from_counts(rows.copy(), columns.copy(), seed=19)
    for family in ("field", "pearson", "deviance"):
        np.testing.assert_array_equal(first[f"{family}_null"], second[f"{family}_null"])
        np.testing.assert_array_equal(
            first[f"{family}_destroyed"], second[f"{family}_destroyed"]
        )
        np.testing.assert_array_equal(
            first[f"{family}_variance"], second[f"{family}_variance"]
        )


def test_donor_equal_weighted_tertiles_do_not_let_large_donor_dominate():
    small = np.asarray([0.0, 1.0, 2.0])
    large = np.repeat(np.asarray([10.0, 11.0, 12.0]), 100)
    values = np.concatenate((small, large))
    donors = np.asarray(["small"] * len(small) + ["large"] * len(large))
    weights = np.where(donors == "small", 1 / (2 * len(small)), 1 / (2 * len(large)))
    cuts = _weighted_quantile(values, weights)
    assert cuts[0] < 10.0
    assert cuts[1] > 2.0


def test_cut_ties_remain_in_the_lower_state_as_in_the_frozen_reducer():
    cuts = np.asarray([1.0, 2.0])
    values = np.asarray([0.0, 1.0, 1.5, 2.0, 3.0])
    np.testing.assert_array_equal(
        np.searchsorted(cuts, values, side="left"), [0, 0, 1, 1, 2]
    )


def test_lopo_refits_tertiles_without_the_held_development_donor(monkeypatch):
    cells = pd.DataFrame(
        {
            "donor": np.repeat(kotliarov.DEVELOPMENT, 6),
            "lineage": ["B"] * (6 * len(kotliarov.DEVELOPMENT)),
        }
    )
    prediction_index = np.arange(len(cells))
    values = np.concatenate(
        [np.arange(6, dtype=float) + 100 * index for index in range(10)]
    )[None, :]
    entities = pd.DataFrame(
        {
            "entity_id": ["m@@B"],
            "marker_index": [0],
            "marker_id": ["m"],
            "gene_symbol": ["G"],
            "adt_target": ["m"],
            "module": ["module"],
            "lineage": ["B"],
        }
    )
    captured = []

    def fake_stats(cells, entities, rna, adt, donors, seed_label):
        captured.append((seed_label, rna.copy(), adt.copy()))
        return {"held_out_index": np.asarray(0)}

    monkeypatch.setattr(kotliarov, "_stats_for_donors", fake_stats)
    data = {
        "cells": cells,
        "prediction_index": prediction_index,
        "entities": entities,
        "rna_values": values,
        "development_adt_values": values.copy(),
    }
    kotliarov._lopo_folds(data)
    assert len(captured) == 10
    # Holding out the lowest-valued donor moves the cuts above that donor's range.
    held_low_states = captured[0][1][0, :6]
    assert np.all(held_low_states == 0)
    # Holding out the highest-valued donor moves the cuts below that donor's range.
    held_high_states = captured[-1][1][0, -6:]
    assert np.all(held_high_states == 2)


def test_primary_and_structured_comparators_never_use_zero_boundary_penalties(
    monkeypatch,
):
    entities = pd.DataFrame(
        {
            "entity_id": ["m@@B"],
            "marker_index": [0],
            "marker_id": ["m"],
            "gene_symbol": ["G"],
            "adt_target": ["m"],
            "module": ["module"],
            "lineage": ["B"],
        }
    )
    calls = []

    def fake_graph(entities, *, permute_external_membership=False):
        return np.zeros((1, 1)), {"permuted": permute_external_membership}

    def fake_cv(folds, entities, family, laplacian, candidates):
        calls.append((family, tuple(candidates)))
        return tuple(candidates[0]), []

    def fake_fit(values, variance, laplacian, nuclear, graph):
        return values.copy(), {"nuclear_multiplier": nuclear, "graph_penalty": graph}

    monkeypatch.setattr(kotliarov, "_embedding_hypergraph", fake_graph)
    monkeypatch.setattr(kotliarov, "_cv_grid", fake_cv)
    monkeypatch.setattr(kotliarov, "_structured_fit", fake_fit)
    development = {}
    for family, width in (("field", 4), ("pearson", 9), ("deviance", 9)):
        development[family] = np.ones((10, 1, width))
        development[f"{family}_variance"] = np.ones((10, 1, width))
    development["field_destroyed"] = np.zeros((10, 1, 4))
    predictions, tuning = kotliarov._fit_predictions(
        {"entities": entities}, development, []
    )
    assert set(predictions) == set(kotliarov.FIELD_METHODS) | set(
        kotliarov.CLASSICAL_METHODS
    )
    # Full structured searches: primary, membership-permuted, Pearson, deviance.
    for _, candidates in (calls[0], calls[3], calls[4], calls[5]):
        assert all(nuclear > 0 and graph > 0 for nuclear, graph in candidates)
    assert all(nuclear > 0 and graph == 0 for nuclear, graph in calls[1][1])
    assert all(nuclear == 0 and graph > 0 for nuclear, graph in calls[2][1])
    assert tuning["selected"]["field_primary"][0] > 0
    assert tuning["selected"]["field_primary"][1] > 0


def _write_bundle(root: Path, outputs: tuple[str, ...], manifest_name: str) -> None:
    root.mkdir()
    records = []
    for index, relative in enumerate(outputs):
        path = root / relative
        path.write_bytes(f"payload-{index}\n".encode())
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    pd.DataFrame(records).to_csv(root / manifest_name, sep="\t", index=False)


def test_separate_manifests_byte_bind_prediction_and_score_payloads(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(kotliarov, "ROOT", tmp_path)
    prediction = tmp_path / "prediction"
    score = tmp_path / "score"
    _write_bundle(prediction, ("public.txt",), "prediction_manifest.tsv")
    _write_bundle(score, ("held.npy.gz",), "score_manifest.tsv")
    prediction_bundle = _artifact_bundle(
        prediction,
        ("public.txt",),
        "prediction_manifest.tsv",
        "prediction/1.0",
    )
    score_bundle = _artifact_bundle(
        score, ("held.npy.gz",), "score_manifest.tsv", "score/1.0"
    )
    assert prediction_bundle["artifacts"][0]["path"] == "public.txt"
    assert score_bundle["artifacts"][0]["path"] == "held.npy.gz"
    (score / "held.npy.gz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="does not match"):
        _artifact_bundle(
            score, ("held.npy.gz",), "score_manifest.tsv", "score/1.0"
        )


def test_public_binding_requires_exact_commit_and_blob_path():
    commit = "0123456789abcdef0123456789abcdef01234567"
    path = "results/kotliarov_pbmc_predictions.json"
    url = f"https://github.com/example/repo/blob/{commit}/{path}"
    _require_github_commit_url(url, commit, blob_path=path)
    with pytest.raises(PermissionError, match="exact artifact path"):
        _require_github_commit_url(url, commit, blob_path="results/other.json")
    with pytest.raises(PermissionError, match="40 lowercase"):
        _require_github_commit_url(url, commit[:-1], blob_path=path)


def _gate_fixture():
    marker_ids = [f"M{index}" for index in range(16) for _ in range(2)]
    entities = pd.DataFrame(
        {
            "entity_id": [f"{marker}@@{lineage}" for marker, lineage in zip(marker_ids, itertools.cycle(("B", "CD4 T")))],
            "marker_id": marker_ids,
            "gene_symbol": [f"G{index // 2}" for index in range(32)],
            "lineage": list(itertools.islice(itertools.cycle(("B", "CD4 T")), 32)),
        }
    )
    primary = np.linspace(-1, 1, 32 * 4).reshape(32, 4)
    truth = {"field": np.broadcast_to(primary, (9, 32, 4)).copy()}
    predictions = {
        name: (primary.copy() if name.startswith("field_") else np.zeros((32, 9)))
        for name in set(kotliarov.FIELD_METHODS) | set(kotliarov.CLASSICAL_METHODS)
    }
    losses = {
        name: np.full((9, 32), 0.65)
        for name in kotliarov.ALL_METHODS
    }
    losses["field_primary"][:] = 0.40
    losses["field_direct"][:] = 0.60
    losses["field_destroyed"][:] = 0.70
    for name in kotliarov.MATCHED_METHODS:
        if name != "field_direct":
            losses[name][:] = 0.58
    return truth, predictions, losses, entities


def test_strict_gate_requires_every_declared_condition(monkeypatch):
    monkeypatch.setattr(kotliarov, "BOOTSTRAPS", 200)
    truth, predictions, losses, entities = _gate_fixture()
    summary, bootstrap = summarize(
        truth,
        predictions,
        losses,
        entities,
        integrity_checks={"all_frozen_checks": True},
    )
    assert summary["gate_passed"] is True
    assert summary["favorable_donor_counts"]["positive_field_correlation"] == 9
    assert summary["exact_one_sided_sign_flip_p"]["mean_fisher_z"] == pytest.approx(
        1 / 512
    )
    assert len(bootstrap["mean_fisher_z"]) == 200
    losses["field_nuclear"][:] = 0.30
    summary, _ = summarize(
        truth,
        predictions,
        losses,
        entities,
        integrity_checks={"all_frozen_checks": True},
    )
    assert summary["best_matched_field"] == "field_nuclear"
    assert summary["gate_checks"]["best_matched_field"] is False
    assert summary["gate_passed"] is False


def test_exact_sign_flip_enumerates_all_512_assignments():
    values = -np.arange(1, 10, dtype=float)
    assert _sign_flip_p(values, "less") == pytest.approx(1 / 512)
    assert _sign_flip_p(-values, "greater") == pytest.approx(1 / 512)


def test_adaptive_sign_flip_reselects_best_comparator_in_every_assignment():
    first = -np.arange(1, 10, dtype=float)
    second = -np.arange(2, 11, dtype=float)
    observed_only = _sign_flip_p(
        np.asarray([first, second])[np.argmax([first.mean(), second.mean()])],
        "less",
    )
    adaptive = _adaptive_best_sign_flip_p(np.asarray([first, second]))
    assert adaptive >= observed_only
    assert 0 < adaptive <= 1
