#!/usr/bin/env python3
"""Recompute reported summaries from the released derived numerical arrays."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import rel_entr
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
COHORTS = ("Cambridge", "Newcastle", "T-ALL")
RETAINED_GAIN_BOOTSTRAP_95 = {
    "Cambridge": (89.08681633895621, 94.03079832004514),
    "Newcastle": (96.2776157298557, 98.37605422200372),
    "T-ALL": (93.41823583662247, 97.2468719982872),
}


def read_json(folder: str) -> dict[str, Any]:
    return json.loads((RESULTS / folder / "evaluation.json").read_text())


def load_npz(folder: str, name: str) -> dict[str, np.ndarray]:
    with np.load(RESULTS / folder / name, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def assert_close(actual: Any, expected: Any, path: str = "root") -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), path
        for key in expected:
            assert_close(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), path
        if expected and isinstance(expected[0], (dict, list)):
            for index, item in enumerate(expected):
                assert_close(actual[index], item, f"{path}[{index}]")
        else:
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12, err_msg=path)
    elif isinstance(expected, float):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12, err_msg=path)
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def loss(truth: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    axes = tuple(range(2, predictions.ndim))
    return 2 * rel_entr(truth[:, None], predictions).sum(axes) / 81


def percent_comparison(
    values: np.ndarray,
    winner: int,
    baseline: int,
    draws: np.ndarray,
    family_tail: float,
    win_epsilon: float,
) -> dict[str, Any]:
    bootstrap = 100 * (
        1
        - values[draws, winner].mean(axis=1)
        / values[draws, baseline].mean(axis=1)
    )
    return {
        "reduction_percent": float(
            100 * (1 - values[:, winner].mean() / values[:, baseline].mean())
        ),
        "bootstrap_95": np.quantile(bootstrap, (0.025, 0.975)).tolist(),
        "bootstrap_familywise": np.quantile(
            bootstrap, (family_tail, 1 - family_tail)
        ).tolist(),
        "donor_wins": int(
            (values[:, winner] < values[:, baseline] - win_epsilon).sum()
        ),
    }


def verify_main() -> int:
    predictions = load_npz("results", "predictions.npz")
    saved = load_npz("results", "losses.npz")
    methods = predictions["methods"].tolist()
    cohorts = predictions["cohorts"]
    recomputed = loss(saved["truth"], predictions["predictions"])
    np.testing.assert_array_equal(recomputed, saved["losses"])

    expected = read_json("results")
    actual = {"evidence_status": expected["evidence_status"], "cohorts": {}}
    comparisons = 0
    for cohort in COHORTS:
        values = recomputed[cohorts == cohort]
        draws = np.random.default_rng(906091).integers(
            0, len(values), (20000, len(values))
        )
        summaries = {}
        for baseline, name in enumerate(methods[:-1]):
            summaries[name] = percent_comparison(
                values, 3, baseline, draws, 0.05 / 18, 1e-12
            )
            comparisons += 1
        actual["cohorts"][cohort] = {
            "mean_losses": dict(zip(methods, values.mean(axis=0).tolist())),
            "donor_losses": values.tolist(),
            "comparisons": summaries,
        }
    assert_close(actual, expected, "main")
    return comparisons


def verify_mechanism(
    main: dict[str, np.ndarray],
) -> tuple[int, dict[str, list[float]]]:
    released = load_npz("mechanism_results", "predictions.npz")
    saved = load_npz("mechanism_results", "losses.npz")["losses"]
    cohorts = released["cohorts"]
    reconstructed = 2 * rel_entr(
        main["truth"], released["predictions"]
    ).sum((1, 2, 3)) / 81
    np.testing.assert_array_equal(reconstructed, saved[:, 1])
    np.testing.assert_array_equal(saved[:, (0, 2)], main["losses"][:, (0, 3)])

    expected = read_json("mechanism_results")
    actual = {"evidence_status": expected["evidence_status"], "cohorts": {}}
    names = ("means_only", "covariance_preserving", "both_patterns")
    comparisons = 0
    retained_intervals = {}
    for cohort in COHORTS:
        values = saved[cohorts == cohort]
        draws = np.random.default_rng(906092).integers(
            0, len(values), (20000, len(values))
        )
        bootstrap_means = values[draws].mean(axis=1)
        retained = 100 * (
            (bootstrap_means[:, 0] - bootstrap_means[:, 1])
            / (bootstrap_means[:, 0] - bootstrap_means[:, 2])
        )
        retained_intervals[cohort] = np.quantile(
            retained, (0.025, 0.975)
        ).tolist()
        np.testing.assert_allclose(
            retained_intervals[cohort],
            RETAINED_GAIN_BOOTSTRAP_95[cohort],
            rtol=0,
            atol=1e-12,
            err_msg=f"{cohort} retained-gain bootstrap interval",
        )
        summary = {}
        for name, winner, baseline in (
            ("full_vs_covariance", 2, 1),
            ("covariance_vs_means", 1, 0),
        ):
            summary[name] = percent_comparison(
                values, winner, baseline, draws, 0.05 / 12, 1e-12
            )
            comparisons += 1
        means = values.mean(axis=0)
        actual["cohorts"][cohort] = {
            "n": len(values),
            "mean_losses": dict(zip(names, means.tolist())),
            "fraction_gain_retained": float(
                (means[0] - means[1]) / (means[0] - means[2])
            ),
            "comparisons": summary,
        }
    assert_close(actual, expected, "mechanism")
    return comparisons, retained_intervals


def verify_context(main: dict[str, np.ndarray]) -> int:
    released = load_npz("context_results", "predictions.npz")
    saved = load_npz("context_results", "losses.npz")["losses"]
    personal = load_npz("mechanism_results", "losses.npz")["losses"][:, 1]
    pooled = loss(main["truth"], released["predictions"])
    np.testing.assert_array_equal(saved, np.column_stack((personal, pooled)))

    expected = read_json("context_results")
    actual = {"evidence_status": expected["evidence_status"], "cohorts": {}}
    cohorts = released["cohorts"]
    names = ("recipient_covariance", "source_pool", "other_recipient_pool")
    comparisons = 0
    for cohort in COHORTS:
        values = saved[cohorts == cohort]
        draws = np.random.default_rng(906093).integers(
            0, len(values), (20000, len(values))
        )
        summary = {}
        for baseline, name in enumerate(names[1:], 1):
            summary[name] = percent_comparison(
                values, 0, baseline, draws, 0.05 / 12, 1e-12
            )
            comparisons += 1
        actual["cohorts"][cohort] = {
            "n": len(values),
            "mean_losses": dict(zip(names, values.mean(axis=0).tolist())),
            "comparisons": summary,
        }
    assert_close(actual, expected, "context")
    return comparisons


def ranked_groups(
    scores: np.ndarray, gains: np.ndarray, size: int = 27
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-scores, axis=1, kind="stable")
    ordered = np.take_along_axis(gains, order, axis=1)
    return ordered[:, :size].mean(axis=1), ordered[:, -size:].mean(axis=1), ordered


def verify_relationship(main: dict[str, np.ndarray]) -> int:
    score_data = load_npz("relationship_results", "scores.npz")
    saved_gain = load_npz("relationship_results", "gains.npz")["gain"]
    personal = load_npz("mechanism_results", "predictions.npz")["predictions"]
    pooled = load_npz("context_results", "predictions.npz")["predictions"][:, 1]
    recomputed_gain = 2 * (
        rel_entr(main["truth"], pooled) - rel_entr(main["truth"], personal)
    ).sum((2, 3))
    np.testing.assert_array_equal(recomputed_gain, saved_gain)

    scores = score_data["scores"]
    cohorts = score_data["cohorts"]
    top, bottom, ordered = ranked_groups(scores[:, 0], saved_gain)
    direct, _, direct_ordered = ranked_groups(scores[:, 1], saved_gain)
    expected = read_json("relationship_results")
    actual = {"evidence_status": expected["evidence_status"], "cohorts": {}}
    comparisons = 0
    for cohort in COHORTS:
        mask = cohorts == cohort
        n = int(mask.sum())
        draws = np.random.default_rng(906094).integers(0, n, (20000, n))
        contrast_summary = {}
        for name, delta in (
            ("top_minus_bottom", top - bottom),
            ("top_minus_direct", top - direct),
        ):
            values = delta[mask]
            bootstrap = values[draws].mean(axis=1)
            contrast_summary[name] = {
                "mean": float(values.mean()),
                "bootstrap_95": np.quantile(bootstrap, (0.025, 0.975)).tolist(),
                "bootstrap_familywise": np.quantile(
                    bootstrap, (0.05 / 12, 1 - 0.05 / 12)
                ).tolist(),
                "donor_wins": int((values > 1e-12).sum()),
            }
            comparisons += 1
        actual["cohorts"][cohort] = {
            "n": n,
            "contrasts": contrast_summary,
            "mean_gain_all_pairs": float(saved_gain[mask].mean()),
            "mean_gain_top_third": float(top[mask].mean()),
            "mean_gain_bottom_third": float(bottom[mask].mean()),
            "mean_gain_direct_top_third": float(direct[mask].mean()),
            "cumulative_mean_gain": {
                str(k): {
                    "mechanism": float(ordered[mask, :k].mean()),
                    "direct": float(direct_ordered[mask, :k].mean()),
                }
                for k in range(9, 82, 9)
            },
            "pair_mean_gain": saved_gain[mask].mean(axis=0).tolist(),
            "pair_mean_score": scores[mask, 0].mean(axis=0).tolist(),
        }
    assert_close(actual, expected, "relationship")
    return comparisons


def verify_continuation() -> int:
    released = load_npz("continuation_results", "continuation.npz")
    relationship_scores = load_npz("relationship_results", "scores.npz")["scores"][:, 0]
    np.testing.assert_array_equal(released["scores"], relationship_scores)
    expected = read_json("continuation_results")
    actual_cohorts: dict[str, Any] = {}
    for cohort in COHORTS:
        mask = released["cohorts"] == cohort
        cohort_scores = released["scores"][mask]
        cohort_discrepancy = released["discrepancy"][:, mask]
        n = int(mask.sum())
        draws = np.random.default_rng(906095).integers(0, n, (20000, n))
        rows = []
        for t, exact in zip(released["grid"], cohort_discrepancy):
            within = np.array(
                [spearmanr(score, value).statistic for score, value in zip(cohort_scores, exact)]
            )
            overlap = np.array(
                [
                    len(
                        set(np.argsort(-score, kind="stable")[:27])
                        & set(np.argsort(-value, kind="stable")[:27])
                    )
                    / 27
                    for score, value in zip(cohort_scores, exact)
                ]
            )
            rows.append(
                {
                    "t": float(t),
                    "pooled_spearman": float(
                        spearmanr(cohort_scores.ravel(), exact.ravel()).statistic
                    ),
                    "mean_within_donor_spearman": float(within.mean()),
                    "within_donor_spearman_bootstrap_95": np.quantile(
                        within[draws].mean(axis=1), (0.025, 0.975)
                    ).tolist(),
                    "mean_top27_overlap": float(overlap.mean()),
                    "top27_overlap_bootstrap_95": np.quantile(
                        overlap[draws].mean(axis=1), (0.025, 0.975)
                    ).tolist(),
                    "exact_to_quadratic_ratio": float(
                        exact.sum() / (t * t * cohort_scores.sum())
                    ),
                }
            )
        actual_cohorts[cohort] = {"n": n, "grid": rows}
    assert_close(actual_cohorts, expected["cohorts"], "continuation.cohorts")
    assert expected["t1_maximum_prediction_error"] <= 1e-12
    return sum(len(item["grid"]) for item in actual_cohorts.values())


def verify_adaptation_only() -> int:
    predictions = load_npz("adaptation_only_results", "predictions.npz")
    saved = load_npz("adaptation_only_results", "losses.npz")
    recomputed = loss(saved["truth"], predictions["predictions"])
    np.testing.assert_array_equal(recomputed, saved["losses"])
    methods = predictions["methods"].tolist()
    cohorts = predictions["cohorts"]
    actual: dict[str, Any] = {}
    comparisons = 0
    for cohort in COHORTS:
        values = recomputed[cohorts == cohort]
        draws = np.random.default_rng(906097).integers(
            len(values), size=(20000, len(values))
        )
        summary = {}
        for baseline in (0, 2, 4):
            summary[methods[baseline]] = percent_comparison(
                values, 3, baseline, draws, 0.025 / 9, 0
            )
            comparisons += 1
        actual[cohort] = {
            "n": len(values),
            "mean_losses": dict(zip(methods, values.mean(axis=0).tolist())),
            "comparisons": summary,
        }
    expected = read_json("adaptation_only_results")
    assert actual == expected, "strict adaptation-only evaluation is not an exact match"
    source_fit = load_npz("adaptation_only_results", "source_fit.npz")
    assert source_fit["b"].shape == (9, 9) and source_fit["priors"].shape == (2, 9)
    assert all(np.isfinite(value).all() for value in source_fit.values())
    assert comparisons == 9
    return comparisons


def main() -> None:
    main_arrays = load_npz("results", "losses.npz")
    mechanism_comparisons, retained_intervals = verify_mechanism(main_arrays)
    report = {
        "status": "passed",
        "scope": "numerical reanalysis of released predictions and derived arrays; no raw-data fitting",
        "people": int(main_arrays["losses"].shape[0]),
        "marker_pairs": int(main_arrays["truth"].shape[1]),
        "paired_bootstrap_comparisons": {
            "main": verify_main(),
            "mechanism": mechanism_comparisons,
            "recipient_context": verify_context(main_arrays),
            "query_ranking": verify_relationship(main_arrays),
            "strict_adaptation_only_exact": verify_adaptation_only(),
        },
        "retained_gain_bootstrap_95_percent": retained_intervals,
        "continuation_cohort_grid_rows": verify_continuation(),
        "donor_barcodes_released": False,
        "raw_counts_released": False,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
