#!/usr/bin/env python3
"""Reconstruct released assay-resolution predictions from sufficient marginals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, root
from scipy.special import logsumexp


HERE = Path(__file__).resolve().parent
DEFAULT_INPUTS = HERE / "reconstruction_inputs.npz"
DEFAULT_RESULTS = HERE / "results"


def ipf(score: np.ndarray, row: np.ndarray, column: np.ndarray) -> np.ndarray:
    """Scale a positive kernel to fixed positive row and column marginals."""
    kernel = np.exp(score - np.max(score))
    left = np.ones(len(row))
    right = np.ones(len(column))
    error = np.inf
    for iteration in range(10000):
        left = row / np.einsum("ij,j->i", kernel, right, optimize=False)
        right = column / np.einsum("ij,i->j", kernel, left, optimize=False)
        if iteration % 5 == 0:
            joint = left[:, None] * kernel * right[None, :]
            error = max(
                float(np.max(np.abs(joint.sum(axis=1) - row))),
                float(np.max(np.abs(joint.sum(axis=0) - column))),
            )
            if error < 1e-12:
                return joint
    raise RuntimeError(f"IPF did not converge; marginal error {error}")


def compatible(states: np.ndarray, means: np.ndarray) -> np.ndarray:
    """Retain states consistent with marker means fixed at zero or one."""
    keep = np.ones(len(states), dtype=bool)
    keep &= np.all(states[:, means <= 1e-14] == 0, axis=1)
    keep &= np.all(states[:, means >= 1 - 1e-14] == 1, axis=1)
    return keep


def fit_joint(
    score: np.ndarray,
    xstates: np.ndarray,
    ystates: np.ndarray,
    xmean: np.ndarray,
    ymean: np.ndarray,
    xmass: np.ndarray | None = None,
    ymass: np.ndarray | None = None,
) -> np.ndarray:
    """Fit complete marginals when supplied and marker means otherwise."""
    if xmass is not None and ymass is not None:
        return ipf(score, xmass, ymass)
    if ymass is not None:
        return fit_joint(
            score.T, ystates, xstates, ymean, xmean, ymass, xmass
        ).T

    active_y = (ymean > 1e-14) & (ymean < 1 - 1e-14)
    if xmass is not None:
        features = ystates[:, active_y]

        def objective(beta: np.ndarray, output: bool = False):
            logits = score + np.einsum("vi,i->v", features, beta, optimize=False)
            normalizer = logsumexp(logits, axis=1)
            joint = xmass[:, None] * np.exp(logits - normalizer[:, None])
            gradient = np.einsum(
                "v,vi->i", joint.sum(axis=0), features, optimize=False
            ) - ymean[active_y]
            value = float(
                np.sum(xmass * normalizer) - np.sum(ymean[active_y] * beta)
            )
            return joint if output else (value, gradient)

        size = int(active_y.sum())
    else:
        active_x = (xmean > 1e-14) & (xmean < 1 - 1e-14)
        xf = xstates[:, active_x]
        yf = ystates[:, active_y]
        target = np.concatenate((xmean[active_x], ymean[active_y]))
        nx = xf.shape[1]

        def objective(beta: np.ndarray, output: bool = False):
            logits = (
                score
                + np.einsum("ui,i->u", xf, beta[:nx], optimize=False)[:, None]
                + np.einsum("vi,i->v", yf, beta[nx:], optimize=False)[None, :]
            )
            normalizer = logsumexp(logits)
            joint = np.exp(logits - normalizer)
            gradient = np.concatenate(
                (
                    np.einsum("u,ui->i", joint.sum(axis=1), xf, optimize=False),
                    np.einsum("v,vi->i", joint.sum(axis=0), yf, optimize=False),
                )
            ) - target
            value = float(normalizer - np.sum(target * beta))
            return joint if output else (value, gradient)

        size = len(target)

    beta = np.zeros(size)
    if size:
        fitted = minimize(
            objective,
            beta,
            jac=True,
            method="BFGS",
            options={"gtol": 2e-10, "maxiter": 1000},
        )
        beta = fitted.x
        if np.max(np.abs(objective(beta)[1])) > 1e-10:
            beta = root(
                lambda value: objective(value)[1],
                beta,
                method="hybr",
                options={"xtol": 1e-10},
            ).x
    joint = objective(beta, output=True)
    if np.max(np.abs(objective(beta)[1])) > 1e-8:
        raise RuntimeError("Moment reconstruction did not converge")
    return joint


def aggregate(
    joint: np.ndarray, xstates: np.ndarray, ystates: np.ndarray
) -> np.ndarray:
    """Aggregate a fine coupling into all marker-pair binary tables."""
    xmean = np.einsum("u,ui->i", joint.sum(axis=1), xstates, optimize=False)
    ymean = np.einsum("v,vi->i", joint.sum(axis=0), ystates, optimize=False)
    both = np.einsum("ui,uv,vj->ij", xstates, joint, ystates, optimize=False)
    tables = np.stack(
        (
            1 - xmean[:, None] - ymean[None, :] + both,
            ymean[None, :] - both,
            xmean[:, None] - both,
            both,
        ),
        axis=-1,
    )
    return tables.reshape(-1, 2, 2)


def reconstruct_arm(
    interaction: np.ndarray,
    states: np.ndarray,
    row: np.ndarray,
    column: np.ndarray,
    full_row: bool,
    full_column: bool,
) -> np.ndarray:
    xmean = np.einsum("u,ui->i", row, states, optimize=False)
    ymean = np.einsum("u,ui->i", column, states, optimize=False)
    xkeep = row > 0 if full_row else compatible(states, xmean)
    ykeep = column > 0 if full_column else compatible(states, ymean)
    xs = states[xkeep]
    ys = states[ykeep]
    xmass = row[xkeep] if full_row else None
    ymass = column[ykeep] if full_column else None
    score = np.einsum("ui,ij,vj->uv", xs, interaction, ys, optimize=False)
    joint = fit_joint(score, xs, ys, xmean, ymean, xmass, ymass)
    tables = aggregate(joint, xs, ys)
    if not np.isfinite(tables).all() or tables.min() < -1e-10:
        raise RuntimeError("Reconstruction produced an invalid query table")
    return np.maximum(tables, 0)


def reconstruct_four(
    interaction: np.ndarray, states: np.ndarray, marginals: np.ndarray
) -> np.ndarray:
    row, column = marginals
    return np.asarray(
        [
            reconstruct_arm(interaction, states, row, column, full_row, full_column)
            for full_row, full_column in (
                (False, False),
                (True, False),
                (False, True),
                (True, True),
            )
        ]
    )


def independence_tables(
    states: np.ndarray, marginals: np.ndarray
) -> np.ndarray:
    xmean = np.einsum("u,ui->i", marginals[0], states, optimize=False)
    ymean = np.einsum("u,ui->i", marginals[1], states, optimize=False)
    both = xmean[:, None] * ymean[None, :]
    tables = np.stack(
        (
            1 - xmean[:, None] - ymean[None, :] + both,
            ymean[None, :] - both,
            xmean[:, None] - both,
            both,
        ),
        axis=-1,
    )
    return tables.reshape(-1, 2, 2)


def load_inputs(path: Path = DEFAULT_INPUTS) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def load_predictions(results: Path, folder: str) -> np.ndarray:
    with np.load(results / folder / "predictions.npz", allow_pickle=False) as archive:
        return archive["predictions"]


def verify_bundle(
    inputs: dict[str, np.ndarray],
    limit: int | None = None,
    results: Path = DEFAULT_RESULTS,
) -> dict[str, object]:
    states = inputs["states"]
    expected = {
        "conditional": load_predictions(results, "results"),
        "covariance": load_predictions(results, "mechanism_results"),
        "pooled": load_predictions(results, "context_results"),
        "strict": load_predictions(results, "adaptation_only_results"),
    }
    count = len(inputs["conditional_marginals"])
    if limit is not None:
        count = min(count, limit)
    errors = {"conditional": 0.0, "covariance": 0.0, "pooled": 0.0, "strict": 0.0}

    for index in range(count):
        conditional = reconstruct_four(
            inputs["conditional_b"], states, inputs["conditional_marginals"][index]
        )
        covariance = reconstruct_arm(
            inputs["conditional_b"],
            states,
            *inputs["covariance_marginals"][index],
            True,
            True,
        )
        pooled = np.asarray(
            [
                reconstruct_arm(inputs["conditional_b"], states, *marginals, True, True)
                for marginals in inputs["pooled_marginals"][index]
            ]
        )
        strict_four = reconstruct_four(
            inputs["strict_b"], states, inputs["strict_marginals"][index]
        )
        strict = np.concatenate(
            (
                strict_four,
                independence_tables(states, inputs["strict_marginals"][index])[None, :],
            ),
            axis=0,
        )
        for name, actual, reference in (
            ("conditional", conditional, expected["conditional"][index]),
            ("covariance", covariance, expected["covariance"][index]),
            ("pooled", pooled, expected["pooled"][index]),
            ("strict", strict, expected["strict"][index]),
        ):
            errors[name] = max(errors[name], float(np.max(np.abs(actual - reference))))
        if (index + 1) % 10 == 0:
            print(f"reconstructed {index + 1}/{count}", flush=True)

    maximum = max(errors.values())
    if maximum >= 1e-8:
        raise AssertionError(f"Maximum reconstruction error {maximum} exceeds 1e-8")
    return {
        "status": "passed",
        "people": count,
        "reconstructions_per_person": 12,
        "maximum_absolute_errors": errors,
        "maximum_absolute_error": maximum,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            verify_bundle(load_inputs(args.inputs), args.limit, args.results), indent=2
        )
    )


if __name__ == "__main__":
    main()
