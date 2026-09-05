"""Finite-support checks of the distribution-to-table regret corollary."""

import math

import numpy as np
import pytest
from scipy.optimize import linprog
from scipy.special import logsumexp, rel_entr

from mapreg.predictive_conditional import normal_mixture_prediction


def table_law(total, row, column, atoms, weights):
    support = np.arange(max(0, row + column - total), min(row, column) + 1)
    log_weight = np.log(
        [
            float(math.comb(column, int(a)) * math.comb(total - column, int(row - a)))
            for a in support
        ]
    )
    logits = log_weight[None, :] + np.asarray(atoms)[:, None] * support
    conditional = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
    probability = (np.asarray(weights)[:, None] * conditional).sum(axis=0)
    tables = np.array(
        [[[a, row - a], [column - a, total - row - column + a]] for a in support]
    )
    mean = (probability[:, None, None] * tables).sum(axis=0)
    return probability, tables, mean


def wasserstein_squared(atoms_g, weights_g, atoms_h, weights_h):
    cost = (np.asarray(atoms_g)[:, None] - atoms_h) ** 2
    rows = np.zeros((len(atoms_g) + len(atoms_h), cost.size))
    for i in range(len(atoms_g)):
        rows[i].reshape(cost.shape)[i, :] = 1
    for j in range(len(atoms_h)):
        rows[len(atoms_g) + j].reshape(cost.shape)[:, j] = 1
    result = linprog(
        cost.ravel(),
        A_eq=rows,
        b_eq=[*weights_g, *weights_h],
        bounds=(0, None),
        method="highs",
    )
    assert result.success
    return result.fun


def expected_deviance(probability, tables, prediction):
    terms = rel_entr(tables, prediction[None, :, :])
    losses = 2 * terms.sum(axis=(1, 2)) / tables.sum(axis=(1, 2))
    return float((probability * losses).sum())


@pytest.mark.parametrize(
    "total,row,column", [(2, 1, 1), (12, 6, 6), (20, 3, 16), (6, 0, 3)]
)
@pytest.mark.parametrize(
    "atoms_g,weights_g,atoms_h,weights_h",
    [
        ([-2.0, 2.0], [0.5, 0.5], [0.0], [1.0]),
        ([-1.0, 1.5], [0.2, 0.8], [-2.0, 0.4, 3.0], [0.3, 0.4, 0.3]),
        ([1.0], [1.0], [-1.0], [1.0]),
        ([-2.0, 2.0], [0.5, 0.5], [-2.0, 2.0], [0.5, 0.5]),
    ],
)
def test_finite_mixture_regret_chain(
    total, row, column, atoms_g, weights_g, atoms_h, weights_h
):
    pg, tables, qg = table_law(total, row, column, atoms_g, weights_g)
    ph, _, qh = table_law(total, row, column, atoms_h, weights_h)
    regret = expected_deviance(pg, tables, qh) - expected_deviance(pg, tables, qg)
    table_kl = 2 * rel_entr(qg / total, qh / total).sum()
    law_kl = 2 * rel_entr(pg, ph).sum()
    width = min(row, column) - max(0, row + column - total)
    bound = width**2 / 4 * wasserstein_squared(atoms_g, weights_g, atoms_h, weights_h)
    np.testing.assert_allclose(regret, table_kl, atol=1e-13)
    assert -1e-13 <= table_kl <= law_kl + 1e-13
    assert law_kl <= bound + 1e-13
    if width == 0:
        assert regret == table_kl == law_kl == bound == 0


def test_equal_mean_tables_need_not_mean_equal_mixing_laws():
    pg, _, qg = table_law(12, 6, 6, [-2, 2], [0.5, 0.5])
    ph, _, qh = table_law(12, 6, 6, [0], [1])
    np.testing.assert_allclose(qg, qh, atol=1e-12)
    assert rel_entr(pg, ph).sum() > 0
    assert wasserstein_squared([-2, 2], [0.5, 0.5], [0], [1]) == 4


@pytest.mark.parametrize(
    "mu_g,sd_g,mu_h,sd_h",
    [(1.0, 0.8, -0.4, 1.2), (0.0, 1.2, 0.0, 0.0), (1.0, 0.0, 1.0, 0.0)],
)
def test_normal_distribution_bound(mu_g, sd_g, mu_h, sd_h):
    rows, columns = np.array([5, 7]), np.array([8, 4])
    g = normal_mixture_prediction(mu_g, sd_g**2, rows, columns)
    h = normal_mixture_prediction(mu_h, sd_h**2, rows, columns)
    table_kl = 2 * rel_entr(g.mean_table / 12, h.mean_table / 12).sum()
    law_kl = 2 * rel_entr(g.probabilities, h.probabilities).sum()
    width = 5 - 1
    bound = width**2 / 4 * ((mu_g - mu_h) ** 2 + (sd_g - sd_h) ** 2)
    assert -1e-12 <= table_kl <= law_kl + 1e-12
    assert law_kl <= bound + 1e-12
