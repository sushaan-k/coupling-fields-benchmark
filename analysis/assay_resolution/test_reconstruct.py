#!/usr/bin/env python3
"""Tests for the portable assay-resolution reconstruction."""

import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reconstruct import (
    aggregate,
    ipf,
    load_inputs,
    reconstruct_four,
    verify_bundle,
)


def product_mass(states, means):
    return np.prod(np.where(states == 1, means, 1 - means), axis=1)


class ReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.states = ((np.arange(8)[:, None] >> np.arange(3)) & 1).astype(float)

    def test_zero_interaction_is_independence(self):
        row = product_mass(self.states, np.array([0.2, 0.45, 0.8]))
        column = product_mass(self.states, np.array([0.35, 0.6, 0.7]))
        result = reconstruct_four(np.zeros((3, 3)), self.states, np.asarray((row, column)))
        np.testing.assert_allclose(
            result, np.repeat(result[0][None, :], 4, axis=0), rtol=0, atol=1e-9
        )
        tables = result[0].reshape(3, 3, 2, 2)
        xprob = tables[:, 0].sum(axis=2)
        yprob = tables[0].sum(axis=1)
        expected = np.einsum("ia,jb->ijab", xprob, yprob)
        np.testing.assert_allclose(tables, expected, rtol=0, atol=1e-9)

    def test_boundary_means_and_fixed_margins(self):
        row = product_mass(self.states, np.array([0.0, 0.4, 0.7]))
        column = product_mass(self.states, np.array([0.3, 1.0, 0.6]))
        interaction = np.arange(9, dtype=float).reshape(3, 3) / 10
        result = reconstruct_four(interaction, self.states, np.asarray((row, column)))
        self.assertTrue(np.isfinite(result).all())
        tables = result.reshape(4, 3, 3, 2, 2)
        xmean = np.einsum("u,ui->i", row, self.states, optimize=False)
        ymean = np.einsum("u,ui->i", column, self.states, optimize=False)
        expected_x = np.repeat(np.stack((1 - xmean, xmean), axis=1)[:, None, :], 3, axis=1)
        expected_y = np.repeat(np.stack((1 - ymean, ymean), axis=1)[None, :, :], 3, axis=0)
        for arm in tables:
            np.testing.assert_allclose(arm.sum(axis=3), expected_x, rtol=0, atol=1e-9)
            np.testing.assert_allclose(arm.sum(axis=2), expected_y, rtol=0, atol=1e-9)
        np.testing.assert_allclose(tables[:, 0, :, 1, :], 0, rtol=0, atol=1e-11)
        np.testing.assert_allclose(tables[:, :, 1, :, 0], 0, rtol=0, atol=1e-11)

    def test_support_permutations_do_not_change_full_reconstruction(self):
        rng = np.random.default_rng(6201)
        row = rng.dirichlet(np.ones(8))
        column = rng.dirichlet(np.ones(8))
        interaction = rng.normal(size=(3, 3))
        score = self.states @ interaction @ self.states.T
        original = aggregate(ipf(score, row, column), self.states, self.states)
        px, py = rng.permutation(8), rng.permutation(8)
        permuted = aggregate(
            ipf(score[np.ix_(px, py)], row[px], column[py]),
            self.states[px],
            self.states[py],
        )
        np.testing.assert_allclose(permuted, original, rtol=0, atol=1e-11)

    def test_released_first_person(self):
        report = verify_bundle(load_inputs(), limit=1)
        self.assertEqual(report["status"], "passed")
        self.assertLess(report["maximum_absolute_error"], 1e-8)


if __name__ == "__main__":
    unittest.main()
