import unittest
import numpy as np
from run import SUPPORT, binary, marginal, query_tables, summaries


class ProtocolTests(unittest.TestCase):
    def test_threshold_is_fixed_and_ties_are_low(self):
        x = np.ones((256, 9))
        y = np.tile(np.arange(256)[:, None], (1, 9))
        _, states, threshold = binary(x, y)
        np.testing.assert_array_equal(threshold, np.full(9, 127.5))
        np.testing.assert_array_equal(states.sum(0), np.full(9, 128))
        _, other, retained = binary(x, np.full_like(y, 1000), threshold)
        np.testing.assert_array_equal(retained, threshold)
        self.assertTrue(other.all())
        self.assertFalse(binary(x, np.ones_like(y))[1].any())

    def test_marginal_mass_and_permutation(self):
        cells = SUPPORT[:256]
        p = marginal(cells, np.full(9, .4))
        self.assertAlmostEqual(p.sum(), 1)
        self.assertGreater(p.min(), 0)
        np.testing.assert_array_equal(p, marginal(cells[::-1], np.full(9, .4)))

    def test_binary_query_table_order(self):
        x = np.tile([0, 0, 1, 1], (9, 1)).T
        y = np.tile([0, 1, 0, 1], (9, 1)).T
        np.testing.assert_array_equal(query_tables(x, y), np.full((81, 2, 2), .25))

    def test_paired_bootstrap_and_negative_control(self):
        losses = np.ones((11, 5))
        losses[:, 3] = .8
        result = summaries(losses)
        self.assertTrue(result['generalization_criterion_passed'])
        for value in result['comparisons'].values():
            np.testing.assert_allclose(value['bootstrap_familywise'], [20, 20])
        self.assertFalse(summaries(np.ones((11, 5)))['generalization_criterion_passed'])


if __name__ == '__main__':
    unittest.main()
