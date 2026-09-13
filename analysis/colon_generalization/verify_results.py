"""Reconstruct the released predictions and recompute the frozen endpoint."""

import json
from pathlib import Path

import numpy as np
from scipy.special import rel_entr
from threadpoolctl import threadpool_limits

from run import MODEL, SUPPORT, check_freeze, digest, independence_tables, reconstruct_four, summaries


def main():
    check_freeze()
    work = Path(__file__).resolve().parent / 'results'
    frozen = json.loads((work / 'prediction_freeze.json').read_text())
    assert digest(work / 'predictions.npz') == frozen['predictions_sha256']
    assert digest(MODEL) == frozen['source_sha256']
    with np.load(MODEL) as model, np.load(work / 'predictions.npz') as saved:
        probabilities, marginals = saved['predictions'], saved['marginals']
        reconstructed = np.array([
            np.concatenate([reconstruct_four(model['b'], SUPPORT, masses),
                            independence_tables(SUPPORT, masses)[None]])
            for masses in marginals])
    prediction_error = float(np.max(np.abs(reconstructed-probabilities)))
    assert prediction_error < 1e-8
    with np.load(work / 'losses.npz') as saved:
        truth = saved['truth']
        assert truth.shape == (11, 81, 2, 2)
        np.testing.assert_allclose(truth.sum((2, 3)), 1, atol=1e-12, rtol=0)
        pair_losses = 2 * rel_entr(truth[:, None], reconstructed).sum((3, 4))
        losses = pair_losses.mean(2)
        np.testing.assert_allclose(pair_losses, saved['pair_losses'], atol=1e-8, rtol=0)
        np.testing.assert_allclose(losses, saved['losses'], atol=1e-8, rtol=0)
    actual = summaries(losses)
    expected = json.loads((work / 'evaluation.json').read_text())
    for method, value in actual['mean_losses'].items():
        np.testing.assert_allclose(value, expected['mean_losses'][method], atol=1e-8, rtol=0)
    for method, comparison in actual['comparisons'].items():
        for name, value in comparison.items():
            np.testing.assert_allclose(value, expected['comparisons'][method][name], atol=1e-6, rtol=0)
    assert actual['generalization_criterion_passed'] == expected['generalization_criterion_passed']
    print(json.dumps({'status': 'passed', 'reconstructed_predictions': 55,
                      'maximum_probability_difference': prediction_error,
                      'verified_comparisons': 3}, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
