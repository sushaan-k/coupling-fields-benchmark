"""Frozen adaptation-only prediction and donor-level evaluation."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import rel_entr
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'assay_resolution'))
from reconstruct import reconstruct_four, independence_tables

METHODS = ('means_only', 'rna_patterns', 'protein_patterns', 'both_patterns', 'independence')
SUPPORT = ((np.arange(512)[:, None] >> np.arange(9)) & 1).astype(float)
MODEL = HERE.parent / 'assay_resolution/results/adaptation_only_results/source_fit.npz'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')


def check_freeze():
    frozen = json.loads((HERE / 'freeze.json').read_text())
    for name, expected in frozen['sha256'].items():
        assert digest(HERE / name) == expected, name


def binary(rna, protein, threshold=None):
    if threshold is None:
        threshold = np.median(protein, axis=0)
    return (rna > 0).astype(float), (protein > threshold).astype(float), threshold


def marginal(cells, prior):
    codes = np.sum(cells.astype(int) * 2**np.arange(9), axis=1)
    counts = np.bincount(codes, minlength=512)
    pseudocounts = 4 * np.prod(np.where(SUPPORT == 1, prior, 1-prior), axis=1)
    return (counts + pseudocounts) / (len(cells) + 4)


def query_tables(x, y):
    return np.asarray([np.asarray([np.mean((x[:, i] == a) & (y[:, j] == b))
                       for a in (0, 1) for b in (0, 1)]).reshape(2, 2)
                       for i in range(9) for j in range(9)])


def summaries(losses):
    draws = np.random.default_rng(9132026).integers(len(losses), size=(20000, len(losses)))
    comparisons = {}
    for control in (0, 2, 4):
        boot = 100 * (1 - losses[draws, 3].mean(1) / losses[draws, control].mean(1))
        comparisons[METHODS[control]] = {
            'reduction_percent': float(100 * (1-losses[:, 3].mean()/losses[:, control].mean())),
            'bootstrap_95': np.quantile(boot, [.025, .975]).tolist(),
            'bootstrap_familywise': np.quantile(boot, [.025/3, 1-.025/3]).tolist(),
            'donor_wins': int(np.sum(losses[:, 3] < losses[:, control]))}
    passed = all(v['bootstrap_familywise'][0] > 0 for v in comparisons.values())
    passed = passed and comparisons['means_only']['reduction_percent'] >= 10
    return {'n': len(losses), 'mean_losses': dict(zip(METHODS, losses.mean(0).tolist())),
            'comparisons': comparisons, 'generalization_criterion_passed': bool(passed)}


def predict(work):
    check_freeze()
    assert not (work / 'predictions.npz').exists()
    with np.load(MODEL) as model, np.load(work / 'adaptation_counts.npz') as counts:
        b, priors = model['b'], model['priors']
        rna, protein = counts['rna'], counts['protein']
    assert rna.shape == protein.shape == (11, 256, 9)
    outputs, thresholds, marginals = [], [], []
    for i, (xraw, yraw) in enumerate(zip(rna, protein)):
        x, y, threshold = binary(xraw, yraw)
        masses = np.array([marginal(x, priors[0]), marginal(y, priors[1])])
        four = reconstruct_four(b, SUPPORT, masses)
        outputs.append(np.concatenate([four, independence_tables(SUPPORT, masses)[None]]))
        thresholds.append(threshold)
        marginals.append(masses)
        print('predicted', i+1, '/11', flush=True)
    probabilities = np.array(outputs)
    assert np.isfinite(probabilities).all() and probabilities.min() > 0
    np.savez_compressed(work / 'predictions.npz', predictions=probabilities,
                        thresholds=thresholds, marginals=marginals, methods=METHODS)
    save(work / 'prediction_freeze.json', {'predictions_sha256': digest(work / 'predictions.npz'),
         'adaptation_sha256': digest(work / 'adaptation_counts.npz'), 'source_sha256': digest(MODEL)})


def evaluate(work):
    check_freeze()
    frozen = json.loads((work / 'prediction_freeze.json').read_text())
    assert frozen['predictions_sha256'] == digest(work / 'predictions.npz')
    assert not (work / 'evaluation.json').exists()
    with np.load(work / 'predictions.npz') as result, np.load(work / 'scoring_counts.npz') as counts:
        predictions, thresholds = result['predictions'], result['thresholds']
        truth = np.array([query_tables(*binary(x, y, t)[:2])
                          for x, y, t in zip(counts['rna'], counts['protein'], thresholds)])
    pair_losses = 2 * rel_entr(truth[:, None], predictions).sum((3, 4))
    losses = pair_losses.mean(2)
    result = summaries(losses)
    selection = json.loads((HERE / 'selection.json').read_text())
    conditions = np.array([p['condition'] for p in selection['patients']])
    result['descriptive_groups'] = {
        c: {'n': int(np.sum(conditions == c)),
            'mean_losses': dict(zip(METHODS, losses[conditions == c].mean(0).tolist()))}
        for c in sorted(set(conditions))}
    result['prediction_sha256'] = frozen['predictions_sha256']
    np.savez_compressed(work / 'losses.npz', truth=truth, losses=losses, pair_losses=pair_losses)
    save(work / 'evaluation.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('predict', 'evaluate'))
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        {'predict': predict, 'evaluate': evaluate}[args.stage](args.work)
