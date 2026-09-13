"""Export all donor and query losses without selecting favorable results."""

import csv
import json
from pathlib import Path

import numpy as np

from run import METHODS


def main():
    here = Path(__file__).resolve().parent
    selection = json.loads((here / 'selection.json').read_text())
    with np.load(here / 'results/losses.npz') as result:
        losses, pairs = result['losses'], result['pair_losses']
    with (here / 'results/donor_losses.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['donor', 'condition', *METHODS])
        for donor, values in zip(selection['patients'], losses):
            writer.writerow([donor['donor'], donor['condition'], *values])
    markers = selection['selected_feature_names']['RNA']
    with (here / 'results/query_losses.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['rna', 'protein', *METHODS, 'both_minus_means'])
        for index, values in enumerate(pairs.mean(0).T):
            writer.writerow([markers[index//9], markers[index%9], *values,
                             values[3]-values[0]])


if __name__ == '__main__':
    main()
