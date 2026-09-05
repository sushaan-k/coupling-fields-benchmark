# Fixed-margin prediction of cell-level RNA-surface-protein dependence across cohorts

This repository contains the reference implementation, public benchmark, and
reproducibility files for fixed-margin prediction of cross-modal dependence in
linked single-cell assays. The method estimates log-linear interaction from
linked source donors and combines that estimate with the observed margins of a
recipient sample. The margins describe the same cell sample with its cross-modal
pairing withheld. Reconstruction preserves those margins instead of copying
source marginal frequencies.

The public paper and interactive results are available at
<https://sushaan-k.github.io/coupling-fields-benchmark/>.

## Main result

The primary confirmation fitted the model on 36 Cambridge donors and evaluated
it on 56 Newcastle donors from the Stephenson CITE-seq study. Mean deviance per
cell was 0.01220, compared with 0.01478 for the development-selected
signed-deviance transfer. This is a 17.46% reduction. The paired-donor 95%
confidence interval for the loss difference was -0.00413 to -0.00080, and 50
of 56 donors favored fixed-margin prediction. A margin-preserving destroyed-link
control had mean deviance 0.02274.

A matched post hoc comparison favors calibrated normal random-effects
prediction. Its predictive mixture reduces deviance by 8.46% relative to the
original hierarchy (95% paired-donor interval, 5.42-11.55%) and by 11.79%
relative to common conditional estimation with the same ridge penalty
(7.43-16.10%). Integrating the fitted distribution improves on its population
plug-in by 1.33% (0.72-1.96%). The advantage persists after excluding two
recipients with almost no counts across the selected protein panel.

These results support classical random-effects estimation, calibration, and
predictive integration in this setting, not superiority of the original
hierarchy. Predictive count-interval coverage improves from 70.01% to 83.17%,
still below the nominal 95% level.

The benchmark also includes a corrected held-cohort analysis of GSE239452,
development studies, retrospective linked-assay panels, and every procedural
refusal from the public study sequence. These records have different evidence
roles and should not be pooled as independent confirmations.

## Release contents

The original version-2.0.3 benchmark contains 37 analysis records:

- 12 scored records
- 24 procedural refusals without a held performance value
- 1 infrastructure-unevaluable record without a scientific decision

The machine-readable files distinguish confirmatory, post-access,
development, retrospective, refused, and infrastructure-unevaluable analyses.
The main ledgers are:

- `results/benchmark_panels_v2.tsv`: one row per analysis record
- `results/benchmark_comparisons_v2.tsv`: matched method comparisons and uncertainty
- `results/benchmark_sequence_v2.tsv`: plan, access, prediction, scoring, and refusal chronology
- `benchmark_manifest.json`: release metadata and checksums for cited artifacts

`results/final_public_benchmark_table.tsv` is retained only as a historical
version-1 input. New analyses should use the version-2 ledgers.

Version 2.0.4 adds the predictive reanalysis, recipient-heterogeneity simulations,
composition-preserving controls, a source-defined protein-threshold sensitivity,
and a selected-panel quality-control sensitivity. These post hoc analyses are
stored separately in `results/development/`; they do not replace or increase the
number of original confirmations.

Small aggregate tables in `data/development/` support zero-download reanalysis.
The [replay instructions](docs/reanalysis_provenance/README.md) describe the
optional R dependencies and the exact random-effects comparison.

## Code layout

- `mapreg/`: estimators, fixed-margin table reconstruction, and classical comparators
- `experiments/`: dataset reduction, development, simulation, and held-evaluation programs
- `tests/`: numerical, protocol, and release-integrity tests
- `data/`: small derived inputs, source manifests, and access records
- `results/`: derived results and benchmark ledgers
- `docs/`: analysis plans, evidence ledger, and the paper website

The public API is exposed from `mapreg`. The primary binary estimator is in
`mapreg/hierarchical_conditional_coupling.py`; recipient tables are reconstructed
in `mapreg/table_prediction.py`.

For a normal distribution of recipient log odds, integrate the conditional
prediction instead of evaluating it at the mean parameter:

```python
from mapreg.predictive_conditional import normal_mixture_prediction

prediction = normal_mixture_prediction(
    mu=1.0, tau2=1.44, row_totals=[20, 108], column_totals=[100, 28]
)
table = prediction.mean_table
count_interval = prediction.count_interval(level=0.95)
```

The interval concerns the upper-left count and holds the fitted source
parameters fixed. `docs/FIXED_MARGIN_PREDICTIVE_REANALYSIS.md` specifies the
post hoc comparison with exact conditional random-effects estimation and the
cell-composition and threshold sensitivities. The original confirmation is
unchanged.

## Installation

Python 3.9 or newer is required.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

## Verify the release

The zero-download verification path checks every distributed checksum, rebuilds
the aggregate ledgers, and runs the packaged estimator and protocol tests:

```bash
shasum -a 256 -c SHA256SUMS
python -m experiments.build_public_benchmark_release --check
python -m scripts.verify_public_benchmark_release
./reproduce.sh
```

At the release tag, a clean clone can also verify the remote and tag binding:

```bash
python -m scripts.verify_public_benchmark_release --require-clean --require-tag
```

Some historical tests require source objects that are not redistributed. The
reproduction script identifies these tests and runs the available
checksum-bound alternatives. Raw public matrices remain at their original
repositories.

## Interpretation

The principal endpoint is prediction of a recipient joint table from linked
source data and recipient margins. It is not ordinary protein-abundance
prediction, causal effect estimation, or zero-shot perturbation ranking.

Cells are observations, not biological replicates. Confirmatory summaries give
equal weight to physical donors or samples. The version-2 ledgers report the
evaluation unit, uncertainty interval, comparator, decision, and evidence role
for each result. Failed support checks and negative comparisons remain in the
release so that the reported sequence can be reconstructed without selecting
panels by outcome.

## Data and license

Raw matrices are not redistributed. `data/manifests/sources.json` and the
study-specific source manifests record accessions, URLs, sizes, and checksums
when available. Derived artifacts retain study-provided, de-identified donor
labels, not names or direct participant identifiers.

The code is released under the MIT License. Source datasets retain their
original terms. The release has no assigned archive DOI and is not a
registry-hosted preregistration.
