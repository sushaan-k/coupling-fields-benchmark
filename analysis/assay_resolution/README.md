# Assay-resolution numerical verification

This folder releases the derived prediction tables, donor-level loss arrays,
query scores, finite-strength discrepancies, and evaluation summaries used to
check the reported assay-resolution analyses. It does not contain the original
raw cell-level counts or donor barcodes.

The verifier performs numerical reanalysis of the released predictions and
derived donor arrays. It recomputes the reported losses, seeded paired-donor
bootstrap summaries, covariance-retained fractions, query-ranking contrasts,
cumulative ranked gains, and finite-strength rank and overlap diagnostics. It
also recomputes the paired-bootstrap 95% intervals for the retained-gain
fractions and reproduces all nine strict adaptation-only comparisons exactly.
This is not a rerun of preprocessing, source-model fitting, histogram
construction, or prediction fitting from raw data.

Install the two numerical dependencies and run:

```bash
python3 -m pip install -r requirements.txt
python3 verify_reported_results.py
```

A successful run prints a JSON report with `"status": "passed"`. The
`results/` subdirectories mirror the analysis stages. Their NPZ files omit
donor identifiers, and their `evaluation.json` files omit donor-barcode lists.
The retained row order is consistent across files and cohort labels are kept so
that the published cohort-level summaries can be recomputed. This
prediction-level release contains 32 named arrays across 12 result NPZ files.

The standalone reconstruction path reconstructs 12 arms per person: four
conditional arms, one covariance arm, two pooled arms, four strict
adaptation-only arms, and one independence arm. `reconstruction_inputs.npz`
contains only sufficient marginal masses, canonical binary states, and two
fitted interaction matrices. Reference probabilities come from the sibling
released `results/` tree. The reconstruction input contains no raw counts,
identifiers, cohort labels, or filesystem paths; the result files retain cohort
labels but omit donor identifiers. Run the implementation and tests with:

```bash
python3 reconstruct.py
python3 -m unittest -v test_reconstruct.py
```

The [paper page](https://sushaan-k.github.io/coupling-fields-benchmark/) links
the combined manuscript and analysis archive. Its checksums bind the included
files; `verification.json` records the clean-extraction numerical and PDF checks.
These retrospective analyses are separate from the original v2.0.4 benchmark.
