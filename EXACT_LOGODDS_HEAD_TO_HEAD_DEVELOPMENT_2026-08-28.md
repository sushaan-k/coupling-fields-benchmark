# Exact log-odds versus classical residual transfer

## Status

This analysis is retrospective, adaptive development evidence. Candidate selection and performance summaries use the same nonheld units. The bootstrap intervals are paired descriptive intervals, not confirmatory inference. No COMBAT held sample or BMMC held-donor feature row was decoded, and no held prediction was formed.

## Comparison

The primary estimator fits donor-specific log odds by the exact fixed-margin conditional likelihood and shrinks them toward a population field. Its grid contains neighborhood size, donor-heterogeneity, ridge, graph, and transport penalties; graph penalty zero is a primary candidate. The comparator grid contains signed Pearson and signed-root Poisson-deviance coordinates, raw or exactly null-centered. Each source coordinate is divided by the square root of its table total, pooled with donor-equal weight, multiplied by the target square-root total, and inverted at the target margins. Coordinate excess is clamped to the feasible table boundary and is disclosed in the result.

The COMBAT comparison fits 12 Oxford calibration samples and evaluates 24 Oxford pilot samples from the authorized reduced artifact. The BMMC comparison fits donors 11466 and 19593 and evaluates the four batches of bridge donor 15078. The BMMC reader decodes only these 4,607 nonheld rows from the bound H5AD; all 4,419 held-donor rows remain forbidden.

## Results

| Development panel | Exact field deviance | Best residual deviance | Relative reduction | Paired 95% interval, field minus residual | Favorable units |
|---|---:|---:|---:|---:|---:|
| COMBAT calibration to pilot | 0.011558 | 0.015605 | 25.9% | -0.005967 to -0.002248 | 17/24 |
| BMMC fit donors to bridge batches | 0.010851 | 0.013148 | 17.5% | -0.005036 to -0.000385 | 4/4 |

The selected field used graph penalty zero in both panels. The best positive-graph candidate was 1.9% worse than graph zero in COMBAT and 5.6% worse in BMMC. Against the pairing-destroyed field, relative deviance reductions were 75.1% and 18.7%; both paired descriptive intervals excluded zero.

## Artifacts

- Evaluator: `experiments/evaluate_exact_logodds_head_to_head.py`
- Tests: `tests/test_exact_logodds_head_to_head.py`
- Result: `results/development/exact_logodds_head_to_head_v1.json`

The result binds the evaluator, source artifacts, H5AD identity, reference exact-likelihood module, and classical restoration numerics by SHA-256. Its access audit distinguishes the opaque full-file identity hash from numerical row decoding.
