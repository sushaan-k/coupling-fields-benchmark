# Coupling-fields public benchmark data card

**Release candidate:** `coupling-fields-v1`
**Snapshot date:** 28 August 2026
**Distribution status:** public, commit-addressed prospective protocol freeze;
outcome acquisition authorized, scoring still prediction-gated

## Scope

This benchmark evaluates perturbation-specific dependence between linked
single-cell measurements. It contains one row per completed public panel,
machine-readable aggregate results, analysis protocols, source manifests,
deterministic runners, the reference implementation, and integrity tests.
Failed and refused panels are part of the benchmark.

The current snapshot covers seven public panels:

| Study | Linked measurements | Evaluation unit | Decision |
|---|---|---|---|
| PerturbSci-Kinetics, GSE218566 | pre-existing and nascent RNA | sequence-distinct guides | pairing signal only |
| Frangieh Perturb-CITE-seq | RNA and surface protein | sequence-distinct guides | refuse |
| Papalexi ECCITE-seq | RNA and surface protein | deposited treatment replicates | refuse |
| MultiPerturb-seq, GSE277747 | RNA and chromatin accessibility | sequence-distinct guides | refuse |
| PerturbFate, GSE291147 | labeled and unlabeled RNA | technical dates | refuse |
| ReSisTrace, GSE223003 | linked pre/post lineage states | two deposited cultures | arm-level linkage only |
| Arce T-cell Perturb-CITE-seq, GSE278572 | RNA and surface protein | held donor | refuse |

The exact values, uncertainty intervals, controls, decisions, and provenance
are in `results/final_public_benchmark_table.tsv`. The evidence boundary is in
`docs/FINAL_PUBLIC_EVIDENCE_LEDGER.md`.

Two additional public-data confirmations are specified but unscored. Candidate A
is the Lawlor HCA PBMC CITE-seq held-donor experiment bound in
`LAWLOR_CANDIDATE_DESIGNATION.json`. Candidate B is the separate GSE143417
PoKI-seq held-donor experiment, bound by its candidate designation and
preanalysis lock exposed as `POKI_CANDIDATE_DESIGNATION.json` and
`POKI_PREANALYSIS_LOCK.json`; the complete source chain is retained at the
repository root. Neither is counted among the seven completed panels. Their
exact analyses were first published with outcome access disabled in commit
`2e5f47a8676000c743be0459b9d979262e7eb147`; commit
`044478d35d46783eba9d91e2ab17925327af0f92` authorized acquisition without
changing the analysis bytes.

## Intended use

- Recompute the declared aggregate metrics and decision gates.
- Compare estimators under identical target universes, held units, and
  link-destruction controls.
- Audit positive, negative, and refused results without outcome-based panel
  deletion.
- Add a prospectively designated held-donor or held-study confirmation under
  the frozen protocol.

This package is not a benchmark for ordinary marginal-response prediction,
zero-shot perturbation ranking, causal direction, or population-level donor
inference.

## Inputs and outputs

Inputs are same-cell, same-nucleus, or explicitly lineage-linked measurements
with perturbation and control labels. Each panel defines a finite state in both
views using controls that are separate from evaluation observations. The
primary output is a perturbation-by-interaction-coordinate matrix after
fixed-margin permutation centering. Result files also report direct estimates,
structured estimates, endpoint and cross-covariance baselines, destroyed-link
controls, target-bootstrap intervals, support checks, and refusal decisions.

Raw source matrices are not bundled. They remain at their upstream public
repositories. Source manifests record accessions, URLs, sizes, and checksums;
prepared state caches are omitted where redistribution terms are not explicit.

## Evaluation contract

Targets or deposited biological units are the replication units stated for
each panel. Cells are never called biological replicates. The principal metrics
are pooled Pearson correlation, standardized RMSE, paired target-level
squared-error differences, and link-destruction contrasts. Ninety-five percent
intervals use 2,000 deterministic target bootstrap draws for predictive panels.
All declared panels remain in the table regardless of outcome.

The second-confirmation protocol fixes eligibility, encoders, splits,
comparators, endpoints, pass criteria, exclusions, link controls, and
multiplicity before outcome access. Candidate A is `SEALED`: its source
checksums, six-development/four-confirmation donor split, six biological
contrasts, runner, reducer, and implementation hashes are fixed and linked to
the public freeze. Candidate B is independently `SEALED` under its dated
protocol and runner; it is not a replacement for Candidate A. No result
produced from an unsealed candidate is confirmatory.

## Classical interaction baseline

The prospective protocols retain the full 3-by-3 signed Poisson-deviance and
Pearson residual matrices from the independence model under the same state
tables and held-unit split. A four-coordinate projection is not used for the
primary classical comparison because it discards residual entries. Every
method predicts the same held joint table from a common observed baseline
anchor and held margins, and all tables are written before held pairing is
opened. The coupling field is a log-linear interaction coordinate; it is not
claimed to be distinct from a saturated log-linear interaction
parameterization. The residual comparison tests whether conditional centering
and structured field estimation add predictive value over the classical
independence-residual construction.

## Integrity and reproducibility

`benchmark_manifest.json` records the packaged path, source path, SHA-256, byte
count, and role of every artifact. `SHA256SUMS` binds the manifest, this data
card, the candidate designation, the score-authorization template, and all
packaged files. The source checkout rebuilds the package with:

```bash
python3 scripts/package_public_coupling_benchmark.py
```

Validate the package with:

```bash
python3 scripts/package_public_coupling_benchmark.py --check
```

The public entry point `./reproduce.sh` checks every distributed hash,
committed aggregate results, and focused implementation tests. Full
public-data reruns require locally
acquired checksum-matched source objects or prepared caches.

Candidate A preflight is read-only:

```bash
python3 -m experiments.confirm_lawlor_hca_pbmc preflight
```

The `predict` command is authorized by the public candidate seal. The `score`
command remains blocked by the additional public-prediction hash gate described
below.

The packaged Candidate A runner exposes separate `predict` and `score`
commands. Held stimulus RNA and ADT margins may be used to write every
predicted table to one JSON file, but `score` refuses to form their cell pairing
until the file's exact SHA-256 and byte count appear at an immutable public URL
and commit in `LAWLOR_SCORE_AUTHORIZATION.json`.

## Known limits

Only PerturbSci supplies a held-guide pairing signal, and the structured
estimator does not beat its strongest endpoint baseline. ReSisTrace supplies
arm-level linkage evidence but not a replicated treatment contrast. The only
completed held-donor panel, Arce, fails the full gate. Target-bootstrap
intervals condition on the deposited biological units. State definitions and
finite-sample association estimates remain representation-dependent.
The Lawlor PBMC and PoKI-seq candidates have no outcomes in this snapshot and
therefore add no positive evidence yet.

The completed three-panel atlas was rerun with the current full-matrix
classical residual implementation, SHA-256 `35516883a567...`. Coupling fields
did not beat the classical representation uniformly; the completed result and
benchmark table preserve those comparisons.

## Publication and licensing status

The following fields state the distribution boundary of this snapshot:

| Field | Status |
|---|---|
| Public repository URL | `https://github.com/sushaan-k/coupling-fields-benchmark` |
| Immutable protocol release tag | `protocol-v1.0.1` (metadata correction; `protocol-v1` preserves the same frozen analysis bytes) |
| Archive DOI | not assigned |
| Repository code license | none granted |
| scGPT-derived embedding | omitted; checksum and derivation manifest supplied |
| Raw public matrices | omitted; upstream URLs and checksums supplied |
| Completed-result implementation provenance | current and checksum-bound |

A public commit-addressed benchmark does not require an archive DOI or an
inferred license. It must not be described as open source or DOI-archived. The
analysis is a prospective public protocol, not a registry-hosted
preregistration.
