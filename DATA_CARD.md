# Coupling-fields public benchmark data card

**Release candidate:** `coupling-fields-v1`
**Snapshot date:** 28 August 2026
**Distribution status:** release candidate for an outcome-disabled GSE279451
held-donor analysis plan; prior terminal refusals retained

## Scope

This benchmark evaluates perturbation-specific dependence between linked
single-cell measurements. It contains one row per declared public panel,
machine-readable aggregate results, analysis protocols, source manifests,
deterministic runners, the reference implementation, and integrity tests.
Failed and refused panels are part of the benchmark.

The current snapshot covers seven scored public-data panels, five procedural
refusals, and one outcome-disabled prospective candidate:

| Study | Linked measurements | Evaluation unit | Decision |
|---|---|---|---|
| PerturbSci-Kinetics, GSE218566 | pre-existing and nascent RNA | sequence-distinct guides | pairing signal only |
| Frangieh Perturb-CITE-seq | RNA and surface protein | sequence-distinct guides | refuse |
| Papalexi ECCITE-seq | RNA and surface protein | deposited treatment replicates | refuse |
| MultiPerturb-seq, GSE277747 | RNA and chromatin accessibility | sequence-distinct guides | refuse |
| PerturbFate, GSE291147 | labeled and unlabeled RNA | technical dates | refuse |
| ReSisTrace, GSE223003 | linked pre/post lineage states | two deposited cultures | arm-level linkage only |
| Arce T-cell Perturb-CITE-seq, GSE278572 | RNA and surface protein | held donor | refuse |
| PoKI-seq, GSE143417 | RNA and chromatin accessibility | held donor | preflight refusal; not scored |
| Lawlor HCA PBMC | RNA and surface protein | held donors | reducer refusal; not scored |
| Hao, GSE164378 | RNA and surface protein | held donors | support refusal; not scored |
| Kotliarov PBMC | RNA and surface protein | held batch and disjoint donors | support refusal; not scored |
| NeurIPS 2021 BMMC CITE-seq | RNA and surface protein | held donors | terminal numerical development refusal; not scored |
| GSE279451 adult sepsis CITE-seq | RNA and surface protein | 19 development and 21 held donors | outcome access disabled |

The exact values, uncertainty intervals, controls, decisions, and provenance
are in `results/final_public_benchmark_table.tsv`. The evidence boundary is in
`docs/FINAL_PUBLIC_EVIDENCE_LEDGER.md`.

The Lawlor and Hao family is closed. Both candidates were executed under the
public freeze but stopped before held joint-table scoring: Lawlor at the
frozen reducer's unsupported deposited-object check and Hao at the frozen
development marginal-support gate. Their refusal artifacts are included and
are not predictive failures.

The preceding scoreable family contained one candidate, Kotliarov PBMC
CITE-seq, bound in `KOTLIAROV_CANDIDATE_DESIGNATION.json`. It uses ten
development donors from batch 1 and nine disjoint held donors from batch 2,
excluding donor 209 from both because that donor occurs in both batches. Tag
`confirmatory-family-v3` was independently verified from a fresh clone at
commit `a034fd272ef631d70f39debc467570568ef8754a` before outcome access was
authorized. Its one authorized preparation stopped when fewer than four of the
five prespecified RNA-only lineages met the frozen requirement of at least 50
retained cells in every one of 19 donors. The ADT file was byte-hashed for
integrity, but its HDF5 count dataset was never opened. No held RNA-ADT
pairing, joint table, prediction, score, or performance estimate was formed.
The candidate is terminal and was not rerun.

The subsequent BMMC candidate ended after three recorded development attempts.
The frozen exact conditional optimizer failed its final common-effect refit
after the permitted numerical-equivalence repair. No prediction or held score
was formed, all six held-donor count slices remained unopened, and BMMC cannot
be revived.

`GSE279451_CANDIDATE_DESIGNATION.json` defines the current prospective
candidate. It fixes 19 development donors, 21 disjoint held donors, nine
RNA--ADT markers, 81 ordered interactions, and 1,024 deterministically selected
cells per donor. The development gate and held gate compare the hierarchical
exact conditional estimator directly with the strongest matched signed Pearson
or Poisson-deviance residual transfer. At this snapshot, the source manifest,
development attempt, reduced tables, prediction, authorization, and score do
not exist; no count matrix has been acquired.

PoKI-seq is outside the current scoreable family. Its earlier frozen run
failed the state-occupancy support gate before prediction and scoring. The
refusal, retained designation, lock, cache, runner, and protocol are packaged
at `results/gse143417_pokiseq_preflight_refusal.json` and their corresponding
source paths. The refusal has `outcome_scored: false` and is neither a
confirmatory test nor evidence for or against model performance.

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
repositories. Source manifests record accessions, URLs, sizes, and available
checksums. The small checksum-bound PoKI preflight cache and Hao reducer
provenance are retained solely to audit the recorded refusals. Raw candidate
matrices are not included.

## Evaluation contract

Targets or deposited biological units are the replication units stated for
each panel. Cells are never called biological replicates. The principal metrics
are pooled Pearson correlation, standardized RMSE, paired target-level
squared-error differences, and link-destruction contrasts. Ninety-five percent
intervals use 2,000 deterministic target bootstrap draws for predictive panels.
All declared panels remain in the table regardless of outcome.

The GSE279451 protocol fixes the donor split, marker panel, cell budget, state
rules, graph construction, estimator grid, classical comparators, donor-equal
loss, inference, one-shot seals, and refusal rules before count access. Every
development configuration is selected by leave-one-donor-out prediction. Held
truth cannot be paired until a prediction artifact is committed publicly and
its exact bytes are authorized from an immutable commit.

## Classical interaction baseline

The completed multi-state panels retain full signed Poisson-deviance and
Pearson residual matrices from the independence model. The GSE279451
head-to-head uses the one-degree-of-freedom signed square-root Pearson statistic
and signed square-root Poisson deviance for each 2-by-2 table, with raw and
exact-null-centered variants selected on the same development folds. Every
method predicts the same held table from the same margins before held pairing
is opened. The coupling field is a log-linear interaction coordinate; the
comparison asks whether exact conditional estimation and cross-entity
regularization improve prediction over the classical residual construction.

## Integrity and reproducibility

`benchmark_manifest.json` records the packaged path, source path, SHA-256, byte
count, and role of every artifact. `SHA256SUMS` binds the manifest, this data
card, all candidate designations, score-authorization templates, and all
packaged files. The packager also verifies the internal hashes named by each
designation and the no-score fields in the PoKI refusal. The distribution is
validated directly by its checksum manifest and test entry point.

The release entry point `./reproduce.sh` checks every distributed hash,
committed aggregate result, and candidate suites. It deselects two exact
assertions that certify the earlier disabled phase; their preserved bytes and
successful execution are recorded in the public-freeze verification record.
Authorized prediction and full public-data reruns require locally acquired
checksum-matched source objects, the checksum-matched embedding, or prepared
caches as specified by each command.

The historical Lawlor and Hao runners remain checksum-bound for audit. Their
preflights require the omitted, checksum-matched scGPT embedding and are not
part of the zero-download verification path.

The Kotliarov runner and tests remain checksum-bound for audit. Lawlor, Hao,
and Kotliarov are terminally closed; the frozen Kotliarov candidate permits no
rerun. Its canonical refusal is
`results/kotliarov_pbmc_public_refusal.json`.

The three BMMC development records are checksum-bound and close that candidate
without a held score. The GSE279451 protocol, disabled source and authorization
templates, acquisition/reducer/evaluator, one-shot scorer, and poison tests are
also checksum-bound. They permit development access only after a public freeze
and held access only after a passing development result and a second immutable
prediction authorization.

The packaged runners expose separated prediction and scoring commands. They
refuse to form held joint pairings until the exact prediction SHA-256 and byte
count are bound at an immutable public URL and commit through the candidate's
authorization and release files. This package contains the templates
`LAWLOR_SCORE_AUTHORIZATION_TEMPLATE.json` and
`HAO_SCORE_AUTHORIZATION_TEMPLATE.json` and
`KOTLIAROV_SCORE_AUTHORIZATION_TEMPLATE.json` for historical audit, plus
`GSE279451_SCORE_AUTHORIZATION_TEMPLATE.json` for the current disabled plan.

## Known limits

Only PerturbSci supplies a held-guide pairing signal, and the structured
estimator does not beat its strongest endpoint baseline. ReSisTrace supplies
arm-level linkage evidence but not a replicated treatment contrast. The only
completed held-donor panel, Arce, fails the full gate. Target-bootstrap
intervals condition on the deposited biological units. State definitions and
finite-sample association estimates remain representation-dependent.
Lawlor, Hao, Kotliarov, PoKI-seq, and BMMC have no held joint-table score and
add no positive or negative held predictive evidence. They remain visible as
procedural refusals. GSE279451 is a prospective protocol only and contributes
no performance evidence until its locked sequence completes.

The completed three-panel atlas was rerun with the current full-matrix
classical residual implementation, SHA-256 `35516883a567...`. Coupling fields
did not beat the classical representation uniformly; the completed result and
benchmark table preserve those comparisons.

## Publication and licensing status

The following fields state the distribution boundary of this snapshot:

| Field | Status |
|---|---|
| Public repository URL | `https://github.com/sushaan-k/coupling-fields-benchmark` |
| Prior immutable candidate-freeze tag | `confirmatory-family-v3` |
| GSE279451 candidate-freeze tag | pending for this release candidate |
| Archive DOI | not assigned |
| Repository code license | none granted |
| scGPT-derived embedding | omitted; checksum and derivation manifest supplied |
| Raw public matrices | omitted; upstream URLs and checksums supplied |
| Completed-result implementation provenance | historical run bytes archived and checksum-bound |

This package is public and source-visible. It must not be described as open
source, DOI-archived, or registry-hosted preregistration. A public
commit-addressed release does not imply an archive DOI, an open-source license,
or registry status.
