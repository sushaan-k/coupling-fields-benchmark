# Coupling-fields public benchmark data card

**Release candidate:** `coupling-fields-v1`
**Snapshot date:** 28 August 2026
**Distribution status:** public, tag-addressed Kotliarov candidate freeze;
outcome access disabled pending fresh-clone verification

## Scope

This benchmark evaluates perturbation-specific dependence between linked
single-cell measurements. It contains one row per declared public panel,
machine-readable aggregate results, analysis protocols, source manifests,
deterministic runners, the reference implementation, and integrity tests.
Failed and refused panels are part of the benchmark.

The current snapshot covers seven scored public-data panels, three procedural
refusals, and one prospectively frozen candidate:

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
| Kotliarov PBMC | RNA and surface protein | held batch and disjoint donors | outcome access disabled |

The exact values, uncertainty intervals, controls, decisions, and provenance
are in `results/final_public_benchmark_table.tsv`. The evidence boundary is in
`docs/FINAL_PUBLIC_EVIDENCE_LEDGER.md`.

The Lawlor and Hao family is closed. Both candidates were executed under the
public freeze but stopped before held joint-table scoring: Lawlor at the
frozen reducer's unsupported deposited-object check and Hao at the frozen
development marginal-support gate. Their refusal artifacts are included and
are not predictive failures.

The current scoreable family contains one candidate, Kotliarov PBMC CITE-seq,
bound in `KOTLIAROV_CANDIDATE_DESIGNATION.json`. It uses ten development donors
from batch 1 and nine disjoint held donors from batch 2, excluding donor 209
from both because that donor occurs in both batches. The designation is
`OUTCOME_ACCESS_DISABLED` until tag `confirmatory-family-v3` is independently
verified from a fresh clone.

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

The Kotliarov protocol fixes eligibility, RNA-only lineages, exact aliases,
donor split, estimator grid, complete comparators, same-margin endpoint,
donor-cluster inference, pass criteria, and refusal rules before count access.
The nine held donors are the biological replicates. Prediction publication and
a later public scoring authorization are mandatory before the score-only held
pairing bundle can be opened.

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

The active Kotliarov preflight is:

```bash
python3 -m experiments.confirm_kotliarov_pbmc preflight
```

Lawlor and Hao are terminally closed. Kotliarov acquisition is not authorized
until the v3 freeze passes independent fresh-clone verification.

The packaged runners expose separated prediction and scoring commands. They
refuse to form held joint pairings until the exact prediction SHA-256 and byte
count are bound at an immutable public URL and commit through the candidate's
authorization and release files. This package contains the templates
`LAWLOR_SCORE_AUTHORIZATION_TEMPLATE.json` and
`HAO_SCORE_AUTHORIZATION_TEMPLATE.json` for historical audit and
`KOTLIAROV_SCORE_AUTHORIZATION_TEMPLATE.json` for the active candidate.

## Known limits

Only PerturbSci supplies a held-guide pairing signal, and the structured
estimator does not beat its strongest endpoint baseline. ReSisTrace supplies
arm-level linkage evidence but not a replicated treatment contrast. The only
completed held-donor panel, Arce, fails the full gate. Target-bootstrap
intervals condition on the deposited biological units. State definitions and
finite-sample association estimates remain representation-dependent.
Lawlor, Hao, and PoKI-seq have no held joint-table score and add no positive or
negative predictive evidence. They remain visible as procedural refusals.

The completed three-panel atlas was rerun with the current full-matrix
classical residual implementation, SHA-256 `35516883a567...`. Coupling fields
did not beat the classical representation uniformly; the completed result and
benchmark table preserve those comparisons.

## Publication and licensing status

The following fields state the distribution boundary of this snapshot:

| Field | Status |
|---|---|
| Public repository URL | `https://github.com/sushaan-k/coupling-fields-benchmark` |
| Immutable candidate-freeze tag | `confirmatory-family-v3` |
| Archive DOI | not assigned |
| Repository code license | none granted |
| scGPT-derived embedding | omitted; checksum and derivation manifest supplied |
| Raw public matrices | omitted; upstream URLs and checksums supplied |
| Completed-result implementation provenance | current and checksum-bound |

This package is public and source-visible. It must not be described as open
source, DOI-archived, or registry-hosted preregistration. A public
commit-addressed release does not imply an archive DOI, an open-source license,
or registry status.
