# Coupling-fields public benchmark data card

**Release candidate:** `coupling-fields-v1`
**Snapshot date:** 28 August 2026
**Distribution status:** public, outcome-disabled GSE299043 held-site plan;
prior terminal refusals retained

## Scope

This benchmark evaluates perturbation-specific dependence between linked
single-cell measurements. It contains one row per declared public panel,
machine-readable aggregate results, analysis protocols, source manifests,
deterministic runners, the reference implementation, and integrity tests.
Failed and refused panels are part of the benchmark.

The current snapshot covers seven scored public-data panels, six procedural
refusals, and one active outcome-disabled candidate:

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
| GSE279451 adult sepsis CITE-seq | RNA and surface protein | 19 development and 21 held donors | terminal development-evaluation refusal; not scored |
| GSE299043 MLN CITE-seq | RNA and surface protein | 10 Cambridge development and 10 LiveOnNY/Columbia held donors | outcome access disabled; not scored |

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

`GSE279451_CANDIDATE_DESIGNATION.json` defined the final candidate. It fixed 19
development donors, 21 disjoint held donors, nine
RNA--ADT markers, 81 ordered interactions, and 1,024 deterministically selected
cells per donor. The development gate and held gate compare the hierarchical
exact conditional estimator directly with the strongest matched signed Pearson
or Poisson-deviance residual transfer. One terminal attempt acquired and reduced
the 19 development matrices. The evaluation refused because the declared
`common_effect_graph` and `common_effect_ridge_only` control families were
unavailable, so it issued no head-to-head decision. No prediction,
authorization, held pairing, or held score was formed; all 21 held-donor matrix
members remained unopened. The refusal prohibits a rerun.

The active GSE299043 candidate fixes mesenteric lymph node, 10x 5-prime v2
TotalSeq-C, ten donors per site, nine exact RNA--ADT marker pairs, 81 ordered
interactions, and 512 outcome-independent MLN assignments per donor. HashSolo
defines assignments except for one exact source-backed 694B single-tissue
member. Marginal-only marker graphs accompany the exact conditional field in a
direct comparison with the
strongest signed Pearson or Poisson-deviance residual transfer. No H5AD member
has been opened in this release snapshot.

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
- Audit the GSE299043 one-shot development and held-site gates without changing
  the frozen code or target universe.

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

The GSE279451 protocol fixed the donor split, marker panel, cell budget, state
rules, graph construction, estimator grid, classical comparators, donor-equal
loss, inference, one-shot seals, and refusal rules before count access. Every
development configuration was to be selected by leave-one-donor-out prediction.
The declared gate could not be evaluated because both common-effect control
families were unavailable. Held truth was never opened.

The GSE299043 development screen selects every family by ten-fold
leave-one-donor-out prediction. Because selection and promotion use the same ten
development donors, its bootstrap interval is a promotion heuristic. The
untouched held-site gate is confirmatory and requires wins over both the
selected classical residual and destroyed-link coupling, at least 5% relative
deviance reduction, a paired-bootstrap upper endpoint below zero, at least
eight favorable donors, and an exact one-sided sign-flip `p <= 0.025` for each
comparison.

## Classical interaction baseline

The completed multi-state panels retain full signed Poisson-deviance and
Pearson residual matrices from the independence model. The GSE279451 and
GSE299043 head-to-heads use the one-degree-of-freedom signed square-root Pearson statistic
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
committed aggregate result, and candidate suite. Its main pass deselects four
assertions: two that certify earlier disabled phases, one BMMC assertion whose
deposited axis is not redistributed, and one Kotliarov assertion bound to a
historical estimator. The script reruns the Kotliarov assertion against those
exact historical bytes; all four tests remain checksum-bound.
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
without a held score. The GSE279451 protocol, source manifest, terminal
acquisition and evaluation attempts, reduced development table, refusal,
authorization template, acquisition/reducer/evaluator, one-shot scorer, and
poison tests are also checksum-bound. The refusal records zero held-matrix
access and forbids a rerun.

The GSE299043 protocol, designation, 207-member source template, metadata
preflight, family policy, authorization template, reducer, acquisition runner,
development evaluator, held scorer, and four adversarial test suites are also
checksum-bound. A separate publication template requires the later active
authorization itself to be fetched byte-for-byte from an immutable public
commit. The release has no outcome artifact for this candidate.

The packaged runners expose separated prediction and scoring commands. They
refuse to form held joint pairings until the exact prediction SHA-256 and byte
count are bound at an immutable public URL and commit through the candidate's
authorization and release files. This package contains the templates
`LAWLOR_SCORE_AUTHORIZATION_TEMPLATE.json` and
`HAO_SCORE_AUTHORIZATION_TEMPLATE.json` and
`KOTLIAROV_SCORE_AUTHORIZATION_TEMPLATE.json` for historical audit, plus
`GSE279451_SCORE_AUTHORIZATION_TEMPLATE.json` for the terminal GSE279451 audit,
and `data/confirmation/gse299043_mln/score_authorization_template_v1.json` for
the active candidate, together with its authorization-publication template.

## Known limits

Only PerturbSci supplies a held-guide pairing signal, and the structured
estimator does not beat its strongest endpoint baseline. ReSisTrace supplies
arm-level linkage evidence but not a replicated treatment contrast. The only
completed held-donor panel, Arce, fails the full gate. Target-bootstrap
intervals condition on the deposited biological units. State definitions and
finite-sample association estimates remain representation-dependent.
Lawlor, Hao, Kotliarov, PoKI-seq, BMMC, and GSE279451 have no held joint-table score and
add no positive or negative held predictive evidence. They remain visible as
procedural refusals. GSE279451 contributes no performance evidence: its
development evaluation refused before a prediction or classical-comparator
decision, and the held matrices remained unopened.
GSE299043 is an outcome-disabled plan and contributes no performance evidence
until a terminal held result is published.

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
| GSE279451 candidate-freeze tag | `gse279451-sepsis-v1-protocol` |
| GSE299043 candidate-freeze tag | `gse299043-mln-v1-protocol` |
| Archive DOI | not assigned |
| Repository code license | none granted |
| scGPT-derived embedding | omitted; checksum and derivation manifest supplied |
| Raw public matrices | omitted; upstream URLs and checksums supplied |
| Completed-result implementation provenance | historical run bytes archived and checksum-bound |

This package is public and source-visible. It must not be described as open
source, DOI-archived, or registry-hosted preregistration. A public
commit-addressed release does not imply an archive DOI, an open-source license,
or registry status.
