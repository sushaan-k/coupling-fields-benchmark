# Coupling-fields public benchmark data card

**Release candidate:** `coupling-fields-v1`
**Snapshot date:** 28 August 2026
**Distribution status:** public, tag-addressed candidate freeze with
fresh-clone verification complete and outcome access authorized

## Scope

This benchmark evaluates perturbation-specific dependence between linked
single-cell measurements. It contains one row per completed public panel,
machine-readable aggregate results, analysis protocols, source manifests,
deterministic runners, the reference implementation, and integrity tests.
Failed and refused panels are part of the benchmark.

The current snapshot covers seven scored public-data panels and one separate
preflight refusal:

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

The exact values, uncertainty intervals, controls, decisions, and provenance
are in `results/final_public_benchmark_table.tsv`. The evidence boundary is in
`docs/FINAL_PUBLIC_EVIDENCE_LEDGER.md`.

The fixed, scoreable confirmatory family contains exactly two unscored
candidates. Candidate A is Lawlor HCA PBMC CITE-seq version 2, bound in
`LAWLOR_CANDIDATE_DESIGNATION.json`. Candidate B is Hao GSE164378 version 1,
bound in `HAO_CANDIDATE_DESIGNATION.json`. Both must be executed and reported;
a pass cannot stop the family. The packaged files are frozen publicly, but both
designations now state `SEALED`, authorize outcome access, and bind public
freeze commit `51752b40610579375624115ed189e3789d8e8916`. No Lawlor or Hao
outcome is included or claimed in this authorization snapshot.

PoKI-seq is outside this two-candidate scoreable family. Its earlier frozen run
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
checksums. The small checksum-bound PoKI preflight cache is retained solely to
audit the recorded refusal; no Lawlor or Hao outcome cache is included.

## Evaluation contract

Targets or deposited biological units are the replication units stated for
each panel. Cells are never called biological replicates. The principal metrics
are pooled Pearson correlation, standardized RMSE, paired target-level
squared-error differences, and link-destruction contrasts. Ninety-five percent
intervals use 2,000 deterministic target bootstrap draws for predictive panels.
All declared panels remain in the table regardless of outcome.

The two candidate protocols fix eligibility, encoders, splits, comparators,
endpoints, pass criteria, exclusions, link controls, and multiplicity before
outcome access. Lawlor v2 fixes a six-development/four-held donor split and six
biological contrasts. Hao v1 fixes four development donors, three held donors,
and day-3/day-7 absolute tables. The family applies Bonferroni over these two
fixed candidates. Both designations are `SEALED` after independent fresh-clone
verification of the immutable public candidate freeze. Prediction publication
and its separate scoring authorization remain mandatory for each candidate.

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
card, both candidate designations, both score-authorization templates, and all
packaged files. The packager also verifies the internal hashes named by each
designation and the no-score fields in the PoKI refusal. The source checkout
rebuilds the package with:

```bash
python3 scripts/package_public_coupling_benchmark.py
```

Validate the package with:

```bash
python3 scripts/package_public_coupling_benchmark.py --check
```

The release entry point `./reproduce.sh` checks every distributed hash,
committed aggregate result, and both candidate suites. It deselects two exact
assertions that certify the earlier disabled phase; their preserved bytes and
successful execution are recorded in the public-freeze verification record.
Authorized prediction and full public-data reruns require locally acquired
checksum-matched source objects, the checksum-matched embedding, or prepared
caches as specified by each command.

Candidate A preflight is read-only:

```bash
python3 -m experiments.confirm_lawlor_hca_pbmc preflight
```

Candidate B preflight is also read-only when run without `--require-sealed`:

```bash
python3 -m experiments.confirm_hao_gse164378 preflight
```

Both candidates' acquisition and prediction commands are now authorized by the
verified public seal. Scoring still requires each exact public prediction and
its authorization/release chain.

The packaged runners expose separated prediction and scoring commands. They
refuse to form held joint pairings until the exact prediction SHA-256 and byte
count are bound at an immutable public URL and commit through the candidate's
authorization and release files. This package contains only the templates
`LAWLOR_SCORE_AUTHORIZATION_TEMPLATE.json` and
`HAO_SCORE_AUTHORIZATION_TEMPLATE.json`, not completed authorizations.

## Known limits

Only PerturbSci supplies a held-guide pairing signal, and the structured
estimator does not beat its strongest endpoint baseline. ReSisTrace supplies
arm-level linkage evidence but not a replicated treatment contrast. The only
completed held-donor panel, Arce, fails the full gate. Target-bootstrap
intervals condition on the deposited biological units. State definitions and
finite-sample association estimates remain representation-dependent.
The Lawlor PBMC and Hao candidates have no outcomes in this snapshot and add no
positive or negative evidence. PoKI-seq has a preflight refusal but no score,
so it likewise adds no evidence about predictive performance.

The completed three-panel atlas was rerun with the current full-matrix
classical residual implementation, SHA-256 `35516883a567...`. Coupling fields
did not beat the classical representation uniformly; the completed result and
benchmark table preserve those comparisons.

## Publication and licensing status

The following fields state the distribution boundary of this snapshot:

| Field | Status |
|---|---|
| Public repository URL | `https://github.com/sushaan-k/coupling-fields-benchmark` |
| Immutable candidate-freeze tag | `confirmatory-family-v2` |
| Archive DOI | not assigned |
| Repository code license | none granted |
| scGPT-derived embedding | omitted; checksum and derivation manifest supplied |
| Raw public matrices | omitted; upstream URLs and checksums supplied |
| Completed-result implementation provenance | current and checksum-bound |

This package is public and source-visible. It must not be described as open
source, DOI-archived, or registry-hosted preregistration. A public
commit-addressed release does not imply an archive DOI, an open-source license,
or registry status.
