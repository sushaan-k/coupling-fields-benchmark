# Coupling-fields benchmark

This repository is a public, source-visible benchmark for
perturbation-specific dependence in linked single-cell assays. It preserves
positive, negative, and refused analyses under explicit evidence roles. The
version-2 release candidate contains 37 panel records: 12 scored held,
replicate, pilot, or nonheld-development analyses; 24 procedural refusals; and
one published infrastructure-unevaluable attempt. These are analysis
records rather than unique studies because BMMC and COMBAT each retain both a
frozen campaign outcome and a separately labeled retrospective adaptive
development analysis.

The prospectively frozen Stephenson held-site confirmation is the first
positive confirmatory transfer in the benchmark. Across 56 physical samples,
hierarchical exact-conditional transfer reduced mean Poisson deviance per cell
by 17.46% relative to the pilot-selected signed-deviance residual, with a
paired 95% interval for the raw loss difference of -0.00413 to -0.00080; 50 of
56 samples were favorable (exact one-sided sign-test p=5.09e-10). It reduced
loss by 46.36% relative to destroyed links. A later, explicitly post-hoc audit
found 6.18% lower loss than the exact common-effect CMLE and 3.39% lower loss
than pooled-table log odds reconstructed with the conditional fixed-margin
expectation. The frozen artifact originally labeled the latter as pooled
Poisson; its bytes are preserved, but the release ledger uses the corrected
method name. These post-hoc comparisons are descriptive and do not alter the
frozen gate.

The GSE239452 held-cohort analysis is retained as a post-access correction. A
subsequent numerical audit found that endpoint underflow in signed-root
deviance inversion had misreconstructed 80 of 729 residual coordinates. The
corrected inversion preserves the primary loss of 0.0085063650 and gives
selected-residual loss 0.0141314858, a 39.8056% reduction with donor-bootstrap
difference interval -0.00709509 to -0.00423212; all nine donors favor the
primary. The 78.89%
destroyed-link reduction is unchanged, and the exact common-effect CMLE remains
2.20% better than the primary. The aggregate uses the correction artifact,
while the original sealed prediction and score remain byte-identical and
preserve the original chronology. Neither analysis is prospective confirmation.

A separate post-hoc GSE239452 audit implements the standard fixed-interaction
Poisson prediction: it fits pooled saturated table interactions on development
donors, selects transport multiplier 1 without held donors, and refits row and
column nuisance parameters at each recipient's margins. The structured primary
had mean loss 0.0085063650 versus 0.0099824140 for this comparator, a 14.7865%
reduction with donor-bootstrap relative interval 12.1841% to 17.3694% and raw
difference interval -0.00184323 to -0.00116575; all nine donors were favorable
(one-sided sign-test p=1/512). The audit replayed all 81 saturated source tables
to maximum normalized error 1.78e-16 and reproduced the nine official held
donor pairs sequentially before deleting each raw pair. This is a post-hoc
comparison within the existing corrected cohort, not a confirmation or an
independent cohort.

PoKI-seq stopped at state-occupancy preflight, Lawlor at deposited-object
compatibility, Hao and the first Kotliarov campaign at frozen support gates,
BMMC at numerical development, GSE279451 at comparator availability, GSE299043 at source feature
preflight, and COMBAT at pilot candidate availability. GSE314416 also stopped
at its pilot gate. Twelve later source campaigns terminated before held
scoring and remain visible in the aggregate ledger. The source-only Kotliarov
binary-v2 replacement then refused because no frozen configuration completed
every source-held fold; it produced no comparisons or held run. The subsequent
GSE179221 BMMC candidate refused at exact cognate-axis preflight on the first
source donor. Only its barcode and feature axes were decoded; no count dataset,
model, comparison, held file, prediction, or score was opened or formed. The
GSE214546 TEA-seq campaign completed a frozen 512-cell, 53-marker reduction for
its first source donor, then refused because the second source donor had fewer
than 512 matched singlets. The exact overlap count was not serialized. The
remaining six source H5s and all eight held H5s remained unopened, and no model,
comparison, prediction, or score was formed. The
GSE317605 longitudinal PBMC campaign reached its public calibration gate on
seven patients, 28 visits, and a 16-by-16 RNA--ADT marker field. Primary loss
was 0.0065297608 versus 0.0066086202 for tuned time-conditioned ridge-Poisson,
a 1.193% reduction with six of seven patients favorable. This missed the
frozen 5% threshold. The selected hypergraph penalty was zero, and the primary
equaled the retuned graph-zero fit. The campaign is therefore an unscored
calibration refusal; no pilot or held matrix was requested. The
recovery-amended unused-Cambridge attempt produced no prediction or score and is
infrastructure-unevaluable; its published terminal record assigns no scientific
decision and its performance fields remain empty.

The BMMC candidate ended in a terminal numerical development refusal after
three recorded attempts. It produced no prediction or held score and cannot be
revived. Its freeze-era runner is retained under `experiments/historical/`,
while the terminal-attempt runner, evaluator, and exact test occupy their
current public paths and match the hashes recorded by attempt 3. The subsequent
GSE279451 candidate was frozen publicly before count access. Its 19 development
donors were reduced under the fixed 1,024-cell budget, but the terminal
development evaluation could not instantiate either prespecified common-effect
control. The evaluator therefore refused without writing a development result.
No prediction or held score was formed, all 21 held-donor matrix members
remained unopened, and the candidate cannot be rerun.

The later GSE299043 protocol reserved ten Cambridge organ donors and ten
donor-disjoint LiveOnNY/Columbia donors for a mesenteric-lymph-node
RNA--protein confirmation. Its one development attempt completed 21 of 56
member reductions, then stopped at the next member's feature preflight because
the file lacked an accepted MLN HTO ID. The failing member's matrix values and
all 151 held members remained unopened. No model, prediction, authorization,
pairing, or score was formed, and the candidate cannot be rerun.

The method represents each finite joint table by its double-centered
log-linear interaction, centers finite-sample estimates with fixed-margin
permutations, and optionally shares information across entities with low-rank
and hypergraph penalties. The field is a centered parameterization of the
saturated log-linear interaction; it is not claimed to be a different
classical estimand. Every prospective confirmation compares the complete
pipeline against full Pearson and signed Poisson-deviance residual matrices
from the independence model on the same predicted held tables.

The seven historical scored panels and earlier procedural refusals retain their exact
historical estimator at `mapreg/historical/coupling_fields_29a3875.py`. The
current `mapreg/coupling_fields.py` contains the deterministic einsum update
and is separately bound, with the hierarchical estimator and its transitive
dependencies, by the terminal GSE279451 attempt. Historical result hashes
have not been relabeled as if they were produced by the current implementation.

## Verify the snapshot

Python 3.9 or newer is required.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
shasum -a 256 -c SHA256SUMS
python -m experiments.build_public_benchmark_release --check
python -m scripts.verify_public_benchmark_release
```

After the aggregate tag is published, a clean fresh clone can additionally
verify that the local and public tag resolve to the checked-out commit and that
`SHA256SUMS` covers every tracked release byte:

```bash
python -m scripts.verify_public_benchmark_release --require-clean --require-tag
```

This verifies every distributed byte without acquiring candidate outcomes.
The full `./reproduce.sh` entry point runs the estimator and candidate tests,
except for two preserved assertions that certify the earlier disabled phase
and one BMMC deposited-axis assertion whose source file is not redistributed.
Those exact test bytes remain checksum-bound; the disabled-phase assertions
passed in
`docs/CONFIRMATORY_FAMILY_PUBLIC_FREEZE_VERIFICATION_2026-08-28.json`, and the
BMMC axis assertion is retained with the terminal evaluator test. The Kotliarov
disabled-preflight
assertion is run separately in an isolated copy containing its checksum-bound
historical estimator, so the current GSE dependency is never substituted for
the code recorded by that earlier run. The scGPT embedding's expected hash and
derivation inputs are recorded in
`data/scgpt_gene_embeddings_manifest.json`. Raw public matrices are not
redistributed.

## Evidence contract

`results/benchmark_panels_v2.tsv` is the metric-aware panel ledger;
`results/benchmark_comparisons_v2.tsv` separates methods, metrics, evidence
roles, uncertainty, and decisions; and `results/benchmark_sequence_v2.tsv`
records plan, prediction, authorization, score, and refusal stages. The older
`results/final_public_benchmark_table.tsv` is retained as a byte-stable
historical input, not as the complete current benchmark. The deterministic
builder and verifier are
`experiments/build_public_benchmark_release.py` and
`scripts/verify_public_benchmark_release.py`.
`docs/FINAL_PUBLIC_EVIDENCE_LEDGER.md` states the corresponding claim boundary.
The GSE179221 candidate, implementation amendment, one-shot source attempt,
exclusive consumption record, and terminal result are published under the
`gse179221-bmmc-v1-*` tags. The first source file failed exact cognate-axis
uniqueness before its count matrix was opened; the other seven source files and
all ten held files remained unrequested.
The GSE214546 pre-access lineage and source attempt are published under the
`gse214546-teaseq-v1-*` tags. Its terminal source result records four completed,
byte- and hash-verified source-only downloads: one complete reduction and one
axis-only source H5 that failed the 512-cell support gate. It requested no held
H5 and cannot be rerun.
The GSE317605 protocol, candidate, implementation, and one-shot calibration
attempt were published before the corresponding access stages. Tag
`gse317605-longitudinal-v1-calibration-result` binds the terminal result to
commit `7f229baca0b261e1a0ee832defcc5cfa96aad023`; the result SHA-256 is
`9b3fcec43e38d876a312c8488292264ed747a9c70b4a75609f1d9ac18948040e`.
Its access journal reconciles 210 completed and deleted calibration files and
zero pilot or held requests.
The GSE239452 numerical correction is preserved in
`results/gse239452_citeseq_post_access_correction.json`. Its correction runner
and focused test are checksum-bound, and the original sealed prediction and
score remain unchanged in the benchmark sequence.
The true fixed-interaction Poisson audit is preserved in
`results/development/gse239452_standard_poisson_interaction_posthoc.json` under
tag `gse239452-standard-poisson-v1-result`; its runner and tests are
checksum-bound. The earlier
`results/development/classical_interaction_baselines_posthoc.json` remains
byte-identical and is labeled as pooled-table log odds with conditional
reconstruction wherever it appears in the release ledgers.
The closed Lawlor and Hao candidates are bound in
`LAWLOR_CANDIDATE_DESIGNATION.json` and
`HAO_CANDIDATE_DESIGNATION.json`. Both have status `SEALED`, explicit
outcome-access authorization, and public-freeze commit
`51752b40610579375624115ed189e3789d8e8916`. Their protocols, source manifests, metadata audits, aliases,
runners, reducers, authorization templates, and tests are checksum-bound in
this release. Tag `confirmatory-family-v2` preserves the disabled-outcome
candidate bytes. A fresh-clone verification of that tag passed before a later
commit authorized acquisition of both candidates' RNA and ADT outcome
matrices. Their terminal refusal JSONs are retained under `results/`.

`KOTLIAROV_CANDIDATE_DESIGNATION.json` defines a separate one-candidate family.
It excludes donor 209 from both batches, uses ten development donors and nine
disjoint held donors, and binds a donor-level inference plan plus full Pearson
and signed-deviance residual comparisons. Tag `confirmatory-family-v3` is the
prospective public analysis-plan freeze. Fresh-clone verification passed at
commit `a034fd272ef631d70f39debc467570568ef8754a`; the designation is `SEALED`
and authorized one outcome-preparation attempt. That attempt terminated at the
frozen RNA lineage-support gate. The ADT file was read only as an opaque byte
stream for integrity verification; its HDF5 count dataset was never opened.
No held RNA-ADT pairing, joint table, prediction, score, or performance
estimate was formed, and the candidate was not rerun.

## GSE279451 held-donor protocol

`GSE279451_CANDIDATE_DESIGNATION.json` fixed a donor-disjoint
RNA--ADT confirmation in the GSE279451 adult sepsis CITE-seq data. It reserves
19 development and 21 held physical donors, fixes nine markers and 81 ordered
pairs, and compares the hierarchical exact conditional estimator directly with
the strongest matched Pearson or Poisson-deviance residual transfer and strong
controls. The outcome-disabled plan was verified from a fresh clone at commit
`f63c9dc760a85a1361ce75e13036eb23262b1bc7` before count access. One authorized
development acquisition decoded 19 development matrices and no held matrix.
The terminal evaluation then refused because `common_effect_graph` and
`common_effect_ridge_only` were unavailable across the declared candidate
family. The frozen gate consequently produced no comparison decision against
the classical residual transfer. The refusal is recorded in
`results/development/gse279451_sepsis_evaluation_refusal.json`, SHA-256
`af6d1f26eb7ea3f566612e167843bd51c03cf961a595b434e55ae7ca4d20496b`.
No prediction, authorization, held pairing, or held score exists; no held matrix
member was opened, and `rerun_permitted` is false. BMMC is likewise terminally
closed and cannot be revived.

## GSE299043 held-site protocol

`data/confirmation/gse299043_mln/candidate_designation_v1.json` fixed one tissue
and assay chemistry at both sites, disjoint physical donors, and marginal-only
graphs. The immutable protocol release preceded any H5AD access. The terminal
attempt then refused at development-member feature preflight after 21 completed
reductions. The canonical generated refusal and its aggregate audit are under
`results/development/`; cell-level intermediate reductions were deleted and are
not distributed.

The protocol, source-member seal, reducer, development evaluator, held runner,
authorization template, and adversarial tests are frozen under tag
`gse299043-mln-v1-protocol`. The runner cannot request a held member until a
passing development artifact and every held prediction are committed and
both the prediction and active score authorization are bound through
byte-identical public GitHub blob URLs.
Those later stages were never reached. Tag
`gse299043-mln-v1-terminal-refusal` preserves the terminal record.

## COMBAT CITE-seq held-sample protocol

`data/confirmation/combat_citeseq/candidate_designation_v1.json` fixes a staged
RNA--ADT confirmation in the COMBAT CITE-seq cohort: 12 Oxford calibration
samples, 24 Oxford adaptive-pilot samples, 51 untouched Oxford confirmation
samples, and a separate panel of 10 untouched St George's samples. The primary
method is a Haldane/Paule--Mandel product-graph field. The frozen comparison is
against the pilot-selected matched signed Pearson or signed-root Poisson-deviance
transfer, with a pilot-selected ridge-only Haldane/PM field as the graph-specific
ablation.

The authorized pilot ended in a terminal candidate-availability refusal before
either held panel was opened. Two of eight primary configurations were
evaluable, but all four matched Pearson/deviance comparators and all three
ridge-only Haldane configurations violated the frozen attainable-margin rule
on at least one pilot sample. The pilot gate was therefore undefined and held
margin and outcome access are permanently closed. The structured record is
`results/development/combat_citeseq_pilot_terminal_refusal.json`; it
distinguishes the original authorized evaluator from the later packaging code.

PoKI-seq is not a third scoreable candidate. Its frozen execution stopped at
the state-occupancy support gate before a prediction or score was written.
`results/gse143417_pokiseq_preflight_refusal.json` records the exact failing
arm, input hashes, and `outcome_scored: false`; it supplies no evidence for or
against predictive performance.

The independent verification is recorded in
`docs/CONFIRMATORY_FAMILY_PUBLIC_FREEZE_VERIFICATION_2026-08-28.json`.
The later authorization commit was also verified from a fresh clone, as
recorded in
`docs/CONFIRMATORY_FAMILY_OUTCOME_AUTHORIZATION_VERIFICATION_2026-08-28.json`.
For Kotliarov, the disabled freeze passed fresh-clone verification before
outcome access was authorized. The authorization verification is recorded in
`docs/KOTLIAROV_OUTCOME_AUTHORIZATION_VERIFICATION_2026-08-28.json`, and the
terminal refusal is recorded in `results/kotliarov_pbmc_public_refusal.json`.
This is a source-visible public analysis plan and code-path seal, not a blinded
data enclave, registry-hosted preregistration, or open-source licensing claim.

The derived scGPT gene embedding used by the graph prior is not bundled
pending upstream model-weight redistribution review. Its expected output and
input hashes, official source URLs, and derivation script are supplied in
`data/scgpt_gene_embeddings_manifest.json` and
`scripts/prepare_scgpt_embeddings.py`.

## Layout

- `mapreg/`: coupling fields, structured estimation, classical residuals, and
  common-table reconstruction.
- `experiments/`: deterministic completed-panel and prospective runners.
- `docs/`: frozen protocols, theory boundary, and evidence ledger.
- `results/`: source results plus version-2 panel, comparison, and sequence ledgers.
- `data/`: source manifests and metadata-only eligibility records.
- `tests/`: integrity, estimator, comparator, and pairing-seal tests.

The aggregate release target is tag `coupling-fields-v2-public-benchmark`.
Until that tag and its GitHub release are created, the manifest status remains
`RELEASE_CANDIDATE_READY_FOR_TAG`; the repository
must not cite the target as an existing release. The GSE299043 plan is published at
`https://github.com/sushaan-k/coupling-fields-benchmark` under tag
`gse299043-mln-v1-protocol`. The immutable release and a fresh clone both
verified at commit `87c15787f734b20d06c7b8cb0c66680b2fe5c1b0`; the record is
`docs/GSE299043_PUBLIC_FREEZE_VERIFICATION_2026-08-28.json`. The terminal
attempt is published under tag `gse299043-mln-v1-terminal-refusal`. The terminal
GSE279451 plan remains under tag
`gse279451-sepsis-v1-protocol`; its verification record is
`docs/GSE279451_PUBLIC_FREEZE_VERIFICATION_2026-08-28.json`, and the later
terminal development refusal is retained under `results/development/`. The prior Kotliarov
freeze remains under tag `confirmatory-family-v3`. The repository has no
archive DOI and grants no code license. It must not be called open source,
DOI-archived, or registry-hosted preregistration. Upstream datasets and model
weights remain governed by their own terms.
