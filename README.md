# Coupling-fields benchmark

This repository is a public, source-visible benchmark for
perturbation-specific dependence in linked single-cell assays. It preserves
positive, negative, and refused panels under one fixed evaluation contract.
It contains seven scored public-data panels and five transparent procedural
refusals. PoKI-seq stopped at its state-occupancy preflight, Lawlor stopped when
the frozen reducer rejected the deposited ADT object type, and Hao stopped when
fewer than 12 markers passed its frozen marginal-support rule. None of those
three runs formed a held joint-table score. The prospectively frozen Kotliarov
PBMC candidate also stopped during preparation, before estimator fitting or
scoring, because fewer than four RNA-only lineages met its frozen donor-support
rule.

The BMMC candidate ended in a terminal numerical development refusal after
three recorded attempts. It produced no prediction or held score and cannot be
revived. Its freeze-era runner is retained under `experiments/historical/`,
while the terminal-attempt runner, evaluator, and exact test occupy their
current public paths and match the hashes recorded by attempt 3. GSE279451 is
the next and only outcome-disabled candidate: its public
analysis plan, 19/21 physical-donor split, metadata/feature-axis preflight,
1,024-cell budget, estimator, controls, gates, acquisition/reducer/evaluator,
and one-shot scorer are checksum-bound here before any count acquisition.

The method represents each finite joint table by its double-centered
log-linear interaction, centers finite-sample estimates with fixed-margin
permutations, and optionally shares information across entities with low-rank
and hypergraph penalties. The field is a centered parameterization of the
saturated log-linear interaction; it is not claimed to be a different
classical estimand. Every prospective confirmation compares the complete
pipeline against full Pearson and signed Poisson-deviance residual matrices
from the independence model on the same predicted held tables.

The seven scored panels and earlier procedural refusals retain their exact
historical estimator at `mapreg/historical/coupling_fields_29a3875.py`. The
current `mapreg/coupling_fields.py` contains the deterministic einsum update
and is separately bound, with the hierarchical estimator and its transitive
dependencies, by the outcome-disabled GSE279451 plan. Historical result hashes
have not been relabeled as if they were produced by the current implementation.

## Verify the snapshot

Python 3.9 or newer is required.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
shasum -a 256 -c SHA256SUMS
```

This verifies every distributed byte without acquiring candidate outcomes.
The full `./reproduce.sh` entry point runs the estimator and candidate tests,
except for two preserved assertions that certify the earlier disabled phase
and one BMMC deposited-axis assertion whose source file is not redistributed.
Those exact test bytes remain checksum-bound; the disabled-phase assertions
passed in the fresh-clone verification record, and the BMMC axis assertion is
retained with the terminal evaluator test. The Kotliarov disabled-preflight
assertion is run separately in an isolated copy containing its checksum-bound
historical estimator, so the current GSE dependency is never substituted for
the code recorded by that earlier run. The scGPT embedding's expected hash and
derivation inputs are recorded in
`data/scgpt_gene_embeddings_manifest.json`. Raw public matrices are not
redistributed.

## Evidence contract

`results/final_public_benchmark_table.tsv` is the complete declared-panel ledger.
`docs/FINAL_PUBLIC_EVIDENCE_LEDGER.md` states the corresponding claim boundary.
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

## Next held-donor protocol

`GSE279451_CANDIDATE_DESIGNATION.json` prospectively fixes a donor-disjoint
RNA--ADT confirmation in the GSE279451 adult sepsis CITE-seq data. It reserves
19 development and 21 held physical donors, fixes nine markers and 81 ordered
pairs, and compares the hierarchical exact conditional estimator directly with
the strongest matched Pearson or Poisson-deviance residual transfer and strong
controls. The outcome-disabled plan was verified from a fresh clone at commit
`f63c9dc760a85a1361ce75e13036eb23262b1bc7` before count access. The active
source manifest, development acquisition/evaluation attempts or refusals,
reduced development artifact, prediction, authorization, score attempt, and
held result did not exist at that freeze. BMMC is terminally closed and cannot
be revived.

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
- `results/`: machine-readable completed results and benchmark table.
- `data/`: source manifests and metadata-only eligibility records.
- `tests/`: integrity, estimator, comparator, and pairing-seal tests.

The GSE279451 outcome-disabled analysis plan is published at
`https://github.com/sushaan-k/coupling-fields-benchmark` under tag
`gse279451-sepsis-v1-protocol`; the verification record is
`docs/GSE279451_PUBLIC_FREEZE_VERIFICATION_2026-08-28.json`. The prior Kotliarov
freeze remains under tag `confirmatory-family-v3`. The repository has no
archive DOI and grants no code license. It must not be called open source,
DOI-archived, or registry-hosted preregistration. Upstream datasets and model
weights remain governed by their own terms.
