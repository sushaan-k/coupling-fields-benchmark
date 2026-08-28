# Coupling-fields benchmark

This repository is a public, source-visible benchmark for
perturbation-specific dependence in linked single-cell assays. It preserves
positive, negative, and refused panels under one fixed evaluation contract.
It contains seven scored public-data panels, the unscored PoKI-seq GSE143417
preflight refusal, and a frozen two-candidate confirmatory family: Lawlor HCA
PBMC CITE-seq version 2 and Hao GSE164378 version 1.

The method represents each finite joint table by its double-centered
log-linear interaction, centers finite-sample estimates with fixed-margin
permutations, and optionally shares information across entities with low-rank
and hypergraph penalties. The field is a centered parameterization of the
saturated log-linear interaction; it is not claimed to be a different
classical estimand. Every prospective confirmation compares the complete
pipeline against full Pearson and signed Poisson-deviance residual matrices
from the independence model on the same predicted held tables.

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
The full `./reproduce.sh` entry point runs the estimator tests and both
candidate suites, except for two preserved assertions that certify the earlier
disabled phase. Those exact test bytes remain checksum-bound and passed in the
fresh-clone verification record. The scGPT embedding's expected hash and
derivation inputs are recorded in
`data/scgpt_gene_embeddings_manifest.json`; authorized prediction still
requires that checksum-matched local artifact. Raw public matrices are not
redistributed.

## Evidence contract

`results/final_public_benchmark_table.tsv` is the completed-panel ledger.
`docs/FINAL_PUBLIC_EVIDENCE_LEDGER.md` states the corresponding claim boundary.
The two scoreable candidates are bound in
`LAWLOR_CANDIDATE_DESIGNATION.json` and
`HAO_CANDIDATE_DESIGNATION.json`. Both now have status `SEALED`, explicit
outcome-access authorization, and public-freeze commit
`51752b40610579375624115ed189e3789d8e8916`. Their protocols, source manifests, metadata audits, aliases,
runners, reducers, authorization templates, and tests are checksum-bound in
this release. Tag `confirmatory-family-v2` preserves the disabled-outcome
candidate bytes. A fresh-clone verification of that tag passed before a later
commit authorized acquisition of both candidates' RNA and ADT outcome
matrices.

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
The remaining binding is candidate-specific: each prediction must be published
and bound by exact
SHA-256 and byte count before the runner can construct a held joint table.
This is a public code-path seal, not a blinded data enclave or a
registry-hosted preregistration.

The derived scGPT gene embedding used by the Lawlor graph prior is not bundled
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

The disabled-outcome freeze is published at
`https://github.com/sushaan-k/coupling-fields-benchmark` under tag
`confirmatory-family-v2`. It has no archive DOI and grants no code license. It
must not be called open source, DOI-archived, or registry-hosted
preregistration. Upstream datasets and model weights remain governed by their
own terms.
