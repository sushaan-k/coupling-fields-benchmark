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
The full `./reproduce.sh` entry point also runs both candidate lock suites. In
the disabled phase, sealed operations refuse before loading the omitted scGPT
embedding. Its expected hash and derivation inputs are recorded in
`data/scgpt_gene_embeddings_manifest.json`; authorized prediction still
requires that checksum-matched local artifact. Raw public matrices are not
redistributed.

## Evidence contract

`results/final_public_benchmark_table.tsv` is the completed-panel ledger.
`docs/FINAL_PUBLIC_EVIDENCE_LEDGER.md` states the corresponding claim boundary.
The two scoreable candidates are bound in
`LAWLOR_CANDIDATE_DESIGNATION.json` and
`HAO_CANDIDATE_DESIGNATION.json`. Both have status
`OUTCOME_ACCESS_DISABLED`, `outcome_access_authorized: false`, and null public
freeze fields. Their protocols, source manifests, metadata audits, aliases,
runners, reducers, authorization templates, and tests are checksum-bound in
this freeze. Tag `confirmatory-family-v2` preserves the disabled-outcome
candidate bytes. It does not authorize acquisition of either candidate's RNA
or ADT outcome matrix.

PoKI-seq is not a third scoreable candidate. Its frozen execution stopped at
the state-occupancy support gate before a prediction or score was written.
`results/gse143417_pokiseq_preflight_refusal.json` records the exact failing
arm, input hashes, and `outcome_scored: false`; it supplies no evidence for or
against predictive performance.

The declared production workflow requires two further bindings. First, the
tagged candidate bytes must be independently verified from a fresh clone
before outcome access is authorized in a later commit. Second, each candidate's
prediction must be published and bound by exact
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
