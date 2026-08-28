# Coupling-fields benchmark

This repository is a public, source-visible benchmark for
perturbation-specific dependence in linked single-cell assays. It preserves
positive, negative, and refused panels under one fixed evaluation contract.
It contains seven scored public-data panels and four transparent procedural
refusals. PoKI-seq stopped at its state-occupancy preflight, Lawlor stopped when
the frozen reducer rejected the deposited ADT object type, and Hao stopped when
fewer than 12 markers passed its frozen marginal-support rule. None of those
three runs formed a held joint-table score. The prospectively frozen Kotliarov
PBMC candidate also stopped during preparation, before estimator fitting or
scoring, because fewer than four RNA-only lineages met its frozen donor-support
rule.

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
The full `./reproduce.sh` entry point runs the estimator and candidate tests,
except for two preserved assertions that certify the earlier
disabled phase. Those exact test bytes remain checksum-bound and passed in the
fresh-clone verification record. The scGPT embedding's expected hash and
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

`BMMC_CANDIDATE_DESIGNATION.json` prospectively fixes a donor-disjoint RNA--ADT
confirmation in the NeurIPS 2021 BMMC CITE-seq data. It reserves six physical
donors, fixes ten markers and 100 ordered pairs, and requires a direct win over
the strongest matched Pearson or Poisson-deviance interaction-residual
transfer. The versioned complete H5AD had not been downloaded when this
protocol snapshot was committed; no feature value, prediction, or held score
exists. BMMC is disabled after any Sanger held-score attempt, so it cannot be
used as a replacement after observing a Sanger result.

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
This is a public code-path seal, not a blinded data enclave or a registry-hosted
preregistration.

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

The disabled-outcome analysis-plan freeze is published at
`https://github.com/sushaan-k/coupling-fields-benchmark` under tag
`confirmatory-family-v3`. It has no archive DOI and grants no code license. It
must not be called open source, DOI-archived, or registry-hosted
preregistration. Upstream datasets and model weights remain governed by their
own terms.
