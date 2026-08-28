# Coupling-fields benchmark

This repository is a source-visible benchmark for perturbation-specific
dependence in linked single-cell assays. It preserves positive, negative, and
refused panels under one fixed evaluation contract. The current protocol
freeze contains seven completed public panels and two untouched held-unit
confirmations: PoKI-seq GSE143417 and the Lawlor Human Cell Atlas PBMC CITE-seq
study.

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
./reproduce.sh
```

The entry point verifies every distributed byte against `SHA256SUMS`, then
runs the public API, estimator, classical-comparator, prospective-seal, and
artifact-consistency tests. Raw public matrices are not redistributed.

## Evidence contract

`results/final_public_benchmark_table.tsv` is the completed-panel ledger.
`docs/FINAL_PUBLIC_EVIDENCE_LEDGER.md` states the corresponding claim boundary.
The two prospective protocols, candidate designations, source manifests,
runners, and disabled locks were committed before expression outcomes were
opened. Commit `044478d35d46783eba9d91e2ab17925327af0f92` then authorized
outcome acquisition without changing the frozen analyses.

Production scoring uses two public freezes. First, protocol and implementation
bytes are committed before outcome acquisition. Second, every held-table
prediction is committed by exact SHA-256 before the runner constructs the held
joint table. This is a public, code-path prospective seal, not a blinded data
enclave: the public source files remain accessible to anyone outside the
declared runner.

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

No code license is granted in this snapshot. Upstream datasets and model
weights remain governed by their own terms.
