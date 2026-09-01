# Coupling-fields public benchmark data card

**Release:** `coupling-fields-v2.0.3-public-benchmark`<br>
**Snapshot date:** 31 August 2026<br>
**Repository:** <https://github.com/sushaan-k/coupling-fields-benchmark><br>
**Status:** released public benchmark

## Purpose

This benchmark evaluates whether association measured in a linked single-cell
source cohort can predict a recipient joint table when only the recipient row
and column margins are available. The main application is RNA-protein CITE-seq,
but the table representation also covers other paired finite-state assays.

The benchmark is designed for methods that predict dependence at recipient
abundance. It is not a benchmark for marginal protein abundance, causal effect,
cell-type classification, or zero-shot perturbation ranking.

## Composition

The release contains 37 analysis records, 29 matched comparisons, and 91
chronology records. Twelve analyses have scored outcomes. Twenty-four are
procedural refusals without a held performance value. One record is
infrastructure-unevaluable and has no scientific decision.

Analysis records are not synonymous with independent studies. Some studies
contribute both a frozen campaign result and a separately labeled retrospective
development analysis. `results/benchmark_panels_v2.tsv` gives the evidence role
for every record.

| Analysis | Evidence role | Evaluation unit | Status |
|---|---|---|---|
| Stephenson Cambridge to Newcastle | confirmatory held site | 56 physical donors | pass |
| GSE239452 nonpregnant to pregnant | post-access corrected cohort | 9 physical donors | pass |
| GSE314416 DB1 to DB1/DB2 | development pilot | 20 physical donors | stopped at pilot |
| COMBAT Oxford | retrospective adaptive development | 24 physical samples | development only |
| BMMC bridge analysis | retrospective adaptive development | 4 batches from one donor | development only |
| Seven linked perturbation panels | retrospective public benchmark | study-specific deposited units | mixed or refused |

The remaining records document source, support, numerical, or infrastructure
stops before held scoring. They carry no held performance estimate.

## Inputs

Each scored analysis uses linked observations from a public study and defines:

- a finite state for each assay
- a biological or deposited evaluation unit
- source joint tables
- recipient marginal tables
- a fixed entity-pair universe
- matched comparators and controls

Raw matrices are not included. Study manifests record the public accession,
source URL, file size, and checksum when one is available. Small derived arrays
are included only when they are required to reproduce a reported result or a
recorded refusal.

## Outputs

The benchmark reports predicted recipient joint tables or their aggregate
scores, depending on the study. The binary conditional-transfer analyses use
multinomial deviance per cell. Earlier multistate panels retain their original
correlation, error, and link-destruction metrics. Metric direction and scale are
explicit in the version-2 ledgers.

The release separates three kinds of machine-readable record:

- `results/benchmark_panels_v2.tsv` records the analysis-level endpoint.
- `results/benchmark_comparisons_v2.tsv` records each matched comparator,
  uncertainty interval, and decision.
- `results/benchmark_sequence_v2.tsv` records the order of plans, data access,
  predictions, scores, and refusals.

## Evaluation units and uncertainty

Cells are never treated as biological replicates. Confirmatory CITE-seq results
weight physical donors or samples equally. The Stephenson and GSE239452
intervals use 20,000 paired donor-bootstrap draws. Retrospective perturbation
panels use the deposited biological or technical units declared in their
records. Development intervals are descriptive unless a panel is explicitly
labeled confirmatory.

## Controls

Binary conditional-transfer analyses compare the estimator with matched
signed Pearson or signed likelihood-ratio transfer at the same recipient
margins. Fitted common-effect and Poisson interaction baselines are reported
where available. The destroyed-link control permutes complete protein profiles
within each donor, preserving assay margins and within-protein dependence while
removing cell-level RNA-protein pairing.

## Provenance and integrity

`benchmark_manifest.json` records release metadata and every artifact cited by
the aggregate ledgers. `SHA256SUMS` covers all tracked release files except the
checksum file itself. The deterministic builder recreates the three version-2
ledgers and manifest; the verifier checks their schema, counts, artifact hashes,
numeric fields, and stage order.

The study protocols remain in `docs/`. They are retained as dated records of
the declared analysis, including unsuccessful campaigns. The concise evidence
boundary is `docs/FINAL_PUBLIC_EVIDENCE_LEDGER.md`.

## Known limits

The positive held evidence comes from observational RNA-protein CITE-seq. The
main analyses use binary states, a fixed cell budget, and small marker panels.
The benchmark does not establish perturbation causality, graph-regularization
benefit, or performance on unlinked spatial summaries. The GSE239452 analysis
is a post-access numerical correction, not a second prospective confirmation.
Retrospective development and historical perturbation panels should not be
combined with the Stephenson held-site result as equivalent evidence.

## Distribution and licensing

The code is available under the MIT License. Public source datasets retain
their original licenses and access conditions. The benchmark does not
redistribute raw matrices or participant identifiers. No archive DOI has been
assigned, and the public commit history is not a registry-hosted
preregistration.
