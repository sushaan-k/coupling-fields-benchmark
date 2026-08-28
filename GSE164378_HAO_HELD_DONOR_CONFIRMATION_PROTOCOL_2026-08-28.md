# GSE164378 held-donor RNA-protein confirmation protocol

**Protocol date:** 2026-08-28  
**Accession:** GSE164378  
**Study:** Hao et al., *Cell* 2021, DOI 10.1016/j.cell.2021.04.048  
**Status at designation:** outcome access disabled

## Confirmatory question

Can RNA-protein association structure learned from four vaccinees predict the
absolute post-vaccine joint RNA-protein tables of three untouched vaccinees?
The primary estimand is the complete day-3 and day-7 table, not a change score.
The development fit, predicted coordinates, held margins, public prediction
binding, and held-joint scoring are separated. The frozen prediction code path
uses held marginal counts but never a held RNA-ADT pairing, joint table, or
paired statistic.

This is an independent-donor CITE-seq confirmation. It is not a CRISPR screen,
does not test perturbation-target transfer, and does not use the authors'
integrated expression values.

## Confirmatory family and multiplicity

The scoreable confirmatory family contains exactly two untouched candidates:

1. the Lawlor HCA PBMC RNA-ADT study; and
2. the Hao GSE164378 vaccine RNA-ADT study specified here.

Both candidates must be executed and reported; a pass cannot stop the family.
Each directional gate is evaluated with an endpoint of a two-sided 95%
marker-bootstrap interval. That endpoint corresponds to a one-sided test at
alpha 0.025. Bonferroni correction over the two fixed candidates therefore
controls familywise alpha at 0.05. PoKI-seq is reported separately but is not in
this scoreable family because its frozen run produced no inferential test and
cannot be promoted.

## Source lock

Only the 10x 5-prime experiment is eligible. It contains 49,147 cells from all
eight donors and uses a 54-antibody panel. The 3-prime experiment is excluded.
The fixed source manifest is
`data/development/hao_gse164378/source_manifest_v1.json`.

The two GEO sample records expose the required 5-prime files directly. Before
outcome access, the source manifest fixes:

- the six direct NCBI URLs and exact filenames;
- each file's exact GEO byte count, totaling 226,047,108 bytes;
- the metadata URL, byte count, and SHA-256;
- independently verified SHA-256 values for both barcode axes and both feature
  lists; and
- the rule that `prepare` must hash every required file while reading it. The
  two matrix hashes remain unknown until authorized acquisition.

The computed hashes are written to `source_acquisition.json`, included in the
reducer manifest, and byte-bound in every later stage. A size mismatch, missing
file, wrong filename, symbolic link, declared-axis mismatch, out-of-range or
negative Matrix Market entry, malformed Matrix Market file, or unmatched cell
axis is a refusal. The 1,467,842,560-byte series archive is not required or
accepted by the reducer. The RNA and ADT barcode axes and feature lists were
inspected to verify same-cell alignment and freeze the exact aliases. Neither
RNA nor ADT matrix values were downloaded or opened in choosing this candidate,
split, lineages, aliases, estimator, or gate.

## Donors and biological blocks

The split is fixed and not generated from outcome data.

| Role | Donors |
|---|---|
| Development | P4, P7, P8, P1 |
| Held | P5, P3, P2 |
| Excluded | P6 |

P6 is excluded because the publication reports a highly activated immune state
before vaccination and removes this donor from its vaccination-response
analysis.

The GEO cell metadata codes the three time points as `0`, `2`, and `7`. The
official antibody/HTO workbook and paper identify the corresponding biological
samples as day 0, day 3, and day 7. The reducer performs this mapping exactly.
The five frozen broad lineages are B, CD4 T, CD8 T, Mono, and NK. DC is excluded
before outcome access because its smallest held block contains only 20 cells.
The frozen support floor is 40 cells per donor-time-lineage block. Every one of
the 7 retained donors by 3 time points by 5 lineages has at least 47 cells in
the deposited metadata; the minimum is P8 day-3 B. Every held block has at
least 71 cells; the minima are P2 day-3 NK and P3 day-3 B.

Each eligible analysis entity is one cognate marker in one lineage. Its primary
grid comprises day 3 and day 7, for eight coupling coordinates per entity.
Day-0 tables are excluded from the primary estimand and used only for the
secondary change analysis.

## Frozen RNA-ADT matches

The exact 33-entry table
`data/development/hao_gse164378/adt_gene_aliases_v1.tsv` was defined from the
official 5-prime feature panel without reading either outcome matrix. Matching
is exact and one-to-one. The reducer rejects duplicate ADT names, duplicate gene
symbols, a missing named feature, or more than one RNA row for a named gene.
There is no normalized-token match and no fallback from an unmatched ADT label
to a similarly named RNA feature. CD16 is excluded because its antibody target
does not identify a unique cognate RNA gene.

## State construction

For each candidate marker-lineage pair, RNA is log1p transformed after
library-size normalization to 10,000 counts. ADT is log1p transformed and
centered within each cell over the complete 54-feature panel. Two tertile
thresholds per assay are estimated from pooled development-donor day-0 cells in
that lineage and applied unchanged to every donor and day.

An entity is retained only if both pairs of thresholds are distinct and every
RNA and ADT state separately occupies at least 5% of cells in every development
donor-day marginal. Before prediction, the identical rule is applied to each
held donor-day using the RNA and ADT marginals separately. No held joint table,
paired statistic, or method score is formed for this support check. Unsupported
entities are reported and never replaced. Fewer than 12 unique retained
cognate markers is a deterministic support refusal.

## Primary representation

For a 3 by 3 table, the coupling field is the centered log-linear interaction
in four Helmert coordinates. Finite-table permutation centering uses 64 fixed
margin-preserving permutations and a pseudocount of 0.5. The table's row and
column margins, their finite-table null means, and their total count are
pairing-invariant. The prediction path computes only those quantities for held
donors; it does not compute or use a held joint table or paired statistic.

The primary response matrix has one row per retained marker-lineage entity and
eight columns: two post-vaccine days by four coupling coordinates. Development
donors are averaged equally. Clipped inverse permutation variance of that mean
weights the structured loss. The designated estimator applies a nuclear penalty
of 0.1 times the largest singular value and a graph penalty of 5.0. Its graph
contains each entity, its six nearest external entities in the sealed scGPT gene
embedding, and one hyperedge for every lineage represented after the support
filter. The membership-permuted control permutes embedding assignments once at
the cognate-marker level while preserving lineage hyperedges exactly.
Similarity ties are resolved by the frozen entity order. Optimization
uses tolerance 1e-9. Every structured fit reports convergence, iteration count,
relative step, objective, effective rank, singular values, and realized penalty
values in the frozen prediction artifact.

## Fixed comparisons

All methods are fit without using held pairing and are evaluated on the same
held tables with the same held margins.

### Matched field family

- development mean (`field_direct`);
- independence (`field_zero`);
- variance scalar shrinkage (`field_scalar`);
- nuclear penalty only (`field_nuclear`);
- hypergraph penalty only (`field_hypergraph`);
- the full designated estimator (`field_primary`);
- an identically fit hypergraph whose marker membership is permuted by the
  fixed seed (`field_membership_permuted`);
- ridge prediction from row and column margin coordinates
  (`field_endpoint_ridge`); and
- direct covariance coordinates (`covariance_direct`).

### Pairing-destroyed link

For every development table, the RNA-ADT pairing is destroyed with the same
fixed-margin permutation machinery used for centering. These destroyed
coordinates receive the identical inverse-variance weights, nuclear penalty,
hypergraph, graph penalty, and optimizer as `field_primary`. Thus, the destroyed
comparison changes the biological link and not the fitting machinery.

### Classical family

The full 3 by 3 Pearson independence-residual matrix and full signed Poisson
deviance-residual matrix are each transferred directly and with the same
structured estimator. They are converted back to tables under the held margins
before scoring. These comparisons test whether coupling coordinates add value
over standard independence-model residuals; a saturated log-linear interaction
is algebraically the same estimand as the coupling field.

## Absolute table reconstruction

The prediction JSON stores coordinates, held row and column margins, and
provenance. It stores no predicted table. After the exact prediction JSON is
published and bound, `score` reloads the locked coordinates and recomputes every
predicted table with the bound implementation before invoking the held-pairing
scoring path. Field coordinates receive the held table's finite-sample null mean.
Classical residual coordinates receive the corresponding null residual and
count scale. Iterative proportional fitting enforces the identical observed
held row and column margins for every method. Only after table reconstruction
and margin checks pass does `score` construct held RNA-ADT joint tables.

The common endpoint is multinomial deviance per held cell, averaged over the
three held donors, retained entities, and two post-vaccine days.

## Promotion gate

The result passes only if all conditions hold:

1. the lower endpoint of the 95% complete-marker bootstrap interval for pooled
   held coupling-field correlation is greater than zero;
2. coupling-field correlation is positive in each held donor;
3. the upper endpoint of the 95% marker-bootstrap interval for primary minus
   pairing-destroyed table deviance is below zero;
4. the upper endpoint for primary minus the best classical method is below
   zero, reselecting the best classical method inside every bootstrap draw;
5. the upper endpoint for primary minus the best matched non-primary method is
   below zero, reselecting the best matched method inside every draw; and
6. source integrity, development support, table reconstruction, margin matching,
   and optimization all complete without refusal.

The bootstrap resamples cognate markers with replacement for 2,000 draws using
seed 20260828 and retains every supported lineage entity belonging to each
sampled marker. Inference is conditional on the deposited held donors P5, P3,
and P2; it is not donor-population inference. The two-candidate confirmatory
family is closed. Any later candidate requires a newly declared alpha-spending
family.

## Secondary change analysis

Development day-3-minus-day-0 and day-7-minus-day-0 field changes are fit with
the direct and designated structured estimators. Their pooled held correlations,
per-donor correlations, and 95% marker-bootstrap interval are reported. Change
scores do not enter the primary table predictions, method selection, or gate.

## Execution barrier

The commands are sequential and non-overwriting.

1. `preflight` validates the disabled or sealed designation and every bound
   implementation hash.
2. After a public candidate freeze authorizes access, `prepare` validates the
   metadata and six direct 5-prime files, streams those files, computes all
   acquisition hashes, produces the fixed reduced directory, and byte-binds
   every reducer artifact. Its required outcome arguments are `--rna-barcodes`,
   `--rna-features`, `--rna-matrix`, `--adt-barcodes`, `--adt-features`, and
   `--adt-matrix`; there is no archive argument or alternate reducer path.
3. `predict` verifies that exact reducer bundle, fits the development models,
   uses only held margins, and writes one prediction JSON. It cannot bypass the
   reducer or accept an alternate reduced path.
4. The prediction JSON must be committed publicly. An independent fresh clone
   must first verify its exact path, bytes, and SHA-256 at that immutable
   commit. `public-bind` accepts only a 40-character lowercase commit and the exact matching GitHub
   `blob/<commit>/results/hao_gse164378_predictions.json` URL. It writes a new,
   non-overwriting score authorization, which must itself be published in a
   later commit before scoring.
5. An independent fresh clone must verify the exact authorization bytes and
   SHA-256 and confirm that its commit descends from the prediction commit.
   `authorize-score` then accepts only the exact authorization GitHub blob URL
   and a 40-character lowercase commit and writes the fixed non-overwriting
   score release. `score` refuses to use held pairing without this binding.
6. `score` verifies the release, authorization, prediction hashes and byte
   counts, both GitHub blob URLs, runner, protocol, reducer bundle, marker set,
   and margins. It recomputes tables and constructs held joint tables once, writing only
   `results/hao_gse164378_confirmation.json` and
   `results/hao_gse164378_confirmation_arrays.npz`.

Any source, integrity, support, optimization, reconstruction, scoring, or result
serialization failure inside an authorized stage produces its fixed
non-overwriting refusal JSON. A refusal, failed gate, or passed gate is final
for this frozen run; no threshold, alias, split, estimator, or baseline may be
changed and reported as the same confirmation.

The runner validates immutable GitHub URL syntax and local hashes. Remote bytes
and commit ancestry are procedural freeze requirements because the disabled
designation cannot authorize network publication of its own future artifacts.
Before outcome access, an independent fresh clone must likewise verify the
disabled designation and every bound protocol, source, implementation, test,
and embedding artifact at the candidate-freeze commit. The verification record
is released with the benchmark.

## Reporting commitment

The complete result, all method losses, donor correlations, bootstrap arrays,
marginal-support exclusions, source-acquisition hashes, prediction commit,
and any refusal are reported. Lawlor and Hao are both run regardless of the
first result. Unsuccessful transfer remains in the public benchmark table.
