# Held-donor PBMC RNA-protein confirmation protocol

**Protocol version:** `lawlor-pbmc-coupling/1.0`  
**Designated locally:** 27 August 2026, before downloading the RNA or ADT
matrices  
**Status:** prospective local specification; it is not externally registered
until these unchanged bytes and the sealed candidate manifest are deposited in
a public repository or archive.

## Question and study

This experiment tests whether a coupling field learned from six donors predicts
stimulus-induced RNA-protein dependence in four untouched donors, and whether
it improves conditional held-table prediction over interaction residuals from
the standard independence model.

The designated study is the Lawlor et al. PBMC CITE-seq project, Human Cell
Atlas project `efea6426-510a-4b60-9a19-277e52bfa815` (ERP124005;
doi:10.3389/fimmu.2021.636720). It contains paired RNA and 39-antibody ADT
measurements from ten healthy adults at baseline, after LPS, and after
anti-CD3/CD28 stimulation. This is a held-donor stimulus-by-cell-type
RNA-protein experiment, not CRISPR target transfer.

Only this dataset is authorized under this protocol. A refusal or failure stays
in the benchmark. Testing another candidate requires a new dated protocol and
cannot replace this result.

## Outcome seal

Before an RNA or ADT matrix is downloaded,
`benchmark_release/coupling_fields_v1/LAWLOR_CANDIDATE_DESIGNATION.json` must
contain:

- the exact HCA URLs, byte counts, and SHA-256 checksums;
- the metadata-only support artifact and its checksum;
- donor labels, donor split, retained contrasts, and structural exclusions;
- this protocol's checksum;
- the complete analysis runner and its checksum; and
- an immutable public commit identifier and URL containing those unchanged
  artifacts; and
- `status: SEALED` and `outcome_access_authorized: true`.

Any other status forbids outcome access. The sealed runner executes once and
has separate `predict` and `score` commands. `predict` writes a JSON artifact
containing every coordinate and held joint-table prediction without forming
held stimulus pairings. Its exact SHA-256 and byte count must then be recorded
at an immutable public commit and URL in
`LAWLOR_SCORE_AUTHORIZATION.json`. `score` refuses unless that authorization is
sealed, its prediction hash and runner hash match, and outcome access is
explicitly authorized. Only then may it form held stimulus joint tables. The
score is append-only and records all declared methods, intervals, and the
decision. A correction after pairing access is post-lock sensitivity analysis
and cannot overwrite the confirmation.

## Metadata-only cohort and split

The metadata audit uses `CZI.PBMC.cell.annotations.csv` (26,242,593 bytes;
SHA-256 `76a6548089867687461603ed0031ba02f9203885cd8a623b7497bdceccaf7229`).
It retains author-annotated cells with an HTO singlet label in `Baseline`,
`LPS`, or `CD3_CD28`, Demuxlet classification `SNG`, and a nonmissing author
broad cell type. This leaves 16,382 metadata-eligible cells.

Canonical `Donor_of_Origin` labels are ordered by ascending
`SHA256("pbmc-citeseq-coupling-v1" + donor_label)`. The first six are
development donors and the final four are held confirmation donors:

| Split | Author aliases | Canonical labels |
|---|---|---|
| Development | Donor3, Donor5, Donor1, Donor8, Donor6, Donor9 | `202937150118_R01C01`, `202937150118_R03C01`, `202937150091_R01C01`, `202937150118_R06C01`, `202937150118_R04C01`, `202937150118_R07C01` |
| Confirmation | Donor7, Donor2, Donor10, Donor4 | `202937150118_R05C01`, `202937150091_R02C01`, `202937150118_R08C01`, `202937150118_R02C01` |

No held-donor RNA value, ADT value, field, table, residual, metric, or summary
may enter filtering, normalization, threshold fitting, estimator choice, or
penalty selection. After the runner is sealed, each held donor's baseline joint
table is opened as a common calibration anchor for all methods. Held stimulus
pairing remains sealed until every predicted stimulus table has been written to
a checksum-bound prediction-lock artifact; only its separately computed RNA
and ADT margins are supplied during reconstruction. The runner then opens the
pairing once to score those immutable tables.

## Frozen biological grid

The primary grid follows the study's targeted lineages and requires at least 15
metadata-eligible cells in every donor's baseline and stimulus arm.

1. anti-CD3/CD28 in B cells;
2. anti-CD3/CD28 in naive CD4 T cells;
3. anti-CD3/CD28 in memory CD4 T cells;
4. anti-CD3/CD28 in naive CD8 T cells;
5. anti-CD3/CD28 in memory CD8 T cells; and
6. LPS in CD14 monocytes.

All six contrasts pass the metadata threshold. Anti-CD3/CD28 in CD14 monocytes
is structurally absent. NK cells are excluded because one donor has seven
baseline cells. LPS contrasts outside CD14 monocytes are excluded by the
study-targeted lineage rule. These exclusions were fixed without RNA or ADT
values.

Every development-eligible matched RNA-ADT marker is evaluated on the complete
six-contrast grid. Each 3-by-3 table contributes four Helmert interaction
coordinates, giving 24 outputs per marker. No marker, output, or donor may be
removed from the primary grid using a held value.

## Marker matching and states

ADT labels are mapped to cognate human gene symbols by a sealed, explicit
lookup in the runner. Antibodies without one unambiguous RNA counterpart are
excluded. Marker eligibility uses development baseline cells only. A pair must
be observed in both assays, have two distinct development-baseline tertile
cuts in each assay, and place at least 5% of pooled development-baseline cells
in every state of each assay. Failing pairs are listed before held outcomes are
opened.

RNA counts are library-size normalized to 10,000 and transformed with `log1p`.
ADT counts use per-cell centered `log1p` values. For each matched marker, the
one-third and two-third quantiles in pooled development baseline cells define
three RNA states and three ADT states. The same thresholds are applied without
refitting to every stimulus arm and held donor. Tied quantiles or a missing
state produce a marker-level support refusal; no alternative clustering is
tried.

For marker `g`, donor `d`, lineage `l`, and condition `c`, linked cells form a
3-by-3 RNA-ADT table. The association coordinate is the double-centered
log-table field after subtracting the mean from 64 deterministic fixed-margin
link permutations. One additional, disjoint permutation is the destroyed-link
control. Fixed-margin null draws are generated from canonical state vectors
constructed only from the two marginal count vectors, so their exact finite
sample values cannot encode cell pairing. The outcome is the
stimulus-minus-baseline centered-coordinate contrast.

## Fixed estimators and baselines

Development donors are averaged only after donor-specific contrasts are
constructed. The fixed primary applies inverse-permutation-variance weights,
clipped to `[0.05, 20]` after median normalization, and the existing
nuclear-plus-frozen-hypergraph fit. Its nuclear threshold is `0.1` times the
leading development singular value and its graph penalty is `5`.

Every method uses the same markers, six biological contrasts, development
donors, held donors, state thresholds, held-table endpoint, and bootstrap
draws. The coupling and covariance representations have four coordinates per
contrast; the primary classical comparator retains all nine entries of each
residual matrix.

1. direct development coupling fields;
2. zero, variance-scalar, and nuclear shrinkage;
3. frozen-hypergraph shrinkage;
4. the fixed variance-weighted nuclear-plus-hypergraph primary;
5. endpoint-margin ridge with `alpha=0.1`;
6. linear cross-covariance interaction;
7. direct and identically structured full Pearson residual matrices from the
   row-by-column independence model;
8. direct and identically structured full signed Poisson-deviance residual
   matrices; and
9. the destroyed-link field control and membership-permuted graph control.

The variance-scalar estimator multiplies the direct field by
`max(0, 1 - sum(V)/sum(F^2))`, where `F` is the development mean field and `V`
is its fixed permutation-variance estimate. The membership control permutes
the rows and columns of the frozen graph Laplacian with NumPy seed
`SHA256("membership-permuted")[:4]`, then applies the same nuclear and graph
penalties as the primary.

The coupling field is a log-linear interaction parameterization and is not
presented as distinct from a saturated log-linear interaction. The classical
head-to-head tests the complete field pipeline against residuals from the
independence model.

## Common held-table endpoint

All methods predict the same held stimulus 3-by-3 joint tables. For each held
donor, marker, and lineage, the observed baseline joint table supplies an
absolute interaction anchor. Every method adds its predicted
stimulus-minus-baseline centered contrast to its own representation of that
same baseline table. Because the fitted contrast is fixed-margin centered, the
absolute stimulus representation also includes the deterministic shift from
the baseline null mean to the stimulus null mean. If `R_B` is the raw baseline
representation, `mu_B` and `mu_S` are the baseline and stimulus null means, and
`Delta` is the predicted centered contrast, field reconstruction uses
`R_S = R_B + Delta + mu_S - mu_B`. Both null means depend only on margins and
the sealed seed. This rule is applied identically to coupling fields, Pearson
residuals, Poisson-deviance residuals, and controls. The observed held stimulus
row and column margins are supplied to every method as nuisance information;
stimulus pairing within the table remains hidden until every predicted table is
written.

For a baseline-anchored coupling prediction, the runner lifts the four Helmert
coordinates to a zero-sum log-interaction matrix, exponentiates it, and uses
iterative proportional fitting to match the held stimulus margins. The primary
Pearson and deviance comparators retain each full 3-by-3 residual matrix; no
Helmert projection discards residual entries. Development centered residuals
are divided by the square root of each arm's table total before contrasts are
formed. The predicted scale-free contrast is added to the held baseline
centered residual divided by the square root of its baseline table size,
restored at the held stimulus table size, and shifted by the stimulus null
mean: `R_S = ((R_B - mu_B)/sqrt(n_B) + Delta) sqrt(n_S) + mu_S`. For
Pearson, the resulting full residual matrix forms

`N_star = E + residual * sqrt(E)`,

where `E` is the held stimulus independence expectation. Nonpositive cells are
floored at machine-positive mass before iterative proportional fitting to the
same held stimulus margins. The signed Poisson-deviance comparator uses the
same full-matrix scaling and the unique sign-compatible inverse of its residual
equation before the same
flooring and margin fit. Thus every comparator starts from the same kind of
observed baseline anchor and yields a positive table with exactly the same held
stimulus margins.

The primary loss is multinomial deviance from the observed held table to the
predicted table, divided by the held table total. Losses are averaged over the
six contrasts for each marker and then over the four held donors. The lower of the
Pearson- and Poisson-residual losses is the best classical comparator. Its
selection is repeated inside every bootstrap draw.

Representation-space pooled Pearson correlation and standardized RMSE are
secondary. They are reported for the coupling field and both full residual
representations but do not replace the common held-table endpoint.

## Uncertainty and decision

Exactly 2,000 paired marker bootstrap draws use NumPy
`default_rng(20260827)`. Each draw resamples complete marker blocks and applies
the same indices to every donor, output, method, and control. Cells are never
treated as replicates. Donor-specific correlations are reported; marker
bootstrap intervals condition on these ten deposited donors and are not
population-level donor intervals.

The confirmation passes only when every condition holds:

1. the lower endpoint of the 95% marker-bootstrap interval for pooled held-donor
   coupling-field correlation is greater than zero;
2. the coupling-field correlation is positive in each of the four held donors;
3. the upper endpoint of the paired 95% interval for primary-minus-destroyed
   per-cell deviance is below zero;
4. the upper endpoint of the paired 95% interval for
   primary-minus-best-classical per-cell deviance is below zero; and
5. the upper endpoint of the paired 95% interval for
   primary-minus-best-matched-field per-cell deviance is below zero, where
   direct, zero, variance-scalar, nuclear-only, hypergraph-only, endpoint,
   covariance, and membership-permuted methods are reselected within each
   draw; and
6. marker matching, state support, fixed-margin centering, iterative
   proportional fitting, and the structured fit complete without refusal.

This is an intersection-union gate. No component can compensate for another.
The protocol authorizes one dataset and one pooled primary grid, so it has no
cross-dataset multiplicity. Donor-, marker-, lineage-, and stimulus-specific
tests are secondary and use Benjamini-Hochberg correction within each declared
family. They cannot change the primary decision.

## Exclusions, deviations, and release

The structural exclusions are HTO multiplets or empty droplets, the IgM/IgG
HTO condition, non-singlet Demuxlet calls, missing author broad labels, the two
metadata-excluded lineage contrasts, ambiguous RNA-ADT matches, and marker
support failures defined above. No observation is excluded because of an
effect, residual, correlation, influence, or held loss.

Insufficient support, a degenerate state, nonfinite coordinates, incompatible
margins, failed proportional fitting, or optimizer nonconvergence yields
`REFUSE`. Relaxing a threshold, changing a marker map, removing a comparator,
selecting outputs, or choosing another dataset after outcome access is
exploratory. The frozen result, including a refusal or negative comparison,
remains in the public benchmark table.

The release record must bind this protocol, candidate designation, metadata
support artifact, source manifest, runner, estimator, result, and benchmark
table by SHA-256. Target-level predictions, truths, common held-table losses,
and bootstrap draws are released when upstream terms permit. A local protocol
or package is not called preregistered or public until an immutable public URL
and archive DOI exist.
