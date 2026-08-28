# Held-donor PBMC RNA-protein confirmation protocol

**Protocol version:** `lawlor-pbmc-coupling/2.0`

**Designated locally:** 28 August 2026, before downloading the RNA or ADT
matrices

**Status:** `OUTCOME_ACCESS_DISABLED`; version 2 supersedes version 1 before
outcome access and requires a new public freeze of the hardened runner and
disabled designation before either outcome matrix may be downloaded.

## Question and study

This experiment tests whether coupling fields learned from six donors predict
stimulus-induced RNA-protein dependence in four held donors, and whether
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
`data/confirmation/pbmc_citeseq_hca/candidate_designation_v2.json` must
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
has separate `predict` and `score` commands. Confirmatory `predict` always runs
the checksum-validated reducer from the three deposited source files into the
fixed, previously absent `data/development/lawlor_hca_pbmc/reduced_v2`
directory. It has no reduced-data bypass. The runner verifies the reducer's
exact five-file manifest, rejects extra files and symbolic links, and records
the relative path, byte count, and SHA-256 of all five outputs and the manifest
itself in prediction provenance. It verifies those bytes again after fitting.

`predict` writes every coordinate, entity definition, and pairing-independent
held margin to a JSON artifact; it stores no predicted or observed held
stimulus joint table. The prediction must be published at an exact GitHub
`blob` URL. `public-bind` then writes
`data/confirmation/pbmc_citeseq_hca/score_authorization_v2.json`, binding the
prediction path, bytes, SHA-256, public commit, runner, and protocol. That
authorization must be published in a later immutable commit. `authorize-score`
binds its public location in
`data/confirmation/pbmc_citeseq_hca/score_release_v2.json`. `score` refuses
without both records. Its refusal-protected section begins before any held
baseline pairing or table reconstruction. It first recomputes every coordinate
from the checksum-bound reducer outputs using development joint tables and
pairing-independent held margins only, and requires bitwise equality to the
published coordinate arrays. Only then does it open held baseline pairing,
reconstruct every method's table from the verified coordinates and held
margins, and check all row and column sums. A correction after pairing access is
post-lock sensitivity analysis and cannot overwrite the confirmation.

This is a public code-path seal, not a blinded enclave or registry-hosted
preregistration. The same public checkout contains prediction and scoring code;
the executable barriers establish that coordinates and both authorization
records are immutable before held joint pairing is opened.

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

No held-donor joint pairing, field, residual, metric, or loss may enter state
thresholds, estimator choice, or penalty selection. Pairing-independent held
RNA and ADT margins enter only the prespecified support check below. Each held
donor's baseline joint table is a common calibration anchor for all methods.
It is first opened inside the authorized score, after coordinates are public;
held stimulus pairing remains sealed while the anchored tables are
reconstructed from separate RNA and ADT margins. The runner then opens stimulus
pairing once to score the immutable coordinate predictions.

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

The analysis unit is an eligible matched marker-contrast pair. Each pair
contributes four Helmert interaction coordinates for one frozen
stimulus-lineage contrast. At least 12 pairs spanning at least 12 unique matched
marker clusters must pass the prespecified marginal-support rule. Held joint
pairing never enters eligibility.

## Marker matching and states

ADT labels are mapped to cognate human gene symbols only through the sealed
alias table; token-to-gene fallback is forbidden. Case-normalized RNA symbols
and alias-table gene symbols must each be unique. Antibodies without one
unambiguous gene-level RNA counterpart are excluded. CD45RA and CD45RO are
excluded because total `PTPRC` counts do not resolve their isoform-specific
epitopes. State thresholds are fitted separately for every matched marker and
frozen lineage using pooled development-donor baseline cells from that lineage.
A marker-contrast pair requires two distinct cuts in each assay and at least 5%
occupancy in every RNA and ADT state in every development donor's baseline and
stimulus arm. Before prediction, the same requirement is applied separately to
the RNA and ADT margins in every held donor's baseline and stimulus arm. This
held check is invariant to cell pairing. Every excluded pair and reason is
written to the prediction artifact.

RNA counts are library-size normalized to 10,000 and transformed with `log1p`.
ADT counts use per-cell centered `log1p` values. For each marker-lineage pair,
the one-third and two-third quantiles in pooled development baseline cells
define three RNA states and three ADT states. The same thresholds are applied
without refitting to the corresponding stimulus arm and held donors. Fewer than
12 eligible pairs or 12 unique eligible marker clusters produces a deterministic
support refusal; no alternative clustering is tried. Prediction is invariant to
any within-donor, condition, and lineage permutation of held ADT cell pairing
because eligibility and held predictor inputs use only separate margins.

For eligible pair `e`, donor `d`, and condition `c`, linked cells form a
3-by-3 RNA-ADT table. The association coordinate is the double-centered
log-table field after subtracting the mean from 64 deterministic fixed-margin
link permutations. One additional, disjoint permutation is the destroyed-link
control. Fixed-margin null draws are generated from canonical state vectors
constructed only from the two marginal count vectors, so their exact finite
sample values cannot encode cell pairing. The outcome is the
stimulus-minus-baseline centered-coordinate contrast.

## Fixed estimators and baselines

Development donors receive equal weight and are averaged only after
donor-specific contrasts are constructed. Each estimator operates on an
entity-by-four-coordinate matrix. The fixed primary uses inverse-permutation-
variance weights, clipped to `[0.05, 20]` after median normalization, in its
structured loss. Its nuclear threshold is `0.1` times the leading development
singular value and its graph penalty is `5`.

The hypergraph is constructed after marginal support is fixed. For every
covered marker, one gene-embedding hyperedge contains all eligible entities for
that marker and its six nearest distinct eligible marker genes in the frozen
scGPT embedding. Additional typed hyperedges join entities with the same frozen
contrast and the same lineage. An uncovered gene receives a marker-only
hyperedge. Fewer than seven covered markers is a support refusal. The membership
control permutes gene-incidence rows among whole marker clusters before the
Laplacian is formed; contrast and lineage incidence remains unchanged.

Every method uses the same eligible marker-contrast pairs, donors, thresholds,
held-table endpoint, and bootstrap draws. Coupling and covariance
representations have four coordinates per entity; each classical residual
comparator retains all nine matrix entries per entity.

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
9. the destroyed-link field passed through the same variance weighting,
   nuclear penalty, and hypergraph penalty as the primary, and the
   membership-permuted graph control.

The variance-scalar estimator multiplies the direct field by
`max(0, 1 - sum(V)/sum(F^2))`, where `F` is the development mean field and `V`
is its fixed permutation-variance estimate. The membership control uses NumPy
seed `SHA256("membership-permuted")[:4]` to permute marker-cluster gene
incidence while retaining the exact contrast and lineage hyperedges, then
applies the same nuclear and graph penalties as the primary.

The coupling field is a log-linear interaction parameterization and is not
presented as distinct from a saturated log-linear interaction. The classical
head-to-head tests the complete field pipeline against residuals from the
independence model.

## Common held-table endpoint

All methods predict the same held stimulus 3-by-3 joint tables. For each held
donor and eligible marker-contrast pair, the observed baseline joint table supplies an
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
stimulus pairing remains hidden until the coordinates are public and every
predicted table has been reconstructed in the authorized score process.

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
floored at `10^-3` times the mean held-table cell count before iterative
proportional fitting to the same held stimulus margins. The signed
Poisson-deviance comparator uses the same full-matrix scaling and the unique
sign-compatible inverse of its residual equation before the same flooring and
margin fit. Thus every comparator starts from the same kind of
observed baseline anchor and yields a positive table with exactly the same held
stimulus margins.

The primary loss is multinomial deviance from the observed held table to the
predicted table, divided by the held table total. Losses are averaged over the
eligible pairs and four held donors. The lower of the
Pearson- and Poisson-residual losses is the best classical comparator. Its
selection is repeated inside every bootstrap draw.

Representation-space pooled Pearson correlation and standardized RMSE are
secondary. They are reported for the coupling field and both full residual
representations but do not replace the common held-table endpoint.

## Uncertainty and decision

Exactly 2,000 paired cluster-bootstrap draws use NumPy
`default_rng(20260827)`. Each draw resamples matched markers and retains every
eligible contrast pair for each sampled marker; the same entity indices apply
to every donor, coordinate, method, and control. Cells are never treated as
replicates. Donor-specific correlations are reported. These intervals condition
on the deposited donors and do not support population-level donor inference.

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
   proportional fitting, the six-neighbor graph, and every structured fit
   complete without refusal.

This is an intersection-union gate. No component can compensate for another.
The frozen confirmation campaign has two untouched, scoreable candidates:
Lawlor HCA PBMC and Hao `GSE164378`. Both are executed and reported; neither
result can suppress the other candidate. The earlier POKI-seq candidate reached a preflight state-
occupancy refusal before any statistical test and is retained as a refusal, not
counted as a scoreable test. Each directional condition above uses one endpoint
of a two-sided 95% interval, hence one-sided alpha `0.025`. Bonferroni over the
two scoreable candidates bounds the campaign familywise error rate at `0.05`
without changing the reported 95% intervals. Donor-, marker-, lineage-, and
stimulus-specific tests are secondary and use Benjamini-Hochberg correction
within each declared family. They cannot change the primary decision.

## Exclusions, deviations, and release

The structural exclusions are HTO multiplets or empty droplets, the IgM/IgG
HTO condition, non-singlet Demuxlet calls, missing author broad labels, the two
metadata-excluded lineage contrasts, ambiguous RNA-ADT matches, and
marker-contrast marginal-support failures defined above. No observation is excluded because of an
effect, residual, correlation, influence, or held loss.

After the public authorization, immutable-path, and output-absence gates pass,
sealed-data parsing begins inside the terminal refusal scope and before any held
pairing is opened. Insufficient support, a degenerate state, nonfinite
coordinates, incompatible margins, failed proportional fitting, or optimizer
nonconvergence writes a deterministic `REFUSE_EXECUTION` record at the frozen
prediction or score path and forbids a replacement run. Relaxing a threshold, changing a marker map,
removing a comparator, selecting outputs, or choosing another dataset after
outcome access is exploratory. The frozen result, including a refusal or
negative comparison, remains in the public benchmark table.

The release record must bind this protocol, version-2 candidate designation,
authorization template, version-2 score authorization and score release,
metadata support artifact, source manifest, runner, reducer, test, estimator, result, and benchmark
table by SHA-256. Target-level predictions, truths, common held-table losses,
and bootstrap draws are released when upstream terms permit. The protocol is
called public only after it has an immutable public URL. Without a registry
record or archive DOI, it is not described as registry-hosted preregistration
or DOI-archived.
