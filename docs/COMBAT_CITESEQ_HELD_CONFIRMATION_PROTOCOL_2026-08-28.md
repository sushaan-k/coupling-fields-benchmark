# COMBAT CITE-seq held-sample confirmation protocol

**Version:** 1.0, 28 August 2026

**Data:** COMBAT CITE-seq, Zenodo DOI `10.5281/zenodo.6120249`

**Study:** COMBAT Consortium, *Cell* (2022), DOI
`10.1016/j.cell.2022.01.012`

**Status:** designated with outcome access disabled

## Question and endpoint

This experiment tests whether an RNA--ADT coupling field learned in paired
Oxford samples predicts same-cell RNA--ADT dependence in untouched Oxford and
St George's samples after conditioning on each recipient's observed RNA margins
and design-fixed ADT margins. For truth table `T` and a finite, nonnegative,
margin-preserving prediction `T_hat`, entity loss is
`D=(2/N)*sum_{ij:T_ij>0} T_ij*log(T_ij/T_hat_ij)`. A positive truth cell paired
with a zero prediction gives `+infinity`; a zero truth cell contributes zero.
The primary endpoint averages `D` equally over informative marker pairs within
sample and then equally over samples. A nonfinite or negative predicted cell or
a margin mismatch is a procedural refusal; an infinite loss from a valid
limiting endpoint is a scored failure. The primary comparisons are the strongest
matched classical residual transfer and a pairing-destroyed coupling field.

The 24-sample pilot is adaptive development evidence. The 51-sample Oxford and
10-sample St George's panels are separate confirmatory panels. Primary-method
promotion requires field-transfer and graph-specific gates in both panels; a
pooled result cannot replace either decision.

## Source and metadata seal

The sole matrix source is `COMBAT-CITESeq-DATA.h5ad` from the official Zenodo
record. The required object is 6,409,089,483 bytes, with MD5
`87c6b1a733ea1adc37c808d9a357de74` and SHA-256
`f628a15f25b9ca2f7cdefeab271fd9d007c8d5e47eb80c4b807d5f65e86ff53d`.
Its content URL is
`https://zenodo.org/api/records/6120249/files/COMBAT-CITESeq-DATA.h5ad/content`.
Acquisition must verify and bind all three values before analysis.

The sample universe is fixed by the official
`CBD-KEY-CITESEQ-GEX-COMPOSITION.tar.gz`, obtained only from
`https://zenodo.org/api/records/6120249/files/CBD-KEY-CITESEQ-GEX-COMPOSITION.tar.gz/content`.
The tarball is 212,203 bytes, with MD5
`fe077fc9f314419536d6d901855c9d84` and SHA-256
`cc6b50cb363b800f356aa79224240f96ca11046a9c2a1c5f4f78603531b3dae3`.
Its exact member
`COMBAT_CITEseq_Composition-PerSample_CellType_Counts_and_PercentFrequencies_out_of_all_PBMCs.csv`
is 33,391 bytes, with MD5 `c4e635d7f16b3f7e3e66571a3359de3b` and SHA-256
`2ad7e92ab122ee52986d5748dbb23c335c02ec1f1f244943ce46fff94c585157`.
The universe is the 97 unique `(COMBAT_ID, scRNASeq_sample_ID)` pairs obtained
from that member and enumerated in the designation. Each exact pair must occur
in the H5AD, whose metadata must yield the frozen institute counts and Oxford
source strata. H5AD rows belonging to any other pair, including additional
timepoints, are outside the experiment and are excluded rather than substituted.

A metadata-only preflight may inspect HDF5 names, attributes, shapes, `obs` and
`var` encodings, and string or categorical metadata. It may hash opaque file
bytes. It may not index or decode `X`, `layers/raw`, or any matrix `data`,
`indices`, or `indptr` dataset. The source manifest must establish the exact 97
designated pairs, institute counts of 87 `Oxford` and 10 `St_Georges`, and the
six Oxford `Source` strata below. The eligible H5AD cell pool is the exact
non-null `Annotation_cell_type` allowlist `B`, `ERYTH`, `HSC`, `MNP`, `NK`,
`PB`, `PLT`, and `T`; every designated sample must contain at least 512 such
cells. `Annotation_cell_type` is an author-derived assay annotation used only
to define this cell pool. No direct RNA or ADT matrix value enters eligibility
or cell selection. `MNP|PLT` occurs in composition metadata but is not an H5AD
cell label and is not selectable. The manifest records each H5AD pool count.
No equality with the descriptive composition-file `TotalPBMC_Count` is imposed.
Any discrepancy is a terminal pre-outcome refusal; no sample may be replaced.

The metadata preflight, source manifest, protocol, designation, runner, and
fresh-clone verification must be public and hash-bound before development
outcomes are decoded. A separately committed
`data/confirmation/combat_citeseq/development_authorization_v1.json` must have
schema `combat-citeseq-development-authorization/1.0`, status
`OUTCOME_ACCESS_AUTHORIZED`, a 40-hex public freeze commit, and exact path and
SHA-256 bindings for the runner, runner test, protocol, designation, source
manifest, metadata preflight script and result, fresh-clone verification, and
every imported numerical module. The frozen numerical-module set is
`experiments/evaluate_gse279451_sepsis_development.py`,
`experiments/reduce_gse279451_sepsis.py`,
`mapreg/heterogeneity_adaptive_coupling.py`,
`mapreg/hierarchical_conditional_coupling.py`, `mapreg/coupling_fields.py`,
`mapreg/classical_residuals.py`, and `mapreg/table_prediction.py`; adding
another numerical import requires adding its exact binding before authorization.
The fresh-clone binding is
`docs/COMBAT_CITESEQ_PUBLIC_FREEZE_VERIFICATION_2026-08-28.json`. Development
access refuses unless every binding and the verification artifact matches.

## Prospective sample allocation

Allocation uses metadata only. Within each Oxford `Source` stratum, calibration
samples are sorted by the ascending hexadecimal digest

```text
SHA256("COMBAT-OXFORD-CALIBRATION-v1" || NUL || Source || NUL ||
       COMBAT_ID || NUL || scRNASeq_sample_ID)
```

with `scRNASeq_sample_ID` as the final lexical tie break; the first two are
calibration samples. After removing them, the same operation with prefix
`COMBAT-OXFORD-PILOT-v1` selects the first four pilot samples per stratum.
UTF-8 encodes every field and `NUL` is the single byte `0x00`.

| Oxford source | Calibration | Pilot |
|---|---|---|
| `COVID_CRIT` | `S00024`, `S00027` | `S00008`, `S00020`, `S00040`, `S00052` |
| `COVID_HCW_MILD` | `G05077`, `G05171` | `G05061`, `G05097`, `G05145`, `G05164` |
| `COVID_MILD` | `S00002`, `S00126` | `S00063`, `S00076`, `S00104`, `S00114` |
| `COVID_SEV` | `S00045`, `S00148` | `S00037`, `S00042`, `S00053`, `S00134` |
| `HV` | `H00052`, `H00054` | `H00058`, `H00064`, `H00070`, `H00072` |
| `Sepsis` | `N00032`, `N00050` | `N00006`, `N00024`, `N00025`, `N00047` |

The digest rules determine membership. Downstream locked sample order is the
table's source order and then the displayed lexical `COMBAT_ID` order within
each source and role.

All other 51 eligible Oxford samples, ordered lexically by `COMBAT_ID`, form the
Oxford held panel. The St George's held panel is exactly `U00501`, `U00502`,
`U00503`, `U00505`,
`U00601`, `U00605`, `U00607`, `U00617`, `U00619`, and `U00701`. Neither held
panel participates in fitting, configuration selection, graph construction,
or pilot promotion.

## Cells, markers, and tables

Each sample contributes exactly 512 cells. Cells are ordered by

```text
SHA256("COMBAT-PBMC-CELL-BUDGET-v1" || NUL || COMBAT_ID || NUL ||
       scRNASeq_sample_ID || NUL || obs_name)
```

and the first 512 eligible cells are retained, with `obs_name` as the final
lexical tie break. Selection uses no RNA or ADT value. The ordered marker panel
and exact RNA feature IDs are:

| Marker | Ensembl ID | ADT alias |
|---|---|---|
| `CD4` | `ENSG00000010610` | `AB_CD4` |
| `CD7` | `ENSG00000173762` | `AB_CD7` |
| `CD14` | `ENSG00000170458` | `AB_CD14` |
| `CD19` | `ENSG00000177455` | `AB_CD19` |
| `CD33` | `ENSG00000105383` | `AB_CD33` |
| `CD38` | `ENSG00000004468` | `AB_CD38` |
| `CD44` | `ENSG00000026508` | `AB_humanCD44` |
| `CD47` | `ENSG00000196776` | `AB_CD47` |
| `CD52` | `ENSG00000169442` | `AB_CD52` |

The raw layer must contain one exact `Gene Expression` feature for each Ensembl
ID and one exact `Antibody Capture` feature for each ADT alias. In particular,
CD44 has the dataset-specific exact alias `AB_humanCD44`; no fuzzy or normalized
fallback is permitted. Missing, duplicate, nonfinite, negative, or non-integer
raw features cause refusal.

RNA state is one for a positive raw count. For each sample and ADT marker,
cells are sorted by raw ADT count and then by

```text
SHA256("COMBAT-PBMC-ADT-v1" || NUL || COMBAT_ID || NUL ||
       scRNASeq_sample_ID || NUL || obs_name || NUL || marker)
```

The first 256 cells receive the lower state and the remaining 256 the upper
state. All 81 ordered RNA-marker by ADT-marker pairs are retained. A pair is
informative when its fixed margins admit more than one 2-by-2 table; a sample
is scoreable with at least 64 informative pairs. Every method uses the same
margin-only informative mask.

## Methods and development

For binary RNA and ADT states, the Helmert specialization of the population
coupling field is one-half of the log odds ratio. The Haldane coordinate below
is twice that field after finite-cell correction, and exact hypergeometric-null
centering is the exhaustive fixed-margin counterpart of permutation centering.
The product graph then shares information across the RNA-by-ADT marker field.
This experiment prospectively validates that binary product-graph
specialization. It does not retroactively validate the nuclear, fixed-embedding,
or guide-aware estimators used in earlier benchmark panels.

Write a 2-by-2 table with upper-left count `a`, fixed margins `M`, and total
`N=512`. Its Haldane coordinate is

```text
h_M(a) = log((a + 1/2)(d + 1/2) / ((b + 1/2)(c + 1/2))).
```

Under the fixed-margin hypergeometric null, let `m0(M)=E0[h_M(A)]`. Each source
table contributes `z=h_M(a)-m0(M)` and its exact enumerated null variance.
Entity-wise Paule--Mandel (PM) random-effects pooling requires at least two
supported donors and floors each within-donor variance at `1e-8`. For
`w_i(t)=1/(v_i+t)` and
`mu(t)=sum_i w_i(t)z_i/sum_i w_i(t)`, PM sets `t=0` when
`sum_i w_i(0)(z_i-mu(0))^2 <= n-1`; otherwise it finds the unique nonnegative
root equaling `n-1`. It returns `mu(t)` and precision `sum_i w_i(t)` for each of
the 81 ordered marker pairs.

The primary Haldane/PM product-graph field smooths these pooled coordinates with
one precision-weighted graph linear solve. Positive PM precisions are divided by
their median; supported diagonal precisions are floored at `1e-8`. The solve
refuses a nonfinite system or condition number above `1e12`. The graph solve,
not PM pooling or recipient reconstruction, is the method's only closed-form
step.

Recipient reconstruction preserves its fixed margins. For a recipient margin
set `M`, define the Fisher noncentral-hypergeometric family
`P_theta(A=a|M) proportional to P_0(A=a|M) exp(theta*a)`. Given transferred
coordinate `z*`, solve
`E_theta[h_M(A)]-m0(M)=z*` and predict `E_theta[T(A)]`. The left side is strictly
increasing on nondegenerate support. An exact attainable endpoint returns its
limiting table and is flagged; a value outside the closed attainable interval
refuses. A one-table margin returns that unique table, is flagged degenerate,
and is excluded from informative loss. Thus the estimator transports an
expected centered Haldane coordinate; it does not invert a pooled coordinate as
though it were a realized fractional count.

The fixed eight configurations are the Cartesian product of graph neighborhood
size `k` in `{1, 2}`, ridge penalty in `{0.01, 0.1}`, and graph penalty in
`{0.1, 1}`. Ties use the first tuple in lexicographic `(k, ridge, graph)` order.

Graph-profile columns are every nonempty `(fit sample,
Annotation_cell_type)` stratum among the selected 512 cells, in locked sample
order and then the declared eight-label order. An RNA marker value is its binary
detection prevalence in the stratum. An ADT marker value is the stratum mean of
`log1p(100 * marker count / nine-marker ADT total)`; a cell with zero
nine-marker total contributes a zero vector. Each marker profile is centered
and scaled across strata with `ddof=1`; zero variance is a refusal. Euclidean
directed `k`-nearest neighbors use locked marker order for distance ties. Their
undirected union, in lexical endpoint order, defines each modality graph.

Each undirected edge becomes one unweighted incidence column with exactly two
unit entries. With incidence `H`, unit edge weights `W`, vertex degree `D_v`,
and edge degree `D_e`, the frozen normalized hypergraph Laplacian is
`L=I-D_v^(-1/2) H W D_e^(-1) H^T D_v^(-1/2)` on supported vertices, with
isolated rows and columns zero. The 81-entity Laplacian is the Kronecker sum
`L_RNA (x) I_9 + I_9 (x) L_ADT`, using RNA-major, ADT-minor ordered pairs.
Graphs are built from the 12 calibration samples for pilot evaluation and
rebuilt from all 36 non-held samples after promotion. No held row enters a
profile or graph.

The frozen comparison set is:

- the strongest transfer among raw and exact-null-centered normalized signed
  Pearson and signed-root Poisson-deviance coordinates;
- raw Haldane log-odds transfer pooled by Paule--Mandel using the Haldane
  delta-method sampling variance;
- the primary family fitted after deterministic pairing destruction;
- an unstructured ridge-only PM Haldane field, pilot-selected over ridge
  penalties `{0, 0.01, 0.1}` with graph penalty zero;
- the primary family with independently label-permuted RNA and ADT graphs; and
- independence at the recipient margins.

For pairing destruction, cells within each source sample are ordered by
`SHA256("COMBAT-DESTROYED-LINK-v1" || NUL || scRNASeq_sample_ID || NUL ||
obs_name)`. The ADT profile of each cell is replaced by that of the next cell
in the cyclic order. This preserves every source ADT margin and all within-ADT
relationships while removing the original RNA--ADT links. The destroyed-link
fit reuses the primary field's unpermuted marginal-profile graphs; only its
source joint tables change.

For the label-permuted control, RNA and ADT marker labels are ordered
separately by `SHA256("COMBAT-LABEL-PERMUTATION-v1" || NUL || modality || NUL
|| marker)` and reassigned to graph positions in lexical marker order. The
unstructured ablation uses the same Haldane coordinates, PM rules, precision
scaling, and moment-calibrated recipient reconstruction as the primary field but
no product-graph smoothing. Ridge zero retains the raw PM pool.

For each classical comparator, let `e_ij=r_i*c_j/N` and let the sign be that of
`ad-bc`, with `sign(0)=0`. The signed Pearson statistic is
`sign*sqrt(sum_ij (t_ij-e_ij)^2/e_ij)`. The signed-root Poisson-deviance
statistic is `sign*sqrt(2*sum_{t_ij>0} t_ij*log(t_ij/e_ij))`. The raw normalized
coordinate is
`q_M(a)=s_M(a)/sqrt(N)`; the exact-null-centered coordinate is
`q_M(a)=(s_M(a)-E0[s_M(A)])/sqrt(N)`. Its PM within-sample variance is the exact
enumerated `Var0[s_M(A)]/N`, subject to the same `1e-8` floor and two-donor
minimum. Raw and centered forms remain separate pilot candidates. Recipient
restoration uses the same Fisher tilt: solve `E_theta[q_M(A)]=q*` for the pooled
target coordinate and return `E_theta[T(A)]`. The full integer support is
enumerated, strict monotonicity is verified, exact endpoints return flagged
limiting tables, out-of-range targets refuse, one-table margins return the
flagged unique table, and zero null variance is unsupported. This freezes a
sample-size-normalized classical transfer rather than a direct residual-to-count
substitution.

Before pilot selection, the runner revalidates the public development
authorization, verifies the bound H5AD digest, deterministically regenerates
all 36 authorized calibration and pilot records from the source assay, and
requires canonical equality with the serialized reduction. This replay occurs
without reading a held matrix row.

All candidate configurations and classical residual variants are fitted using
the 12 calibration samples. Configuration selection minimizes sample-equal
deviance on the 24 pilot samples. The selected primary is then compared with
the pilot-selected strongest classical residual and destroyed-link control on
those same 24 samples. Pilot promotion requires, against each comparator,
at least 5% relative mean-deviance reduction, a paired 95% bootstrap upper
endpoint below zero using 20,000 shared sample resamples from NumPy
`default_rng(20260828)` and linear percentiles, and at least 19 of 24 samples
with lower primary loss.
This is an adaptive development gate, not confirmatory inference. Failure
closes the candidate without held margin or pairing access.

After a pilot pass, every selected configuration is refitted once, without
retuning, on all 36 calibration and pilot samples. The Oxford and St George's
held samples remain inaccessible to fitting and selection.

## Held-access order

Held processing is fail-closed and occurs once:

1. Validate the immutable source, protocol, code, designation, development
   authorization and result, and selected configurations. Require and validate
   the separate public
   `data/confirmation/combat_citeseq/held_rna_margin_authorization_v1.json`,
   then write a terminal held-prediction attempt record before the first held
   margin request.
2. Select the frozen 512 cells per held sample from metadata only.
3. Launch a separate permit-bound subprocess that scans the selected CSR row
   pointers and feature-index slices, decodes numeric `data` values only at the
   nine held RNA columns, emits only the 81 aggregate row margins per sample,
   closes the source, and exits. Because the deposited matrix is row-CSR, its
   structural index scan exposes nonzero column IDs across modalities inside the
   child process; the frozen code discards those positions and emits none of
   them. It decodes no held ADT numeric value and writes no cell-level count or
   state vector.
4. Set every held ADT column margin to the design-implied `[256, 256]`; prediction
   decodes no held ADT numeric value and forms no empirical held ADT state,
   margin, or RNA--ADT pairing.
5. Generate every method's complete held prediction from those margins,
   serialize and hash the predictions, publish the exact bytes at an immutable
   40-hex Git commit URL, and publish a separate authorization binding those
   bytes and all code, protocol, source, and model hashes.
6. Fetch the public prediction and authorization byte-for-byte and verify every
   binding. Write a separate terminal score-attempt record, then open RNA and
   ADT together once to form held truth tables and losses.

The prediction seal is therefore numeric and pairing-specific, not a claim that
the child process is information-theoretically blind to ADT sparsity support.
The attempt and prediction records report CSR row-pointer and feature-index
scans, RNA numeric values decoded, zero ADT numeric values decoded, and zero ADT
states, margins, or tables formed. Cell-level count or state vectors are never
serialized. Any earlier held
pairing, prediction written after truth access, failed binding, existing score
attempt, or attempted rerun of either phase is a terminal refusal.

## Confirmatory decision

For each comparison let `d` be primary sample loss minus comparator sample
loss. Relative reduction is one minus the ratio of sample-equal mean losses;
its comparator mean must be finite and strictly positive. Zero differences are
not favorable, and a nonfinite or undefined gate statistic cannot pass. Paired
95% intervals use 20,000 shared
sample bootstrap resamples from NumPy `default_rng(20260828)` and linear 2.5th
and 97.5th percentiles.

Each held panel must independently beat both the frozen strongest classical
residual and destroyed-link control by at least 5%, have a paired bootstrap
upper endpoint below zero, and meet its favorable-sample and exact sign-test
gates. The sign test treats a strictly negative primary-minus-comparator loss as
favorable, treats zero as nonfavorable, and uses a fixed panel size under the
null probability one-half:

- Oxford: at least 41 of 51 favorable samples and an exact one-sided binomial
  sign-test `p <= 0.025`;
- St George's: at least 9 of 10 favorable samples and an exact one-sided
  binomial sign-test `p <= 0.025`.

The relative-reduction and bootstrap gates concern the sample-equal mean loss.
The exact sign test concerns whether the probability of `d < 0` exceeds
one-half under its fixed-panel binomial null. All losses, contrasts, intervals,
tests, support counts, controls, exclusions, hashes, and any refusal are
reported separately by panel. Independence, unstructured ridge-only PM Haldane,
and label-permuted-graph results are mandatory reported ablations with paired
confidence intervals. A graph-specific superiority claim additionally requires
the primary-minus-frozen-unstructured paired 95% bootstrap upper endpoint to be
below zero in each held panel. That contrast does not add a 5% reduction,
favorable-count, or sign-test gate. Passing the residual and destroyed-link
gates in both panels establishes field transfer. Primary-method promotion also
requires graph-specific superiority in both panels; failure preserves the
field-transfer result but sets primary-method promotion to false. These are
candidate-specific confirmatory tests and are not a
familywise correction across earlier public candidate searches. The first
terminal score or refusal is final.

## Terminal execution record

This section records the outcome after the prospective specification above was
frozen. The authorized evaluator stopped during pilot candidate selection with
`RuntimeError: all matched Pearson/deviance comparators refused on pilot`.
This is the specified fixed-margin refusal, not an optimizer or normalization
failure. The raw and exact-null-centered Pearson variants each had 134
out-of-range sample--entity pairs spanning 23 of 24 pilot samples and 19 ordered
marker pairs. The corresponding deviance variants each had 102 violations
spanning 22 samples and 15 marker pairs. Two of eight primary configurations
were evaluable, so the primary family itself was not degenerate.

A deterministic post-failure diagnostic on the same authorized reduction,
without reopening a matrix, found that all three ridge-only Haldane candidates
also violated the unchanged no-clipping rule: ridge `0`, `0.01`, and `0.1` had
27, 24, and 13 out-of-range sample--entity pairs, respectively. This diagnostic
did not select a model, alter an estimator, or repair a gate. No matched
classical comparator or graph-specific ablation remained, so the pilot gate was
not reached. No held RNA margin, held ADT value, RNA--ADT pairing, or held truth
table was accessed. Held access is permanently closed, and the candidate may
not be rerun. The structured terminal record is
`results/development/combat_citeseq_pilot_terminal_refusal.json`.
