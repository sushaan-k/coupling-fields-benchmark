# GSE313642 HCC PBMC CITE-seq held-patient protocol v2

## Confirmatory question and status

This one-shot experiment tests whether an A/B-context-conditioned exact
conditional coupling field learned from 22 hepatocellular-carcinoma (HCC)
patients predicts RNA--surface-protein dependence in 12 patient-disjoint HCC
samples better than matched classical interaction estimators. The physical
patient is the inferential unit. The recipient margins are always identical
across methods.

`data/confirmation/gse313642_hcc/candidate_designation_v2.json` fixes the
cohort, sample choice, split, markers, file-address rule, and access firewall.
Its current status is `METADATA_ONLY_DESIGNATED_OUTCOME_DISABLED`: no matrix
may be requested until the protocol, runner, tests, manifest, and authorization
chain have been committed, tagged, pushed, and verified on the public remote.

## Metadata boundary and deposited files

GEO reports 194 sample records comprising 97 exact GEX/feature-barcoding
pairs. The family SOFT gzip is 20,765 bytes with SHA-256
`f3d1a3d6dd0fcf707456b205619659c18d6d0017cf3532e4789d99209b96c2db`.
The 43,340-byte official file list has SHA-256
`8fb88cd52c30eb91a8da21caff1352c1307a47dc7ac0ddcc3fdd9960c9141aa9`.
These metadata establish 35 HCC patients and three healthy donors represented
by 21 additional pairs.

The public v1 candidate opened all 140 feature and barcode gzip payloads before
any matrix request. Its mandatory axis gate refused A-33 because replicate 1
contained three duplicated barcode identifiers in each modality at different
row positions. The refusal, complete access journal, and zero-matrix-access
boundary are fixed by tag `gse313642-hcc-v1-terminal-axis-refusal`. V2 excludes
only A-33; calibration and held assignments are unchanged. The 136 retained
axes had already been opened under v1 and are reused without a second GET. No
Matrix Market header or body was requested before v2 designation.

The 2,043,904,000-byte series archive `GSE313642_RAW.tar` is forbidden because
requesting it would expose source and held matrices together. Acquisition uses
only the individual sample supplementary URLs named by the designation. A
local input root contains the exact gzip filenames without renaming; each file
is checked against the bound GEO file list before parsing.

`data/confirmation/gse313642_hcc/source_manifest_v2.json` binds the exact
filename and official byte count for all 204 selected individual members: 136
feature/barcode axes, 22 calibration matrices, 22 post-freeze pilot matrices,
12 held-GEX prediction matrices, and 12 post-prediction held-FB matrices. Its
allowed-stage field is normative. Each retained axis is marked reuse-only with
`v2_get_authorized=false`; only the 68 unopened matrices can receive a v2 GET,
once and only in their assigned stage.

## Patient and sample selection

Only HCC patients enter development or confirmation. Within a patient, select
the earliest deposited biological timepoint. The order is baseline, numeric
`C<cycle>D<day>`, Safety follow up, then Long term follow up. Ties use the
lexically smallest exact GEX/FB pair stem. V2 retains the v1 choice for every
patient except A-33, which is excluded solely because its paired barcode axes
failed the public matrix-free uniqueness gate. This leaves 34 patients.

Within A and B separately, patients are ranked by ascending

```text
SHA256(UTF8("GSE313642-HCC-PBMC-CITESEQ-v1|donor|2026-08-30" NUL deposited_patient_id))
```

with deposited IDs retaining the hyphen. The v1 allocation assigned the first
five A patients and first seven B patients to held, the next five A and seven B
patients to pilot, and all others to calibration. V2 removes A33 from its pilot
stratum without promoting or reassigning another patient.

| Role | A patients | B patients |
|---|---|---|
| Calibration | A30, A12, A08, A32, A34 | B15, B23, B17, B04, B08, B03 |
| Pilot | A04, A03, A35, A31 | B02, B07, B06, B05, B21, B01, B12 |
| Held | A05, A02, A07, A21, A36 | B11, B14, B13, B22, B18, B09, B10 |

The 11 calibration patients determine every configuration. The 11 pilot
patients are a numeric-outcome-sealed source-promotion set. The 12 held
patients remain numeric-outcome-sealed until source promotion and frozen held
predictions are public.

## Mandatory all-patient axis gate

Before any matrix request, revalidate the v1 journal, gzip hashes, and fully
decoded feature and barcode axes for all 34 retained GEX/FB pairs. For every
pair:

1. each barcode axis must be unique and contain at least 512 barcodes;
2. the GEX and FB barcode sets must be exactly equal without normalization;
3. every feature row must have exactly three TSV columns; each of `CD4`,
   `CD7`, `CD14`, `CD19`, `CD33`, `CD38`, `CD44`, `CD47`, and `CD52` must
   occur exactly once in GEX column 2 with column-3 type `Gene Expression`,
   and exactly once in FB column 1 with column-3 type `Antibody Capture`;
   the selected FB column-2 reagent descriptions must equal the nine values
   bound in the designation;
4. the preflight must record the ordered-axis and set hashes, observed byte
   counts, gzip hashes, and all individual URLs.

Any mismatch is a terminal refusal before the first matrix byte. The axis
preflight cannot add, replace, normalize, or reorder a marker.

At each later authorized numeric stage, the first read of a Matrix Market file
checks its header dimensions against the frozen feature and barcode lengths
before parsing any count entry. A header mismatch terminates that stage; it is
not represented as a pre-matrix axis check.

## Cells, states, and tables

For patient `d`, order the exact common barcode set by

```text
SHA256(UTF8("GSE313642-HCC-CELL-v1" NUL deposited_patient_id NUL barcode))
```

and then by barcode. Retain the first 512. Counts, library size, cell label,
treatment, and outcome never enter cell selection.

For RNA marker `g`, state one is raw GEX UMI count greater than zero. For FB
marker `p`, sort the 512 raw counts by descending count, then

```text
SHA256(UTF8("GSE313642-HCC-FB-TIE-v1" NUL deposited_patient_id NUL p NUL barcode))
```

and barcode; the first 256 cells are high and the remaining 256 are low.
Every recipient FB margin is therefore fixed at 256/256 before held FB values
are opened.

The frozen marker order is CD4, CD7, CD14, CD19, CD33, CD38, CD44, CD47,
CD52. Each patient yields 81 ordered RNA-marker by FB-marker 2-by-2 tables,
with rows RNA-negative/RNA-positive and columns FB-low/FB-high. Positive
coupling is a positive full log odds ratio. Each method is evaluated at the
same observed recipient RNA margin and fixed FB margin.

Patient loss is the mean multinomial deviance per retained cell over all 81
tables. A fixed-margin table with singleton support contributes zero and no
interaction information. Patients receive equal weight; cells and tables are
not inferential replicates.

## Primary structured conditional field

The primary model is `fit_structured_context_conditional_log_odds`. For each
patient and marker pair, it fits the exact fixed-margin noncentral-
hypergeometric likelihood with a patient deviation around an A- or B-specific
population log odds. The context matrix has two symmetric one-hot columns:
A is `(1,0)` and B is `(0,1)`. Both coefficients receive ridge `0.01`; an
intercept-plus-indicator parameterization is forbidden because it would
penalize the two cohorts asymmetrically.

Eleven leave-one-patient-out calibration folds select among:

| Parameter | Frozen values |
|---|---|
| Patient-deviation penalty, eta | 0.1, 1.0 |
| Context-coefficient ridge | 0.01 for A and 0.01 for B |
| Graph penalty, gamma | 0.0 |
| Transport multiplier, alpha | 0.75, 1.0 |

The graph Laplacian is the zero matrix. No graph, hypergraph, or neighborhood
information enters this confirmation. In each fold, fit the ten training
patients, multiply the held patient's group-specific population log odds by
alpha, and reconstruct the exact noncentral-hypergeometric expected table at
that patient's margins. Select the finite configuration with the lowest
patient-equal mean fold loss; exact ties use `(eta, ridge, gamma, alpha)` order.

## Matched classical comparators

Every comparator uses the same calibration patients, A/B labels, marker pairs,
recipient margins, patient weighting, and deviance. Calibration-only
leave-one-patient-out loss independently selects alpha from `{0.75, 1.0}` for
the A/B Poisson and signed-deviance transfers. Exact CMLE is the unscaled
unpenalized estimate, independence has no fitted coordinate, and destroyed
links use the intact primary configuration. No held or pilot truth selects a
comparator.

1. **A/B-profiled Poisson interaction.** Call
   `mapreg.poisson_loglinear.fit_poisson_loglinear_interaction` with the
   two cohort labels. This is a standard row-plus-column-plus-interaction
   Poisson log-linear model with patient/table row and column nuisance terms
   profiled out and one unpenalized interaction per cohort and marker pair.
   Reconstruct with `mapreg.poisson_loglinear.reconstruct_poisson_tables` at
   the exact recipient margins. A pooled, context-free fit is reported only as
   an additional diagnostic.
2. **A/B signed-deviance residual transfer.** Compute signed root Poisson-
   deviance residual coordinates under independence, normalize each patient
   to the common 512-cell scale, pool with equal patient weight separately in
   A and B, multiply by the calibration-selected alpha, and invert at the exact
   recipient margins.
3. **A/B exact common-effect CMLE.** Fit one unpenalized exact conditional
   common log odds for each cohort and marker pair and reconstruct at recipient
   margins. Report it when every required fold and final refit is finite;
   otherwise record the boundary refusal without substitution.
4. **Fixed-margin independence.** Set log odds to zero at the recipient
   margins.
5. **Destroyed links.** Within each training patient, order retained cells by
   SHA-256 of `GSE313642-HCC-DESTROY-v1 NUL patient NUL barcode`, then by
   barcode, and cyclically shift each complete nine-FB state vector by 256
   positions. This preserves every RNA and FB margin and all within-FB
   dependence while breaking the RNA--FB pairing. Fit the primary model with
   the intact-data-selected configuration. Validation, pilot, and held truth
   are never destroyed.

The A/B-profiled Poisson, A/B residual, and destroyed-link methods are the
mandatory head-to-head comparators. Exact CMLE and independence are always
reported when defined but do not replace a mandatory comparator. A mandatory
Poisson boundary or numerical refusal is a terminal unevaluable result, not a
reason to switch to its easier pooled diagnostic.

## Source promotion

Calibration and pilot are different commands with different irreversible
claims. The calibration command alone reads the 11 calibration patients,
selects the primary and A/B Poisson/residual transport configurations, fits the
exact CMLE, and writes
`results/development/gse313642_hcc_calibration_selection_v2.json`. That
artifact contains the exact reduced calibration tables and destroyed-link
tables required for the later one-shot 22-patient refit, their hashes and
reduction audits, every calibration LOPO loss, and the frozen fitted fields.
It contains no pilot value.

Before the first pilot GET, the calibration artifact and its complete binding
set are committed, annotated-tagged as `gse313642-hcc-v2-calibration`, pushed,
and verified byte-for-byte at the public remote. The pilot command verifies
that remote tag and its ancestry, then reads the 11 pilot patients once. It
predicts and scores them from the tagged calibration models without refitting
or reselection. Only after the gate is evaluated may it combine the frozen
calibration reductions with the in-memory pilot reductions for the single
22-patient source refit.

Against each mandatory comparator, primary promotion requires all three:

1. strictly lower patient-equal mean deviance; and
2. strictly lower deviance in at least 8 of 11 pilot patients; and
3. strictly lower mean deviance separately among the four A patients and the
   seven B patients.

Failure against any mandatory comparator terminates the candidate and forbids
held matrix access. On a pass, the source result and refitted models are
committed, annotated-tagged as `gse313642-hcc-v2-source`, pushed, and remotely
verified before the first held GEX request. No held value enters selection or
refitting.

## Held prediction and confirmation

After the public source tag and a separately published held-GEX authorization,
the prediction command reads only the 12 held GEX matrices. It combines their
nine RNA margins with the predetermined 256/256 FB margins and writes every
method's 81 expected tables, selected configurations, coordinate order, and
hashes. The prediction artifact must state that zero held FB numerical values
and zero held RNA--FB pairings were read. It is committed, annotated-tagged as
`gse313642-hcc-v2-predictions`, pushed, and remotely verified before score
authorization.

The score authorization binds the remotely verified prediction bytes and is
itself committed, annotated-tagged as
`gse313642-hcc-v2-score-authorization`, pushed, and remotely verified. Only
then may the score command read each held FB matrix once, construct the frozen
states and truth tables, and score the 12 patients. For
primary-minus-comparator patient losses, each mandatory comparison must
satisfy all five conditions:

1. at least 5% lower patient-equal mean deviance;
2. the upper endpoint of a 20,000-draw A/B-stratified paired-patient bootstrap
   95% interval for the mean difference is below zero;
3. lower primary loss in at least 10 of 12 patients; and
4. a one-sided exact sign-test probability no greater than 0.025; and
5. a strictly negative mean loss difference separately among the five A
   patients and the seven B patients.

Bootstrap draws use `numpy.random.default_rng(31364201)`, resample five A and
seven B patients with replacement within cohort, share one draw-index tensor
across methods, and use linear 2.5% and 97.5% quantiles. The sign test discards
exact zero differences, counts strictly negative primary-minus-comparator
differences as favorable, and evaluates the exact binomial upper tail under
probability one half. The separate 10-of-12 rule still treats ties as not
favorable.

The first terminal held result is retained whether it passes, fails, or is
unevaluable. No rerun, alternate split, marker substitution, comparator
relaxation, or threshold revision is permitted.

## Healthy donors and interpretation boundary

HC1, HC2, and HC3 are outside model selection and inference. Their 21 paired
samples may be summarized only after the terminal HCC result and cannot alter
any fit, gate, or claim.

This experiment tests context-conditioned interaction transfer across HCC
patients. It does not test a graph penalty because gamma is fixed at zero, and
it does not identify treatment or longitudinal effects because one
metadata-selected sample represents each patient. The source-pilot gate is a
screening rule without an inferential interval; a passing confirmatory claim
rests on the numeric-outcome-sealed held panel. With no exact loss ties, the 10-of-12 favorable
rule already implies a one-sided sign-test probability of 79/4096; retaining
both conditions makes the decision contract explicit.

## Access and artifact order

The intended executable and artifact paths are:

| Stage | Path |
|---|---|
| Candidate | `data/confirmation/gse313642_hcc/candidate_designation_v2.json` |
| Source manifest | `data/confirmation/gse313642_hcc/source_manifest_v2.json` |
| Runner | `experiments/confirm_gse313642_hcc.py` |
| Numerical core | `experiments/gse313642_hcc_core.py` |
| Focused tests | `tests/test_gse313642_hcc_confirmation.py` |
| Axis preflight | `results/development/gse313642_hcc_axis_preflight_v2.json` |
| Calibration selection and reductions | `results/development/gse313642_hcc_calibration_selection_v2.json` |
| Pilot/source result and 22-patient refit | `results/development/gse313642_hcc_source_v2.json` |
| Held predictions | `results/gse313642_hcc_predictions_v2.json` |
| Held result | `results/gse313642_hcc_confirmation_v2.json` |
| Stage attempts and authorizations | `data/confirmation/gse313642_hcc/` |

The irreversible order begins with public v2 candidate, implementation, and
axis-preflight tags. The preflight tag authorizes calibration. The calibration
claim is committed and published as `gse313642-hcc-v2-calibration-attempt`
before the first calibration GET; its result is published as
`gse313642-hcc-v2-calibration`.

The calibration result authorizes `pilot_authorization_v2.json`, published as
`gse313642-hcc-v2-pilot-authorization`. The separate pilot claim is then
published as `gse313642-hcc-v2-source-attempt` before the first pilot GET; the
gate and 22-patient refit are published as `gse313642-hcc-v2-source`.

The source result authorizes `prediction_authorization_v2.json`, published as
`gse313642-hcc-v2-prediction-authorization`. The prediction claim is published
as `gse313642-hcc-v2-prediction-attempt` before the first held-GEX GET; frozen
predictions are published as `gse313642-hcc-v2-predictions`. Those bytes bind
`score_authorization_v2.json`, published as
`gse313642-hcc-v2-score-authorization`. The score claim is published as
`gse313642-hcc-v2-score-attempt` before the first held-FB GET, and the terminal
result is published as `gse313642-hcc-v2-score`.

Each authorization binds the exact public bytes and SHA-256 values of every
upstream artifact, runner, test, and transitive estimator module. Every tag is
annotated, pushed, and checked against the immutable public remote before the
next capability is consumed. Attempts are exclusive, and any exception after
claiming a stage writes the terminal record before exit.
