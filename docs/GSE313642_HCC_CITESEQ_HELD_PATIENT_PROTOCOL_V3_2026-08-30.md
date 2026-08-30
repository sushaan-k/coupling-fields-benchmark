# GSE313642 HCC PBMC CITE-seq held-patient protocol v3

## Status and question

This campaign tests whether an A/B-conditioned exact conditional coupling
field learned from 21 HCC patients predicts RNA--surface-protein dependence in
12 patient-disjoint HCC samples better than matched classical interaction
estimators. The physical patient is the inferential unit, and every method is
evaluated at identical recipient margins.

V3 is a new pre-outcome frozen public-data holdout after two immutable terminal
campaigns. V1 refused A33 at the barcode-axis gate before matrix access. V2
stopped during the first calibration patient because the frozen parser refused
the deposited `real general` Matrix Market banner. A post-terminal A30-only
debug access reproduced that parser failure on byte-identical files. Its
retrospective disclosure is public at tag
`gse313642-hcc-v2-postterminal-debug-disclosure`; it records that the access was
not authorized or journaled in advance. A30 is exposed development data and is
excluded from every v3 fit, gate, prediction, score, and inferential count.
No pilot or held matrix had been opened when v3 was designated.

The v2 protocol at SHA-256
`a185a1f92585f0dcb26e31be31cbfdd718b381afb3cdf14dad11e442bec7fd7e`
remains normative except for the explicit changes below. This reference fixes
the sample-selection rule, split salt, markers, cell selection, state
definitions, estimators, hyperparameter grid, comparators, loss, gates,
bootstrap, and access barriers without restating them differently.

## Frozen v3 changes

1. Exclude A30 without replacement. Calibration contains A12, A08, A32, A34,
   B15, B23, B17, B04, B08, and B03. The unchanged pilot contains A04, A03,
   A35, A31, B02, B07, B06, B05, B21, B01, and B12. The unchanged held panel
   contains A05, A02, A07, A21, A36, B11, B14, B13, B22, B18, B09, and B10.
2. Select configurations by 10-fold leave-one-patient-out calibration. Each
   fold trains on nine patients. A promoted source refit combines the 10
   calibration and 11 pilot patients, for 21 total.
3. Permit the exact Matrix Market banners `integer general` and `real general`.
   Integer-field parsing is unchanged. Real-field parsing is opt-in for this
   campaign and accepts a stored value only when its ASCII decimal value is
   unsigned, finite, mathematically integral, and no greater than the int64
   maximum. The check covers the entire stream, including unselected entries;
   binary floating-point rounding is forbidden. The observed banner is retained
   in every reduction audit.
4. Journal `MATRIX_PARSE_STARTED`, `MATRIX_PARSE_FINISHED`, and
   `MATRIX_PARSE_FAILED` for each acquired matrix. A failure records modality,
   patient, exception class, stable refusal code, message, and available parser
   partial audit before deletion. The public terminal result uses the stable
   refusal code and binds the detailed append-only journal.
5. Use v3-only artifact paths, capabilities, annotated tags, and scratch
   identity. No v1 or v2 terminal artifact is overwritten or rerun.

No biological estimator, comparator, marker, hyperparameter value, prediction
rule, pilot threshold, held threshold, seed, or held identity changes in v3.

## Bound cohort and files

`candidate_designation_v3.json` binds the v2 designation and its public
failure history. `source_manifest_v3.json` is a deterministic overlay on the
204-record v2 manifest: remove exactly the four A30 axis records and two A30
matrix records, preserving every other URL, filename, role, modality, and byte
count. V3 therefore binds 198 records: 132 already accessed axes, 20
calibration matrices, 22 pilot matrices, 12 held GEX matrices, and 12 held FB
matrices. The series archive remains forbidden. No new axis GET is permitted.

The all-patient preflight revalidates the full historical v1 axis journal and
the 132 locally retained axes for the 33 active patients. It must reproduce the
v2 feature, marker, barcode, and gzip evidence for those patients exactly and
record zero v3 axis GETs. A mismatch is terminal before numeric access.

## Estimators and calibration

Each patient contributes the same 81 ordered 2-by-2 RNA-marker by FB-marker
tables defined in v2. The primary estimator remains the A/B-conditioned exact
conditional field with context ridge 0.01, graph penalty zero, patient
deviation penalty in `{0.1, 1.0}`, and transport multiplier in `{0.75, 1.0}`.
Configuration selection minimizes patient-equal mean held-fold multinomial
deviance across the 10 calibration folds.

The mandatory comparators remain:

1. A/B-profiled Poisson log-linear interaction, with its multiplier selected
   on the same calibration folds;
2. A/B signed-root Poisson-deviance residual transfer, with its multiplier
   selected on the same folds; and
3. destroyed RNA--FB links fitted with the intact primary configuration.

A/B exact common-effect CMLE, fixed-margin independence, and pooled Poisson are
reported diagnostics. Every method receives the same source patients, cohort
labels, marker pairs, target margins, patient weights, and loss. An undefined
mandatory comparator terminates the campaign.

## Pilot promotion

The calibration command reads only the 10 calibration patients, records every
reduced table and audit, selects all configurations, and publishes the result
before pilot authorization. The pilot command reads the unchanged 11-patient
panel once and predicts it without selection or refitting. Against each
mandatory comparator, promotion requires:

1. lower primary patient-equal mean deviance;
2. lower primary deviance in at least 8 of 11 patients; and
3. lower primary mean deviance separately in the four A and seven B patients.

Only a pass permits the one-time 21-patient source refit and held-GEX
authorization. A failure is terminal and held matrices remain unopened.

## Held decision

After public source promotion, the prediction stage opens only the 12 held GEX
matrices and publishes all expected tables before any held FB access. A
separately published score authorization binds those prediction bytes. The
score stage then opens the 12 held FB matrices once.

For primary-minus-comparator patient losses, every mandatory comparison must
satisfy all five v2 criteria:

1. at least 5% lower patient-equal mean deviance;
2. upper endpoint below zero for the 20,000-draw A/B-stratified paired-patient
   bootstrap 95% interval;
3. lower primary loss in at least 10 of 12 patients;
4. one-sided exact sign-test probability at most 0.025; and
5. negative mean loss difference separately among five A and seven B patients.

The bootstrap seed remains 31364201. The first terminal result is retained
whether it passes, fails, or is unevaluable. No rerun, split change, marker
substitution, comparator relaxation, threshold revision, or post-score model
choice is permitted.

## Public order

The irreversible public sequence is:

1. `gse313642-hcc-v3-candidate`;
2. `gse313642-hcc-v3-implementation`;
3. `gse313642-hcc-v3-axis-preflight`;
4. `gse313642-hcc-v3-calibration-attempt`, then terminal calibration tag;
5. pilot authorization and `gse313642-hcc-v3-source-attempt`, then terminal
   source tag;
6. held-GEX authorization and prediction attempt, then frozen predictions;
7. held-FB score authorization and score attempt, then the terminal held tag.

Every authorization and attempt is committed, annotated-tagged, pushed, and
verified against the public remote before the associated capability is
consumed. Each artifact binds its upstream commits and SHA-256 values. V3 is a
publicly time-stamped pre-outcome holdout, not a registration in an external
preregistration registry.
