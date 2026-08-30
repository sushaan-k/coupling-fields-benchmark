# GSE214546 TEA-seq held-donor confirmation protocol v1

## Question and estimand

This one-way experiment tests whether an age-conditioned RNA--protein coupling field learned from eight linked source donors predicts joint RNA--protein tables in eight disjoint recipient donors when only recipient margins and age group are available. The physical donor is the inferential unit. The target loss is donor-equal mean multinomial deviance per cell across the frozen ordered marker-pair axis.

## Frozen split and access boundary

`data/confirmation/gse214546_teaseq/candidate_designation_v1.json` fixes the 8-source/8-held split, 32 candidate cognate markers, file names, public byte counts, and exclusions. Each arm contains four adult and four pediatric donors and has the same batch counts: two B065, two B069, and four B076 donors. The three IMM19 libraries are technical controls from one donor and are excluded.

Before this protocol was published, no GSE214546 assay matrix was downloaded or opened. GEO metadata, HTTP headers, the published ADT barcode reference, and filtered-metadata CSV headers were read. A source H5 file may be acquired before implementation freeze only to inventory HDF5 object names, shapes, string axes, feature types, and compression metadata. Reading any count, index, or sparse-matrix value dataset during that inventory is prohibited and must be recorded as a terminal violation.

Held H5 files remain physically unopened through source fitting and promotion. No failed or completed campaign may be rerun under this protocol.

## Cells, states, and coordinate axis

For each donor, eligible singlets are identified from the official filtered-metadata CSV. The unique metadata barcode column that exactly matches the H5 cell axis is bound during the nonnumeric schema inventory. The 512 eligible cells with the smallest SHA-256 values of `GSE214546-CELL-v1|GSM|barcode` are retained. Fewer than 512 matched singlets is a terminal support refusal.

RNA state is raw count greater than zero. For each protein, exactly 256 retained cells are assigned the high state by raw ADT rank; SHA-256 of `GSE214546-ADT-TIE-v1|GSM|protein|barcode` breaks ties. Complete ADT profiles are cyclically shifted along the ordering induced by `GSE214546-DESTROY-v1|GSM|barcode` to form the destroyed-link control.

Candidate markers resolve by a unique exact RNA symbol and by a unique exact ADT description or frozen ADT barcode. A marker is retained only if its RNA positive count lies from 4 through 508 in every source donor. At least 20 markers must remain. The primary axis is the complete ordered RNA-by-protein cross-product of the retained markers and is frozen after source reduction.

## Primary estimator

For each ordered pair, source donors follow an exact fixed-margin conditional model. Donor log odds equal a linear age-group field plus a penalized donor deviation:

`theta[d,e] = beta[adult,e] + I[pediatric,d] * beta[age,e] + delta[d,e]`.

The estimator minimizes exact conditional negative log likelihood plus quadratic penalties on donor deviations and coefficients. Positive penalties make every finite coordinate fit unique. The recipient interaction is the fitted adult field or the fitted adult-plus-age field, multiplied by a transport factor, and is reconstructed as the exact noncentral-hypergeometric expected table at recipient margins.

Leave-one-source-donor-out cross-validation selects donor-deviation penalty `{0.1, 1, 10}`, age-coefficient penalty `{0.1, 1, 10}`, and transport factor `{0.75, 1.0}`. The intercept ridge is fixed at `0.01`; graph penalty is fixed at zero. The selected configuration minimizes donor-equal mean deviance, with lexicographic parameter order breaking exact ties. Every fold and the final source fit must satisfy the frozen gradient and condition-number certificates.

## Comparators

All methods receive the same source tables, source-selected coordinate axis, held margins, age labels, and 512-cell samples.

1. **Pooled fixed-interaction Poisson:** a standard saturated source Poisson log-linear interaction pooled across donors; recipient row and column parameters are profiled at the transported interaction. Source cross-validation selects transport `{0.75, 1.0, 1.25}`.
2. **Age-stratified fixed-interaction Poisson:** the same classical estimator fitted separately within adult and pediatric source donors. This is a strong reported sensitivity comparison and is not part of the confirmation gate.
3. **Destroyed links:** the selected primary estimator refitted after the deterministic within-donor profile shift.
4. **Common-effect exact conditional, signed-root deviance transfer, and fixed-margin independence:** reported secondary controls.

## Source promotion

Held access remains disabled unless every source fit completes, at least 20 markers survive, and leave-one-donor-out mean deviance for the primary estimator is below both pooled fixed-interaction Poisson and destroyed links, with at least six of eight source donors favorable against each. Failure is terminal and produces no held prediction or score.

## Prediction firewall

After source promotion is public, each held file is reduced once in an RNA-margin stage. That stage may read only the frozen RNA feature rows for the selected cells; ADT value datasets are forbidden. Protein high-state margins are fixed at 256/256 by definition. The source-selected models, held RNA margins, context-specific predictions for every method, coordinate axis, and their SHA-256 values are then published.

Only a later public score authorization may permit a second sequential acquisition of each held file. The scorer reads the frozen RNA and ADT rows, forms truth tables, scores the already published predictions, and deletes the file before requesting the next donor.

## Confirmation decision

The primary estimator passes only if, against both pooled fixed-interaction Poisson and destroyed links:

- donor-equal mean deviance is at least 5% lower;
- the upper endpoint of a 20,000-draw paired-donor bootstrap 95% interval for primary-minus-comparator deviance is below zero;
- at least seven of eight donors favor the primary estimator; and
- mean improvement is positive within both adult and pediatric held strata.

The exact one-sided sign probability for seven or more favorable donors is reported. Bootstrap seed `21454601` and all deterministic salts are fixed above. Results against age-stratified Poisson and the remaining controls are reported regardless of direction and do not alter the decision.

## Terminality and reporting

Schema, support, numerical, source-promotion, prediction, and score failures are distinct terminal outcomes. A terminal artifact records accessed files, byte and SHA-256 checks, dataset slices read, deleted scratch files, and whether held access occurred. The aggregate benchmark table includes the attempt whether it passes, fails, or stops before scoring.
