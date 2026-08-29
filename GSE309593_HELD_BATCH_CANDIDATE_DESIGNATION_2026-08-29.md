# GSE309593 held-batch confirmation candidate

## Question

Can RNA--protein association learned from early GSE309593 acquisition batches predict same-cell joint tables in later, subject-disjoint batches when only the later subjects' RNA and protein margins are supplied?

The 23 eligible samples are distinct, non-bridge, pretreatment bone-marrow samples from individuals with multiple myeloma. The deposited batch order fixes the split before numeric assay access: B092, B099, B110, and B129 form the 14-subject source cohort; B162, B208, and B210 form the nine-subject held cohort. Subjects and batches are disjoint.

## Analysis boundary

Source-only development will select the marker panel and fixed estimator configurations by leave-one-source-batch-out prediction. The primary family is hierarchical exact fixed-margin conditional estimation. Mandatory comparisons are signed Pearson and signed-root Poisson-deviance residual transfer, a stratified common-effect exact conditional fit, a donor-pooled saturated Poisson log-linear interaction, a margin-preserving destroyed-link fit, and independence.

The held files remain inaccessible until the source model and prediction procedure are public and immutable. Held RNA and ADT states will be constructed in separate stages. Predictions will be published from held margins before the two held modalities are joined. Every terminal refusal or scored result will enter the benchmark.

Transfer passes only if mean held deviance is at least 5% lower than both the selected residual and destroyed-link control, both paired-bootstrap upper endpoints for the loss difference are below zero, and at least eight of nine held subjects favor the primary over the residual. Incremental support beyond fitted classical interactions is a separate result: it requires lower point loss and an upper paired-bootstrap endpoint below zero against both the common-effect conditional and pooled saturated-Poisson estimates.

## Axis-only preflight

Before the executable protocol is frozen, a public preflight may read file hashes, H5 group names, RNA feature names and types, and ADT feature names or CSV headers. It may not read barcodes, identifiers, numeric matrix values, same-cell tables, associations, or losses. Axis information may remove unsupported features but cannot change the subject allocation.

One source-subject RNA feature axis and ADT header were inspected before this designation to confirm the deposited format. No barcode, identifier, numeric assay value, pairing, association, or loss was read. No held-subject file has been downloaded or opened.

## Prior GSE309593 branch

The earlier frozen GSE288020-to-GSE309593 protocol never reached target authorization because its source gate required five valid source methods and `external_study_ready=true`. It opened no GSE309593 assay file or identifier. This candidate is a new within-study batch-held analysis, not a repair or rerun of that unavailable cross-study protocol.
