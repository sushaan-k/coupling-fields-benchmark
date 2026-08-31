# GSE252762 celiac confirmation execution contract v2

This contract operationalizes the public sequence in the frozen candidate
protocol without changing the candidate, metadata preflight, or designation.
It adds an authorization and a public attempt before each protected matrix
stage. These extra barriers are a conservative extension of the frozen
protocol: they cannot alter a sample, marker, model, comparator, gate, or
outcome.

## Immutable public chain

Each item is a strict Git descendant of the preceding item. Every named tag is
annotated, pushed to the pinned public repository, remotely peeled, and checked
against the local artifact bytes before the next item is created.

| Order | Annotated tag | Required public artifacts |
| --- | --- | --- |
| 0 | `gse252762-celiac-v2-candidate` | frozen v2 protocol, designation, metadata preflight |
| 1 | `gse252762-celiac-v2-implementation` | runner, numerical core, matched ridge-Poisson implementation, tests, complete transitive code bindings |
| 2 | `gse252762-celiac-v2-source-authorization` | source authorization |
| 3 | `gse252762-celiac-v2-source-attempt` | source attempt |
| 4 | `gse252762-celiac-v2-source-consumption` | source attempt and one-use source consumption lease |
| 5 | `gse252762-celiac-v2-source-result` | source consumption, access journal, selected-count sidecars, terminal source result, and completed reduced tables when reduction succeeds |
| 6 | `gse252762-celiac-v2-held-rna-authorization` | held-RNA authorization |
| 7 | `gse252762-celiac-v2-prediction-attempt` | prediction attempt |
| 8 | `gse252762-celiac-v2-prediction-consumption` | prediction attempt and one-use held-RNA consumption lease |
| 9 | `gse252762-celiac-v2-predictions` | prediction consumption, RNA access journal, selected-count sidecar, frozen predictions |
| 10 | `gse252762-celiac-v2-score-authorization` | held-CITE score authorization |
| 11 | `gse252762-celiac-v2-score-attempt` | score attempt |
| 12 | `gse252762-celiac-v2-score-consumption` | score attempt and one-use held-CITE consumption lease |
| 13 | `gse252762-celiac-v2-held-result` | score consumption, CITE access journal, selected-count sidecar, terminal held result |

The peeled candidate commit is
`fd84891c9c4be03e7faeeffd09838a98f2f1bda1`. A stage execution cannot begin
until its public attempt and consumption tags have been verified. The public
consumption record permanently occupies the designated stage. The runner uses
an exclusive same-filesystem lock, checks the public lease before every GET,
and converts an interrupted execution into a terminal result instead of
issuing another request.

For each stage, the operator runs `authorize-STAGE`, publishes its authorization
tag, runs `claim-STAGE`, publishes the attempt tag, runs `consume-STAGE`, and
publishes the consumption tag before invoking `run-STAGE`. A result or terminal
failure is then committed and published under the stage result tag.

The matrices are public and remain accessible outside this runner. The chain
certifies the designated confirmatory execution; it is not an access-control
claim about independent third parties or separate clones.

## Matrix access budget

The source stage permits one streaming GET for each RNA and CITE matrix in
batches 1 through 5: ten GETs in total. It permits no batch-6 access. The
prediction stage permits one streaming GET for batch-6 RNA and no CITE access.
The score stage permits one streaming GET for batch-6 CITE and no RNA access.
Redirects and automatic retries are disabled.

Each successful reduction records the complete compressed and decompressed
SHA-256 digests, byte counts, Matrix Market dimensions and entry counts,
selected row axis, selected column axis, selected-block digest, counter
algebra, and stream-exhaustion certificates. Before a successful GET is marked
finished, the runner atomically publishes an immutable sidecar containing its
selected marker counts and a digest of the preceding journal snapshot. The
aggregate checkpoint then binds those sidecars. Recovery after a completed
reduction replays the immutable sidecars instead of requesting a matrix again.

## Successor checks

The held-RNA authorization requires a public `SOURCE_PASS`. The runner
recomputes calibration selection and the pilot decision from the public
reduced source tables before it can create the prediction attempt. Prediction
refits the promoted estimators on all 16 source donors.

The calibration-frozen strongest benchmark is selected from independence,
donor-stratified ridge Poisson, bias-reduced context Poisson, and context signed
deviance in that order when mean losses tie. The donor-stratified ridge-Poisson
comparator is always finite and carries donor-specific row and column nuisance
effects, a context-by-entity interaction, and the frozen `0.01` interaction
penalty. Numeric transport and primary-grid ties use the listed numeric order
in the candidate protocol.

The score authorization and score attempt require a public prediction
artifact. Before score consumption, the runner again recomputes the source
selection and all-source refit, reconstructs every held prediction from the
published RNA states and margins, and checks the complete sample and method
axes. Only then can it request the batch-6 CITE matrix.

The paired bootstrap is the 2.5th--97.5th percentile interval computed by
`numpy.quantile(..., method="linear")`. Each draw resamples donors with
replacement within ACD, GFD, and control at the original stratum sizes. Every
comparator uses the same seeded resample indices in frozen preflight donor
order. The exact one-sided sign test discards differences whose absolute value
is at most `1e-12`. A bootstrap upper endpoint equal to zero fails the strict
below-zero gate; a sign-test probability equal to `0.05` passes its at-most
`0.05` gate. The held gate applies its confirmatory interval and sign criteria
to independence, donor-stratified ridge Poisson, and the calibration-frozen
strongest benchmark, and separately requires improvement over destroyed links
and every classical method.

No failed source gate, execution failure, malformed predecessor, or changed
implementation binding can authorize the next protected stage.
