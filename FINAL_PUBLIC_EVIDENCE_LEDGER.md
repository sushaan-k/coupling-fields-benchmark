# Final public evidence ledger

**Snapshot:** 28 August 2026

**Machine-readable table:** `results/final_public_benchmark_table.tsv`

## Bottom line

The completed public evidence supports a conditional association-field estimand,
its fixed-margin null centering, and explicit refusal. It does not yet
support superiority of the structured estimator. PerturbSci-Kinetics contains a
clear held-guide pairing signal, but the fixed primary has higher error than
endpoint ridge and the endpoint-plus-residual fit. ReSisTrace confirms that lineage links carry association in all
eight arms, but none of the three treatment-minus-control contrasts has a
positive lineage-bootstrap interval across the two deposited cultures. The
three RNA-protein/RNA-ATAC development panels and PerturbFate refuse. In the
Arce held-donor RNA-protein confirmation, correlation was positive, but the
fixed primary lost to linear cross-covariance on RMSE and did not separate from
destroyed links under the locked gate. All seven outcome-scored public panels are complete;
none establishes estimator superiority. A separately frozen PoKI-seq held-donor
candidate stopped at its state-occupancy preflight for `Donor1:Stim:41BB`.
No prediction or scored result was written, so this eighth benchmark entry is a
support refusal rather than outcome-scored evidence. The later Lawlor candidate
stopped when its frozen reducer rejected the deposited ADT object type; Hao
stopped when fewer than 12 cognate markers passed its frozen marginal-support
rule. Neither run formed a held joint-table score. The separate Kotliarov
held-batch candidate was frozen publicly before outcome access: ten batch-1
development donors, nine disjoint batch-2 held donors, and donor 209 excluded
from both batches. Fresh-clone verification passed at commit
`a034fd272ef631d70f39debc467570568ef8754a`. Its one authorized preparation
stopped because fewer than four prespecified RNA-only lineages met the frozen
minimum of 50 retained cells in every one of 19 donors. The ADT file was read
only as an opaque byte stream for integrity verification; its HDF5 count
dataset was never opened. No held RNA-ADT pairing, joint table, prediction,
score, or performance estimate was formed, and the candidate was not rerun.
The subsequent BMMC candidate ended in a terminal numerical development
refusal after three recorded attempts; no prediction or held score was formed,
and all six held-donor count slices remained unopened. The subsequent GSE279451
plan fixed 19 development and 21 held physical donors, nine RNA--ADT markers,
81 ordered interactions, a donor-level gate, and a direct comparison with the
strongest matched signed Pearson or Poisson-deviance residual transfer. It is
now terminal. The plan was verified from a fresh clone at commit
`f63c9dc760a85a1361ce75e13036eb23262b1bc7` and is published under tag
`gse279451-sepsis-v1-protocol` before development count access. One authorized
attempt then acquired and reduced the 19 development matrices. The terminal
evaluation refused because `common_effect_graph` and
`common_effect_ridge_only` were unavailable, so the complete gate and its
classical-residual comparison produced no decision. No prediction,
authorization, held pairing, or held score was formed. All 21 held-donor matrix
members remained unopened, and the candidate was not rerun.

GSE299043 was the outcome-disabled successor. Its immutable plan reserved ten
Cambridge development donors and ten donor-disjoint LiveOnNY/Columbia held
donors from mesenteric lymph node, with nine exact RNA--ADT markers and 81
ordered interactions. The one terminal development attempt completed 21 member
reductions, then refused at feature preflight for the next frozen member because
it lacked an accepted MLN HTO ID. That member's matrix values and all 151 held
members remained unopened. No model, comparator decision, prediction, pairing,
or score was formed.

The defensible methods-paper claim is therefore narrower than state-of-the-art
prediction: the framework separates pairing-dependent structure from marginal
response, estimates it against a fixed-margin link control, attaches a
support-aware refusal, and exposes where the estimand does and does not
reproduce across public paired assays.

## Public panels

| Panel | Unit | Targets | Primary result | Best declared baseline | Pairing control | Decision |
|---|---|---:|---|---|---|---|
| PerturbSci-Kinetics | Three sequence-distinct guide rotations | 85 | Pearson 0.525 [0.483, 0.566]; RMSE 0.844 [0.819, 0.868] | Endpoint ridge RMSE 0.833 | Primary-minus-destroyed Pearson CI [0.368, 0.529] | **PROMOTE pairing signal; REFUSE estimator superiority** |
| Frangieh Perturb-CITE-seq | Three sequence-distinct guides per target | 179 | Pearson 0.040 [0.013, 0.066]; RMSE 1.009 [1.002, 1.017] | Scalar shrinkage RMSE 0.999 | Primary-minus-destroyed Pearson CI [-0.092, -0.013] | **REFUSE** |
| Papalexi ECCITE-seq | Three deposited treatment replicates | 24 | Pearson -0.025 [-0.150, 0.096]; RMSE 1.037 [1.006, 1.076] | Zero RMSE 1.000 | Primary-minus-destroyed Pearson CI [-0.161, 0.216] | **REFUSE** |
| MultiPerturb-seq RNA-ATAC | Two sequence-distinct guides per target | 35 | Pearson -0.224 [-0.330, -0.124]; RMSE 1.047 [1.005, 1.095] | Zero RMSE 1.000 | Destroyed-link Pearson -0.011 [-0.065, 0.059] | **REFUSE** |
| PerturbFate | Four deposited technical dates | 49 | Pearson -0.092 [-0.152, -0.035]; RMSE 1.023 [1.010, 1.035] | Linear cross-covariance RMSE 1.000 | Direct-minus-destroyed target-MSE CI [-0.028, 0.079] | **REFUSE** |
| ReSisTrace | Two deposited cultures; lineages resampled | 3 treatment contrasts | No prediction RMSE; treatment cosines -0.674, 0.223, 0.952, and every interval crosses zero | Not applicable | All 8 arm fields exceed the fixed-margin Monte Carlo link control, each BH q=0.0154 | **REFUSE treatment replication** |
| Arce T-cell RNA-protein | Donor A development; donor B held confirmation | 28 | Pearson 0.412 [0.331, 0.494]; RMSE 1.093 [1.012, 1.185] | Linear cross-covariance: Pearson 0.304; RMSE 0.993 [0.989, 0.996] | Primary-minus-destroyed target-MSE CI [-0.448, 0.006] | **REFUSE** |
| PoKI-seq | Donor1 development; Donor2 held confirmation | 33 predeclared queries | Not scored; state-occupancy preflight failed for `Donor1:Stim:41BB` | Not evaluated | Not reached | **REFUSE preflight** |
| Lawlor HCA PBMC | Six development donors; four held donors | Predeclared aliases | Not scored; frozen reducer rejected the deposited ADT object type | Not evaluated | Not reached | **REFUSE execution** |
| Hao GSE164378 | Four development donors; three held donors | Predeclared aliases | Not scored; fewer than 12 markers passed frozen marginal support | Not evaluated | Not reached | **REFUSE support** |
| Kotliarov PBMC | Ten batch-1 development donors; nine disjoint batch-2 held donors | 71 predeclared cognate markers | Not scored; fewer than four prespecified RNA-only lineages met the frozen 50-cell minimum in all 19 donors | Full Pearson and signed-deviance residual transfer not reached | Not reached | **REFUSE support** |
| NeurIPS 2021 BMMC CITE-seq | Two fit donors, one development donor, six held donors | 100 ordered RNA--ADT pairs | Not scored; exact conditional optimizer failed the final common-effect development refit after the frozen numerical-equivalence retry | Strongest signed Pearson or Poisson-deviance residual transfer not reached | Held pairing not opened | **REFUSE numerical development** |
| GSE279451 adult sepsis CITE-seq | Nineteen development donors; 21 held donors | 81 ordered RNA--ADT pairs | Not scored; terminal evaluation refusal after development reduction because both common-effect control families were unavailable | Prespecified raw or exact-null-centered signed Pearson or Poisson-deviance residual transfer; no comparison decision produced | All 21 held matrices unopened | **REFUSE development evaluation** |
| GSE299043 MLN CITE-seq | Ten Cambridge development donors; ten LiveOnNY/Columbia held donors | 81 ordered RNA--ADT pairs | Not scored; terminal feature-preflight refusal after 21 development-member reductions | Prespecified residual comparison not reached | All 151 held members unopened | **REFUSE acquisition** |

Metric intervals for the first five predictive panels and Arce resample
targets. ReSisTrace intervals resample barcode-defined lineages conditional on
two deposited cultures; they are not population-level donor intervals. The
exact values and provenance fields are in the TSV.

## Claim/evidence ledger

| Candidate claim | Evidence | Status | Permissible wording |
|---|---|---|---|
| The population field removes separable marginal tilts. | `association_field` double-centers `log(P)`; Helmert projection retains `(r-1)(s-1)` interaction coordinates. | **SUPPORTED algebraically** | The population coordinate is invariant to positive row/column multiplicative tilts. |
| The implemented finite-sample estimator is exactly marginal invariant. | A pseudocount breaks exact finite-sample invariance. The implementation instead subtracts a fixed-margin permutation mean. | **REFUSE exact claim** | The sample estimator is conditionally centered under fixed empirical margins. |
| The held destroyed-link control is independent of the null mean. | `conditional_association_coordinates` draws `B+1` permutations, centers on draws 2 through `B+1`, and holds draw 1 out. | **SUPPORTED by code and tests** | One permutation is withheld from the `B=64` reference draws as a matched destroyed-link control. |
| Pairing-dependent perturbation structure reproduces across guides. | PerturbSci primary-minus-destroyed Pearson CI is [0.368, 0.529]. | **SUPPORTED in one panel** | Pairing-dependent pre-existing/nascent RNA structure reproduced across sequence-distinct guides in PerturbSci. |
| Pairing-dependent structure exists in linked lineages. | Every ReSisTrace arm rejects the fixed-margin Monte Carlo link control at BH q=0.0154. | **SUPPORTED at arm level** | Lineage links carried pre/post association in all eight deposited arms. |
| Treatment-specific lineage fields reproduce biologically. | All three ReSisTrace lineage-bootstrap cosine intervals include zero; only two cultures are deposited. | **REFUSE** | No treatment-minus-control field met the replication criterion. |
| The structured primary improves matched baselines. | The predeclared three-panel gate records 0 wins; MultiPerturb and PerturbFate also fail. | **REFUSE** | The structural estimator is a fixed denoiser evaluated beside simpler baselines, not a superior predictor. |
| The scGPT hypergraph contributes specific biological information. | The shuffled graph matches or exceeds the primary in PerturbSci and PerturbFate; no positive panel isolates a graph-specific gain. | **REFUSE** | scGPT supplies a frozen structural prior; its specific benefit was not established. |
| Guide/observation-error correction is validated on public data. | `fit_factorial_coupling` is tested and stress-tested synthetically, but the completed public scripts use conditional empirical tables, not the guide-aware likelihood. | **REFUSE real-data claim** | Guide-aware identification and refusal are implemented and assessed in simulation. |
| The public benchmark establishes a positive held biological replicate. | ReSisTrace refuses treatment-contrast replication. Arce donor B has positive correlation, but the fixed primary has RMSE 1.093 versus 0.993 for linear cross-covariance; primary-minus-covariance target MSE is 0.199 [0.038, 0.352], and its primary-minus-destroyed interval includes zero. PoKI-seq, Lawlor, Hao, Kotliarov, BMMC, GSE279451, and GSE299043 stopped before held scoring. | **REFUSE** | No completed panel supplies a positive held biological-replicate result under its full decision gate. |
| Corrected pathway/complex recovery validates the primary. | The strict PerturbSci rerun required high field norm in every held truth and prediction and exclusion under matched destroyed links. It selected no target; no Reactome or CORUM module could pass. | **REFUSE as confirmatory biology** | The earlier averaged-unit enrichments are exploratory; the strict replicated module analysis was negative. |
| Predicted local neighborhoods recover biological relations. | Mean top-5 neighbor recovery was 0.064 [0.050, 0.079] versus 0.051 [0.041, 0.063] after link destruction; the difference interval included zero and the random-label permutation p-value was 0.283. Reactome edge enrichment refused. CORUM was nominal against random, p=0.0269, but not significant after family correction, q=0.0538. | **REFUSE** | No corrected local-neighborhood claim survives the predeclared controls. |
| The method is state of the art for perturbation prediction. | No completed comparison establishes this, and the task is association-field reproducibility rather than standard unseen-perturbation response prediction. | **REFUSE** | Do not make a state-of-the-art or direct-superiority claim. |

## Estimator audit

The completed-panel estimator is preserved byte-for-byte at
`mapreg/historical/coupling_fields_29a3875.py`, SHA-256
`29a3875fa43572ead6c53cd7dea60bb9bdf07c35b417d79d8a97f30cbb230912`.
Its public-data path has four auditable steps: Helmert interaction coordinates,
64-reference fixed-margin centering plus one held destroyed link, clipped
median-normalized inverse permutation-variance weights, and a convex weighted
nuclear-plus-hypergraph fit. Positive weights make the smooth loss strongly
convex, and the implementation refuses a nonconverged proximal fit.

The guide-aware likelihood is separate in `mapreg/factorial_coupling.py`,
SHA-256
`c80afa0efc70de920bdd312420c0e680df3b290528b6aefd3b0a1b98cff3b79e`.
It checks guide and observation-channel rank/conditioning, linkage fraction,
effective arm support, expected joint-cell support, impossible categories, and
optimization convergence. This likelihood is not invoked by the completed
public benchmark scripts and must not be presented as their estimator.

The GSE279451 estimator is
`mapreg/hierarchical_conditional_coupling.py`. It fits donor-varying 2-by-2
log-odds by exact fixed-margin conditional likelihood and regularizes the 81
RNA--ADT interactions over marginal-only marker graphs. Its development and
held contracts require a direct win over the strongest matched classical
residual transfer. The terminal development evaluator could not instantiate
the two common-effect control families required by the full gate, so it refused
without a result or comparator decision.

The GSE299043 estimator uses the same exact conditional likelihood with
donor-varying log odds and marginal-profile RNA and ADT graphs. It adds an
executable hierarchical ridge-only control and selects the strongest raw or
exact-null-centered signed Pearson or signed-root Poisson-deviance transfer on
the same donor folds. The development interval is an adaptive promotion
heuristic because the same ten donors select and gate the model. Confirmatory
inference is reserved for the untouched held-site donors.
Neither estimator selection nor the classical-residual comparison was reached:
the frozen source-member feature gate refused first.

## Synthetic and sensitivity evidence

The final synthetic stress artifact is
`results/development/factorial_coupling_stress_v1.json`, SHA-256
`76f521decdb64945c0bf52e46dd60688eb9705224a7ff50bda027ed4fc981b48`.
It uses the current field and factorial-estimator hashes. Across 200 simulations
per condition, unconditional global rejection under the population null was
3.0% when well specified, 5.5% under heavy-tailed blocks, 3.0% under guide-
channel misspecification, 3.0% under observation-channel misspecification, and
2.0% near the linkage threshold. Sparse-support worlds refused 98.0% and 99.5%
of runs. These are simulation operating characteristics, not biological
validation.

The PerturbSci sensitivity artifact is
`results/development/perturbsci_conditional_sensitivity.json`, SHA-256
`b79006842352df2a07722b772b8018c1400353e85b115c9986a94ecfd74b0690`.
Its central three-state, 64-permutation, pseudocount-0.5 configuration is
bit-for-bit equal to the v4 panel metrics. The sign of the pairing result is
stable at 16, 32, 64, and 128 permutations. Effect size varies materially with
state count and pseudocount, so those choices remain part of the estimand.

## PerturbSci biological secondary validation

The locked secondary analysis is
`results/perturbsci_module_validation.json`, SHA-256
`68032c3bc05fe0702edd8600cf91c86d0850b1db44857e67a42845e0fc164fc2`.
Its runner and test hashes are
`42837ab0e67de088fbce1c498622c9b03fd9190b703fd27e92d305f0d1161628`
and `f302c234b93bf4455711b34b2b585d5c135fa541f5d87b19df00743c74ff1389`.
The target rule required membership in the top field-norm quintile in every
held truth and corresponding prediction, then excluded any target satisfying
the same rule after link destruction. No target passed before destroyed-link
exclusion; XRN2 passed only in the destroyed control. Among the declared
universe, 178 Reactome sets and 95 CORUM sets were tested with one-sided Fisher
tests and within-family Benjamini--Hochberg correction; none was significant.

Top-5 cosine-neighbor recovery was 0.064 [0.050, 0.079] for the primary and
0.051 [0.041, 0.063] for destroyed links. The paired difference was 0.013
[-0.003, 0.030], and the Monte Carlo target-label permutation p-value was
0.283 from 10,000 draws. Reactome neighbor-edge enrichment also refused. CORUM
neighbor-edge enrichment exceeded destroyed links, difference 0.018 [0.008,
0.029], and was nominal against random labels, p=0.0269, but did not survive
correction across the two annotation families, q=0.0538. No interpretable edge
met the replicated reporting rule. All three secondary decisions are
**REFUSE**.

## Provenance and consistency

Result and terminal-attempt artifact hashes at this snapshot:

| Artifact | SHA-256 |
|---|---|
| `results/public_coupling_atlas_benchmark_v4_final_estimator.json` | `eb250bd749c92b7278e7e8c54e89f5126f3367ed489a91b664f1d664ea083195` |
| `results/multiperturb_conditional_fields.json` | `a59d3b82b914991fbb2e08195e5401b2dde249158e5f9c923decfcc1e1df4507` |
| `results/development/perturbfate_conditional_fields.json` | `cdf0c2cf66facfbbdfb03ac7a20c97d4cdaf4fc0bc9c63025e825dbeb1715c26` |
| `results/resistrace_conditional_fields.json` | `edef0dba1d1dd94f19829088cb1fcd00f72dbacbc0695516b9c25afee6b20ffb` |
| `results/arce_gse278572_conditional_field_confirmation.json` | `65d4bf6097a8fafee8e22f352c2c6c14fa13c1d8082073a4589d3ee693ef8b57` |
| `results/arce_gse278572_postlock_controls.json` | `66d3343db745ad338091397226a9b93ca222df759107fbe46eddfcbee4c1a612` |
| `results/perturbsci_module_validation.json` | `68032c3bc05fe0702edd8600cf91c86d0850b1db44857e67a42845e0fc164fc2` |
| `results/gse143417_pokiseq_preflight_refusal.json` | `24f7ad70fbbfd4e7482809db58bd94d1156c1e22c2dd94fa77d66b1d6acdcf24` |
| `results/kotliarov_pbmc_public_refusal.json` | `34d59fcbdcceeefb449a430bca7a0f502611d343a2ebd19fc44a7f5fd26a1324` |
| `results/development/scmmib_bmmc_exact_development_attempt_3_terminal_refusal.json` | `caf920719694487ba228dc64ac14ed4a6579619349f496f7154372920f3e128c` |
| `data/confirmation/gse279451_sepsis/source_manifest_v1.json` | `c15fdc13c68cff14c45cfc16153cdbc309f6c2252457634fda8abdca677e4603` |
| `data/development/gse279451_sepsis/development_attempt_v1.json` | `b8d7e745997e7b7ccd9a3bc8a7a7c3c7670b9e47c8b5d51e3d12fb70a5a8938d` |
| `data/development/gse279451_sepsis/evaluation_attempt_v1.json` | `a072d3340388afa60a33ef220b06e9e8d920a590d6b6894f66f02b999e1ef5a3` |
| `data/development/gse279451_sepsis/reduced_development_v1.json` | `2fab0353c44d65f6cf3d58ca19a214967c94fdccaeda89ed56c7398fb8f4185b` |
| `results/development/gse279451_sepsis_evaluation_refusal.json` | `af6d1f26eb7ea3f566612e167843bd51c03cf961a595b434e55ae7ca4d20496b` |
| `data/development/gse299043_mln/development_attempt_v1.json` | `dc6206cba4186ede6a2d9a178d6f9adb9e09e402983e32dda611b0be11ac00ad` |
| `results/development/gse299043_mln_development_acquisition_refusal.json` | `aab390fc9701171c42267d111dfb69cbbf3282abba5d39887f30431bc1e78635` |
| `results/development/gse299043_mln_terminal_acquisition_audit.json` | `6b5ef33e765adfcd80b534113a3c59dec706b82739c2518342699019011a6475` |

The structured consistency check parsed every audited JSON with non-finite JSON
constants rejected; verified embedded hashes for the v4 runner, estimator, and
three state caches; all five MultiPerturb inputs/code artifacts; all six
PerturbFate inputs/code artifacts; the three stress artifacts; the three
PerturbSci sensitivity artifacts; and all 32 ReSisTrace raw files and byte
counts. It also verified the Arce runner, source manifest, locked protocol,
locally retained small inputs, and analysis cache; the strict PerturbSci runner,
tests, source result, estimator, and annotation inputs; and a byte-identical
strict-analysis rerun.
Every reported metric lay inside its interval, declared-best comparator
selection was reproduced, the v4 promotion count was reproduced, and both the
PerturbSci central sensitivity configuration and strict-analysis fold summary
matched the main result exactly.

The Arce source manifest records SHA-256
`344a890a56a93bb01c4f7d86bd55ecca118950af43e20023e84c9a689cbb00d5`
for the streamed 3.86-GB GEO matrix. That raw matrix is not retained in the
working tree, so the audit rehashed the four retained source files, metadata
audit, and 18-MB analysis cache rather than the matrix itself. The corrected
lock is an analytical holdout, not a literal unopened-file claim: donor-B raw
bytes entered a transient buffer in an interrupted prior stream, but no donor-B
summary, state, field, score, cache, result, or human-visible value informed the
locked estimator.

The v4 JSON records preanalysis hash
`7530303b154cba52cf816562af4cec0330a4ada23b9535f1cb9c7e1ddc6f8ae3`.
This is the exact historical protocol version at run time. The current protocol
file has SHA-256
`34facaeec5c4742decaa127a87ad9a7928bc6435fbd073bb00d13226929a2547`
after later appendices; it is not byte-identical to the recorded run version.

Focused estimator, inference, stress, and public-API tests passed with
`PYTHONPATH=. pytest -q tests/test_coupling_fields.py tests/test_factorial_coupling.py tests/test_factorial_coupling_inference.py tests/test_mapreg_public_api.py tests/test_stress_factorial_coupling.py`:
**47 passed**.
The strict PerturbSci validation plus its adjacent field tests passed with
`PYTHONPATH=. pytest -q tests/test_validate_perturbsci_modules.py tests/test_coupling_fields.py`:
**29 passed**; Ruff reported no violations in the new runner or tests.

## Manuscript boundary

Lead with the estimand, fixed-margin correction, linked-assay input contract,
and transparent refusal. Present PerturbSci as the positive held-guide result
and ReSisTrace as positive arm-level linkage but negative treatment replication.
Keep every RNA-protein/RNA-ATAC/PerturbFate refusal, including the terminal
BMMC and GSE279451 development refusals, in the main benchmark table.
Retain the GSE299043 terminal feature refusal in the main table; it supplies no
performance evidence.
Do not claim structural-prior superiority, confirmatory pathway recovery,
local-neighborhood recovery, real-data validation of guide-error correction,
positive biological replication, or state-of-the-art perturbation prediction
unless a later artifact directly closes the corresponding row above.
