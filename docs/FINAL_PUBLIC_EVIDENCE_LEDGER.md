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
support refusal rather than outcome-scored evidence.

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
| The public benchmark establishes a positive held biological replicate. | ReSisTrace refuses treatment-contrast replication. Arce donor B has positive correlation, but the fixed primary has RMSE 1.093 versus 0.993 for linear cross-covariance; primary-minus-covariance target MSE is 0.199 [0.038, 0.352], and its primary-minus-destroyed interval includes zero. The frozen PoKI-seq candidate refused at the state-occupancy preflight before prediction or scoring. | **REFUSE** | No panel supplies a positive held biological-replicate result under its full decision gate. |
| Corrected pathway/complex recovery validates the primary. | The strict PerturbSci rerun required high field norm in every held truth and prediction and exclusion under matched destroyed links. It selected no target; no Reactome or CORUM module could pass. | **REFUSE as confirmatory biology** | The earlier averaged-unit enrichments are exploratory; the strict replicated module analysis was negative. |
| Predicted local neighborhoods recover biological relations. | Mean top-5 neighbor recovery was 0.064 [0.050, 0.079] versus 0.051 [0.041, 0.063] after link destruction; the difference interval included zero and the random-label permutation p-value was 0.283. Reactome edge enrichment refused. CORUM was nominal against random, p=0.0269, but not significant after family correction, q=0.0538. | **REFUSE** | No corrected local-neighborhood claim survives the predeclared controls. |
| The method is state of the art for perturbation prediction. | No completed comparison establishes this, and the task is association-field reproducibility rather than standard unseen-perturbation response prediction. | **REFUSE** | Do not make a state-of-the-art or direct-superiority claim. |

## Estimator audit

The release estimator is `mapreg/coupling_fields.py`, SHA-256
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

Result artifact hashes at this snapshot:

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
Keep every RNA-protein/RNA-ATAC/PerturbFate refusal in the main benchmark table.
Do not claim structural-prior superiority, confirmatory pathway recovery,
local-neighborhood recovery, real-data validation of guide-error correction,
positive biological replication, or state-of-the-art perturbation prediction
unless a later artifact directly closes the corresponding row above.
