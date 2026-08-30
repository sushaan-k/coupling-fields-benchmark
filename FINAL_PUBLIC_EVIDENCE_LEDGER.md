# Final public evidence ledger

**Snapshot:** 29 August 2026

**Machine-readable ledgers:** `results/benchmark_panels_v2.tsv`,
`results/benchmark_comparisons_v2.tsv`, and
`results/benchmark_sequence_v2.tsv`

## Bottom line

The completed evidence supports the conditional association-field estimand,
fixed-margin null centering, explicit refusal, and one prospective held-site
transfer. In the publicly frozen Stephenson confirmation, the hierarchical
exact-conditional primary reduced mean Poisson deviance per cell by 17.46%
relative to the pilot-selected signed-deviance residual across 56 physical
samples. The paired raw loss-difference interval was [-0.00413, -0.00080], 50
of 56 samples were favorable, and the exact one-sided sign-test p-value was
5.09e-10. The primary also reduced loss by 46.36% relative to destroyed links.
This is the benchmark's confirmatory positive result.

A post-hoc classical audit, executed after held outcomes had been accessed,
found 6.18% lower Stephenson loss than the exact common-effect CMLE (relative
95% interval [3.84%, 8.47%]) and 3.39% lower loss than pooled-table log odds
reconstructed with the conditional fixed-margin expectation ([1.42%, 5.37%]).
The frozen artifact had labeled the latter comparator as pooled Poisson; its
bytes are unchanged, and the public ledger now states the implemented method.
These comparisons are descriptive. The GSE239452 held-cohort analysis is
separately labeled as a post-access correction. A later numerical audit found
that endpoint underflow in signed-root-deviance inversion had misreconstructed
80 of 729 residual coordinates. Corrected inversion leaves primary loss
unchanged at 0.0085063650 and gives residual loss 0.0141314858, a 39.8056%
reduction with a donor-bootstrap difference interval of
[-0.00709509, -0.00423212]; all nine
donors favor the primary. The destroyed-link reduction remains 78.89%, but the
exact common-effect CMLE was 2.20% better than the primary. The original sealed
prediction and score remain byte-identical and preserve the original chronology;
the aggregate panel and gate rows point to the correction artifact.

A separate post-hoc GSE239452 audit used a standard fixed-interaction Poisson
prediction. The structured primary's mean loss was 0.0085063650 versus
0.0099824140 for Poisson, 14.7865% lower, with a donor-bootstrap relative
interval of [12.1841%, 17.3694%] and a raw-difference interval of
[-0.00184323, -0.00116575]. All nine donors favored the primary (one-sided
sign-test p=1/512). The comparison was defined after held-outcome access and
uses the same corrected cohort, so it is neither confirmatory nor an
independent-cohort replication. The positive panels therefore do not support a
claim that structural regularization dominates every classical estimator.

The historical public panels retain their original decisions. PerturbSci has a
reproduced pairing signal but loses to endpoint ridge. ReSisTrace has positive
arm-level link tests but no replicated treatment contrast. Frangieh, Papalexi,
MultiPerturb, PerturbFate, and Arce refuse their full gates. GSE314416 also
refused at its pilot gate: its primary improved on the selected residual by
0.49%, below the frozen five-percent threshold, and the paired interval crossed
zero. Retrospective adaptive development analyses on BMMC and COMBAT reduced
loss relative to matched residuals by 17.47% and 25.94%, respectively; the same
nonheld units selected the configurations and supplied the summaries, so these
are development evidence rather than held confirmation. Both selected graph
penalty zero.

PoKI-seq, Lawlor, Hao, Kotliarov, BMMC, GSE279451, GSE299043, and COMBAT retain
their prior procedural refusals. Twelve subsequent public source campaigns also
terminated before held scoring and are enumerated in the panels ledger. The
source-only Kotliarov binary-v2 replacement separately refused because no
frozen configuration completed every source-held fold; it produced no
comparison decision or held run. The GSE179221 BMMC candidate then refused at
the exact cognate-axis gate on its first source donor. It opened barcode and
feature axes but no count dataset; the remaining seven source files and all ten
held files were unrequested. No table, model, comparison, prediction, or score
was formed. The GSE214546 TEA-seq campaign reduced the first source donor at the
frozen 512-cell, 53-marker budget, then refused because the second source donor
had fewer than 512 matched singlets. The exact overlap count was not serialized;
the remaining six source H5s and all eight held H5s stayed unopened. No model,
comparison, prediction, or score was formed. The recovery-amended unused-Cambridge run
produced no prediction or score and is
infrastructure-unevaluable. Its row contains no performance value and remains
bound to the published terminal record; no scientific decision is assigned.

The supported methods claim is precise: the framework estimates
pairing-dependent interactions against fixed-margin and classical controls,
can transfer those interactions in one prospectively frozen held-site panel,
and exposes negative, corrected, development-only, and unevaluable analyses
without recoding them as confirmations. It is not evidence for state-of-the-art
perturbation prediction or universal structured-estimator superiority.

## Public panels

| Panel | Unit | Targets | Primary result | Best declared baseline | Pairing control | Decision |
|---|---|---:|---|---|---|---|
| Stephenson Newcastle held site | 56 physical samples | 81 ordered RNA--ADT pairs | Mean deviance/cell 0.012197 | Selected signed-deviance residual 0.014777; 17.46% reduction; raw-difference CI [-0.00413, -0.00080] | Destroyed link 0.022738; 46.36% reduction | **CONFIRMATION PASS** |
| GSE239452 held cohort | 9 physical donors | 81 ordered RNA--ADT pairs | Mean deviance/cell 0.008506 | Corrected selected residual 0.014131, 39.8056% reduction, difference CI [-0.00709509, -0.00423212]; exact CMLE 0.008324, 2.20% better than primary; standard fixed-interaction Poisson 0.009982, primary 14.7865% lower | Destroyed link 0.040288; 78.89% reduction | **PASS, POST-ACCESS NUMERICAL CORRECTION; POISSON COMPARISON POST-HOC** |
| GSE314416 immunomicrobiome pilot | 20 physical donors | 81 ordered RNA--ADT pairs | Mean deviance/cell 0.001927 | Selected residual 0.001936; 0.49% reduction; raw-difference CI [-0.0000182, -0.00000113] | Destroyed link 0.001936; 0.45% reduction; CI crosses zero | **REFUSE PILOT** |
| BMMC adaptive development | 4 nonheld units | 100 ordered RNA--ADT pairs | Mean deviance/cell 0.010851 | Selected residual 0.013148; 17.47% reduction | Destroyed link 0.013351; 18.73% reduction | **DESCRIPTIVE DEVELOPMENT ONLY** |
| COMBAT adaptive development | 24 Oxford pilot samples | 81 ordered RNA--ADT pairs | Mean deviance/cell 0.011558 | Selected residual 0.015605; 25.94% reduction | Destroyed link 0.046395; 75.09% reduction | **DESCRIPTIVE DEVELOPMENT ONLY** |
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
| Kotliarov PBMC binary-v2 source replacement | Ten batch-1 development donors; nine disjoint batch-2 held donors reserved but not run | 81 ordered RNA--ADT pairs | Not scored; no frozen configuration completed every source-held fold | No comparison decision produced | Held ADT access remained unauthorized | **REFUSE source execution** |
| GSE179221 BMMC held-donor campaign | Eight source donors; ten donor-disjoint held donors reserved but not run | 81 ordered RNA--ADT pairs | Not scored; exact cognate-axis uniqueness failed on the first source donor before count-matrix access | No estimator or comparison was formed | Seven remaining source files and all ten held files unrequested | **REFUSE source feature-axis preflight** |
| GSE214546 TEA-seq held-donor campaign | Two of eight source donors accessed; eight held donors reserved but not run | 53 candidate matched RNA--protein markers | Not scored; the first donor completed its 512-cell reduction, and the second had fewer than 512 matched singlets | No estimator or comparison was formed | Six remaining source H5s and all eight held H5s unopened | **REFUSE source support** |
| NeurIPS 2021 BMMC CITE-seq | Two fit donors, one development donor, six held donors | 100 ordered RNA--ADT pairs | Not scored; exact conditional optimizer failed the final common-effect development refit after the frozen numerical-equivalence retry | Strongest signed Pearson or Poisson-deviance residual transfer not reached | Held pairing not opened | **REFUSE numerical development** |
| GSE279451 adult sepsis CITE-seq | Nineteen development donors; 21 held donors | 81 ordered RNA--ADT pairs | Not scored; terminal evaluation refusal after development reduction because both common-effect control families were unavailable | Prespecified raw or exact-null-centered signed Pearson or Poisson-deviance residual transfer; no comparison decision produced | All 21 held matrices unopened | **REFUSE development evaluation** |
| GSE299043 MLN CITE-seq | Ten Cambridge development donors; ten LiveOnNY/Columbia held donors | 81 ordered RNA--ADT pairs | Not scored; terminal feature-preflight refusal after 21 development-member reductions | Prespecified residual comparison not reached | All 151 held members unopened | **REFUSE acquisition** |
| COMBAT CITE-seq | Twelve Oxford calibration samples; 24 Oxford pilot samples; 51 Oxford and ten St George's held samples | 81 ordered RNA--ADT pairs | Not scored; two of eight primary configurations survived, but all four matched residual candidates and all three ridge-only Haldane candidates refused under the frozen attainable-margin rule | Pearson: 134 out-of-range pilot sample--entity pairs; deviance: 102 | All 61 held samples unopened | **REFUSE pilot candidate availability** |
| Stephenson unused-Cambridge recovery | Recovery-amended single replacement | 81 ordered RNA--ADT pairs | No prediction or score; infrastructure-unevaluable | Not evaluated | Not reached | **TERMINAL INFRASTRUCTURE-UNEVALUABLE; NO SCIENTIFIC DECISION** |

Metric intervals for the first five predictive panels and Arce resample
targets. ReSisTrace intervals resample barcode-defined lineages conditional on
two deposited cultures; they are not population-level donor intervals. The
exact values and provenance fields are in the version-2 ledgers. Eleven
additional terminal source campaigns appear only in the machine-readable panel
and sequence ledgers because they contain no performance estimate.

## Claim/evidence ledger

| Candidate claim | Evidence | Status | Permissible wording |
|---|---|---|---|
| The population field removes separable marginal tilts. | `association_field` double-centers `log(P)`; Helmert projection retains `(r-1)(s-1)` interaction coordinates. | **SUPPORTED algebraically** | The population coordinate is invariant to positive row/column multiplicative tilts. |
| The implemented finite-sample estimator is exactly marginal invariant. | A pseudocount breaks exact finite-sample invariance. The implementation instead subtracts a fixed-margin permutation mean. | **REFUSE exact claim** | The sample estimator is conditionally centered under fixed empirical margins. |
| The held destroyed-link control is independent of the null mean. | `conditional_association_coordinates` draws `B+1` permutations, centers on draws 2 through `B+1`, and holds draw 1 out. | **SUPPORTED by code and tests** | One permutation is withheld from the `B=64` reference draws as a matched destroyed-link control. |
| Pairing-dependent perturbation structure reproduces across guides. | PerturbSci primary-minus-destroyed Pearson CI is [0.368, 0.529]. | **SUPPORTED in one panel** | Pairing-dependent pre-existing/nascent RNA structure reproduced across sequence-distinct guides in PerturbSci. |
| Pairing-dependent structure exists in linked lineages. | Every ReSisTrace arm rejects the fixed-margin Monte Carlo link control at BH q=0.0154. | **SUPPORTED at arm level** | Lineage links carried pre/post association in all eight deposited arms. |
| Treatment-specific lineage fields reproduce biologically. | All three ReSisTrace lineage-bootstrap cosine intervals include zero; only two cultures are deposited. | **REFUSE** | No treatment-minus-control field met the replication criterion. |
| The structured primary improves matched baselines. | Stephenson passes its frozen residual and destroyed-link comparisons. GSE239452 passes only as a post-access correction. The historical panels and GSE314416 do not pass; BMMC and COMBAT gains are adaptive development summaries. | **SUPPORTED in one prospective held-site panel** | The hierarchical exact-conditional transfer improved the selected residual in the frozen Stephenson confirmation; general superiority is not established. |
| Hierarchical transfer adds beyond classical common-effect interaction models. | In post-hoc audits, Stephenson improves on exact common-effect CMLE by 6.18% and on pooled-table log odds with conditional reconstruction by 3.39%. In GSE239452, the primary is 14.7865% better than the true standard fixed-interaction Poisson comparator, but exact CMLE beats the primary by 2.20%. The older GSE239452 artifact's 14.58% comparison also used conditional reconstruction and is not labeled standard Poisson in the release ledger. | **MIXED, POST-HOC** | The donor-varying transfer can outperform common-effect controls, but the direction is study-dependent and awaits a prospectively frozen classical comparison. |
| The scGPT hypergraph contributes specific biological information. | The shuffled graph matches or exceeds the primary in PerturbSci and PerturbFate; no positive panel isolates a graph-specific gain. | **REFUSE** | scGPT supplies a frozen structural prior; its specific benefit was not established. |
| Guide/observation-error correction is validated on public data. | `fit_factorial_coupling` is tested and stress-tested synthetically, but the completed public scripts use conditional empirical tables, not the guide-aware likelihood. | **REFUSE real-data claim** | Guide-aware identification and refusal are implemented and assessed in simulation. |
| The public benchmark establishes a positive held biological replicate. | Stephenson passes its publicly frozen held-site gate across 56 physical samples. GSE239452 is positive but labeled post-access correction; GSE179221 stopped at source feature-axis preflight before count access, and the remaining held candidates refuse or stop before scoring. | **SUPPORTED once prospectively** | One prospectively frozen held-site confirmation passes; no second independent-study confirmation is claimed. |
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

The GSE179221 protocol prospectively added a standard pooled saturated-Poisson
interaction comparator reconstructed by refitting row and column nuisance
parameters at recipient margins. The first source file failed exact cognate-axis
uniqueness before any count dataset was opened, so neither that comparator nor
the hierarchical estimator was fit. GSE179221 therefore contributes an audited
procedural refusal, not a classical head-to-head result.

The GSE214546 protocol froze age-conditioned exact-conditional transfer and
pooled and age-stratified fixed-interaction Poisson comparators. Its source
support gate refused on the second donor before source fitting, so none of those
estimators or comparisons was formed. GSE214546 contributes a source-support
refusal, not a confirmation or head-to-head result.

Stephenson, GSE239452, GSE314416, and the adaptive BMMC/COMBAT audit use the
same exact conditional interaction family. The Stephenson configuration was
selected on 12 calibration and 24 pilot samples, refitted without retuning,
published with held predictions, and scored once against 56 physical samples.
Its selected graph penalty was zero, so the result validates donor-varying
exact-conditional transfer rather than a graph-specific gain. The classical
audit compares the frozen fields with a donor-stratified exact common-effect
CMLE and pooled-table log odds reconstructed through the conditional
fixed-margin expectation on identical held margins. The frozen artifact called
the latter pooled Poisson; the artifact remains byte-identical, whereas the
release ledger uses the corrected method name. That audit is post-hoc in both
studies.

The GSE239452 aggregate uses
`results/gse239452_citeseq_post_access_correction.json`. The correction replaces
the signed-root-deviance inversion's machine-adjacent feasible endpoints with a
fixed interior epsilon. It resolves all 80 mismatches among 729 reconstructed
residual coordinates without changing the primary or destroyed-link tables.
The original sealed prediction and score artifacts remain unchanged and appear
before the explicit numerical-correction stage in the sequence ledger.

The separate GSE239452 standard-Poisson audit fits donor-pooled saturated 2-by-2
Poisson interactions, fixes the development-selected transport multiplier at
1, and profiles recipient row and column nuisance terms at the observed held
margins. No held donor entered multiplier selection. All 81 refitted source
tables replayed their normalized fitted counts to maximum error 1.78e-16. The
nine official held RNA/ADT file pairs were reproduced sequentially and deleted
after each donor. Because the method and comparison were specified after held
outcomes had been accessed, the result remains post-hoc and nonconfirmatory.
`results/benchmark_comparisons_v2.tsv` preserves the raw and relative loss
differences, intervals, favorable-unit counts, p-values, and inference roles
without merging either audit into the prospective gate.

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
| `results/stephenson_citeseq_confirmation.json` | `5eb5fd2b41df7f4f7d822a92765ffe69854dcbe5f572f2db35cf433d7dd0adb1` |
| `results/gse239452_citeseq_confirmation.json` | `12c23d6502c9192b93838d27ebcae42c4ff92e1a3b68e27cc6584efb92f22e74` |
| `results/gse239452_citeseq_post_access_correction.json` | `1eafd82805a0bc6d94c05afdc4160fd6917e1145d64077fb52a770e09f45793b` |
| `results/development/gse314416_citeseq_development.json` | `6124bbbdee233e987521f004a9c409858203106eeae9d2339bbea2aa4dc35f33` |
| `results/development/exact_logodds_head_to_head_v1.json` | `22bbad8efad2dd9b5172a33e8b2fdaf568114103fd7045a80a58eb842bb09d2a` |
| `results/development/classical_interaction_baselines_posthoc.json` | `bc6efbb2ffe3404a294eae26b51e214054718113a8deeaf6b9f4e73ebf05f305` |
| `results/development/gse239452_standard_poisson_interaction_posthoc.json` | `54b32b1ed12a01030210cd415b4565347868faaed13dd6fc37b6c0100c3aac97` |
| `results/perturbsci_module_validation.json` | `68032c3bc05fe0702edd8600cf91c86d0850b1db44857e67a42845e0fc164fc2` |
| `results/gse143417_pokiseq_preflight_refusal.json` | `24f7ad70fbbfd4e7482809db58bd94d1156c1e22c2dd94fa77d66b1d6acdcf24` |
| `results/kotliarov_pbmc_public_refusal.json` | `34d59fcbdcceeefb449a430bca7a0f502611d343a2ebd19fc44a7f5fd26a1324` |
| `data/confirmation/kotliarov_pbmc_binary_v2/source_attempt_v2.json` | `8d276cc0c404fc2a379390d478ec1a14582f6aa6eaa469b915b15b70d6450b1a` |
| `results/development/kotliarov_pbmc_binary_v2_source_v2.json` | `12aacf4dc05efabcd2d745abc0319f6a2676e5d26eb50054849005424b1a071c` |
| `data/confirmation/kotliarov_pbmc_binary_v2/source_access_code_path_certificate_v2.json` | `1fed7f94958a07a71a195e80ce2b88f326ff2f47274733133c6b4f7dfd47d0d6` |
| `data/confirmation/gse179221_bmmc/source_attempt_v1.json` | `e9785294eec3420c813006d88ed5a264de1ec1a119c3725b800aaf955d87f4ec` |
| `data/confirmation/gse179221_bmmc/source_consumption_v1.json` | `95c0c640fddc6b70cb7dd4b509e27f0e0604186df89ad057eeff7bc93f2fd871` |
| `results/development/gse179221_bmmc_source_v1.json` | `18982f0320c602dbc65df27a94675677dc006edd9951ac62fa3a1ad93e2a06f6` |
| `data/confirmation/gse214546_teaseq/source_attempt_v1.json` | `56a832e0f3b67ceae87e7a645275a7ec4607f350dd56456f37be9149c906795f` |
| `results/development/gse214546_teaseq_source_v1.json` | `fb7ed8218c926cbc41a105b21a94116d8f73de5fd823b98137ac094b20d410ba` |
| `results/development/scmmib_bmmc_exact_development_attempt_3_terminal_refusal.json` | `caf920719694487ba228dc64ac14ed4a6579619349f496f7154372920f3e128c` |
| `data/confirmation/gse279451_sepsis/source_manifest_v1.json` | `c15fdc13c68cff14c45cfc16153cdbc309f6c2252457634fda8abdca677e4603` |
| `data/development/gse279451_sepsis/development_attempt_v1.json` | `b8d7e745997e7b7ccd9a3bc8a7a7c3c7670b9e47c8b5d51e3d12fb70a5a8938d` |
| `data/development/gse279451_sepsis/evaluation_attempt_v1.json` | `a072d3340388afa60a33ef220b06e9e8d920a590d6b6894f66f02b999e1ef5a3` |
| `data/development/gse279451_sepsis/reduced_development_v1.json` | `2fab0353c44d65f6cf3d58ca19a214967c94fdccaeda89ed56c7398fb8f4185b` |
| `results/development/gse279451_sepsis_evaluation_refusal.json` | `af6d1f26eb7ea3f566612e167843bd51c03cf961a595b434e55ae7ca4d20496b` |
| `data/development/gse299043_mln/development_attempt_v1.json` | `dc6206cba4186ede6a2d9a178d6f9adb9e09e402983e32dda611b0be11ac00ad` |
| `results/development/gse299043_mln_development_acquisition_refusal.json` | `aab390fc9701171c42267d111dfb69cbbf3282abba5d39887f30431bc1e78635` |
| `results/development/gse299043_mln_terminal_acquisition_audit.json` | `6b5ef33e765adfcd80b534113a3c59dec706b82739c2518342699019011a6475` |
| `results/development/combat_citeseq_pilot_terminal_refusal.json` | `c41c6e46333c1dc56b460fec74a41f6d1a07a82105ca4894a273ce43dc48e2f9` |

The aggregate builder rejects non-finite JSON, verifies every source artifact
hash before emitting a row, enforces unique panel/comparison identifiers and
valid cross-references, and writes sorted deterministic ledgers. The standalone
verifier rechecks row-level artifact hashes, metric finiteness, evidence counts,
sequence order, the empty unused-Cambridge performance fields, manifest bytes,
and `SHA256SUMS`. The release candidate contains 36 panel, 29 comparison, and
83 sequence records.

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

The release entry point completed with **525 passed, 4 deselected, and 12
numerical warnings**; its isolated historical-estimator assertion also passed.
A second deterministic build was byte-identical, the standalone verifier
returned `PUBLIC_BENCHMARK_RELEASE_VERIFICATION_PASS`, and Ruff reported no
violations in the builder, verifier, or focused release tests.

## Manuscript boundary

Lead with the estimand, exact conditional transfer, the Stephenson prospective
held-site pass, and the frozen residual and destroyed-link gates. Report the
exact-CMLE and pooled-table-log-odds conditional reconstructions as post-hoc.
Report the true fixed-interaction Poisson result separately as a post-hoc,
nonconfirmatory GSE239452 comparison, not as a new cohort. Label GSE239452 as a
post-access numerical correction, retain the original sealed chronology, and
label GSE314416 as a failed pilot. Treat the BMMC/COMBAT gains as retrospective
adaptive development evidence. Retain PerturbSci's positive
held-guide pairing signal and ReSisTrace's positive arm-level linkage beside
their negative full decisions. Keep every procedural refusal and source-stage
terminal record in the machine-readable benchmark. Treat unused Cambridge as
infrastructure-unevaluable, with no scientific decision or performance value.
Treat GSE179221 as a source feature-axis refusal with no numerical or held
outcome: it does not supply a second confirmation or a classical comparison.
Treat GSE214546 as a source-support refusal with one complete source reduction,
one axis-only failed source reduction, and no held access, fit, comparison, or
confirmation.
Do not claim graph-prior superiority, a second independent-study confirmation,
confirmatory pathway or neighborhood recovery, real-data validation of
guide-error correction, universal estimator superiority, or state-of-the-art
perturbation prediction.
