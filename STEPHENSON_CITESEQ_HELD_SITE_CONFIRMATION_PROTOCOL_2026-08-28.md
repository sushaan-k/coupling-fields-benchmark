# Stephenson CITE-seq held-site confirmation protocol

**Prospective status (2026-08-28):** designated with outcome access disabled.
No RNA or ADT matrix value may be decoded until the protocol, source manifest,
runner, tests, and imported numerical modules are present at the immutable
public tag `stephenson-citeseq-v1.1-protocol`, independently fetched from the
public repository, and covered by a separately committed authorization.

## Question and source

This test asks whether an RNA-to-surface-protein coupling field fitted in
Cambridge participants transfers to independently recruited Newcastle
participants. The source is Stephenson et al., E-MTAB-10026
(`10.1038/s41591-021-01329-2`). The official processed file is
`covid_portal_210320_with_raw.h5ad` (7,187,322,881 bytes; MD5
`add2501947c585ea6b6ef7429e4bd9f2`; SHA-256
`ec48f328f2e884c23376c8aa1f26041e11625762be5c30b0bd0869aa8bb1a334`) and the official SDRF is
`E-MTAB-10026.sdrf.txt` (155,174 bytes; SHA-256
`68a27790e45b025f71f445c5ab6dbdc15d5fd74312f8d5366390759ff0580dc5`).
The H5AD `sample_id` joins to SDRF `Source Name`, except for two exact
author-deposited Cambridge label swaps: SDRF `BGCV06_CV0326` maps to H5AD
`BGCV13_CV0326`, and SDRF `BGCV13_CV0201` maps to H5AD `BGCV06_CV0201`.
In both cases the H5AD `patient_id` suffix confirms the SDRF individual. No
other fuzzy or donor-based join is allowed. SDRF `Characteristics[individual]`
is the biological donor key; H5AD `patient_id` is not used for allocation,
particularly because it is not a biological-donor key in Newcastle.

Cambridge and Newcastle used the same BioLegend custom TotalSeq-C panel
(`99813`). The 11 Sanger/London donors used panel `99814` and are excluded.
The six Newcastle LPS-stimulated donors (`IVLPS-1`, `IVLPS-2`, `IVLPS-3`,
`IVLPS-4`, `IVLPS-6`, and `IVLPS-12`) are excluded before allocation. No
donor occurs in more than one site.

## Frozen allocation

For every Cambridge or unstimulated Newcastle biological donor, retain samples
with at least 512 cells and choose one by ascending
`SHA256('STEPHENSON-CITESEQ-SAMPLE-v1' || NUL || donor || NUL || sample)`,
breaking a digest tie by sample ID. This yields 47 Cambridge donors and 56
Newcastle donors.

Within each Cambridge disease stratum, sort by the analogous digest with salt
`STEPHENSON-CAMBRIDGE-CALIBRATION-v1` and take 9 COVID-19 plus 3 normal donors.
Remove them, sort the remainder with salt
`STEPHENSON-CAMBRIDGE-PILOT-v1`, and take 18 COVID-19 plus 6 normal donors.
The remaining 11 Cambridge donors are unused. All 56 eligible Newcastle donors
form the held-site panel.

- Calibration (12): `CV0052:BGCV02_CV0052`, `CV0062:BGCV12_CV0062`,
  `CV0084:BGCV03_CV0084`, `CV0171:BGCV13_CV0171`, `CV0231:BGCV10_CV0231`,
  `CV0262:BGCV04_CV0262`, `CV0279:BGCV09_CV0279`, `CV0284:BGCV14_CV0284`,
  `CV0326:BGCV13_CV0326`, `CV0902:BGCV01_CV0902`, `CV0926:BGCV12_CV0926`,
  `CV0934:BGCV13_CV0934`.
- Adaptive pilot (24): `CV0025:BGCV01_CV0025`, `CV0037:BGCV06_CV0037`,
  `CV0050:BGCV11_CV0050`, `CV0058:BGCV11_CV0058`, `CV0059:BGCV02_CV0059`,
  `CV0068:BGCV02_CV0068`, `CV0074:BGCV03_CV0074`, `CV0104:BGCV07_CV0104`,
  `CV0120:BGCV05_CV0120`, `CV0128:BGCV05_CV0128`, `CV0137:BGCV07_CV0137`,
  `CV0144:BGCV01_CV0144`, `CV0155:BGCV08_CV0155`, `CV0160:BGCV10_CV0160`,
  `CV0164:BGCV04_CV0164`, `CV0178:BGCV06_CV0178`, `CV0198:BGCV10_CV0198`,
  `CV0234:BGCV06_CV0234`, `CV0904:BGCV01_CV0904`, `CV0915:BGCV08_CV0915`,
  `CV0917:BGCV09_CV0917`, `CV0929:BGCV05_CV0929`, `CV0939:BGCV10_CV0939`,
  `CV0944:BGCV15_CV0944`.
- Unused Cambridge (11): `CV0073:BGCV08_CV0073`, `CV0094:BGCV07_CV0094`,
  `CV0100:BGCV04_CV0100`, `CV0134:BGCV07_CV0134`, `CV0176:BGCV15_CV0176`,
  `CV0180:BGCV11_CV0180`, `CV0200:BGCV03_CV0200`, `CV0201:BGCV06_CV0201`,
  `CV0257:BGCV11_CV0257`, `CV0911:BGCV04_CV0911`, `CV0940:BGCV14_CV0940`.
- Held Newcastle (56): `C-8816:MH9143270`, `C-8820:MH9143320`,
  `C-8821:MH9179821`, `C-8822:MH9179822`, `C-8823:MH8919326`,
  `C-8825:MH9143321`, `C-8826:MH9143420`, `C-8827:MH9143271`,
  `C-8828:MH9179823`, `C-8829:MH9179824`, `C-8882:MH9143322`,
  `C-8883:newcastle20`, `C-8884:newcastle21`, `C-8885:MH9143421`,
  `C-8886:MH9143272`, `C-8887:MH9179826`, `C-8890:MH9143323`,
  `C-8892:MH9143370`, `C-8893:MH9143324`, `C-8895:MH9179825`,
  `C-8896:MH8919327`, `C-8897:MH9143422`, `C-8898:MH9143371`,
  `C-8899:MH9143325`, `C-8902:MH9143423`, `C-8903:MH9143274`,
  `C-8905:MH9143424`, `C-8909:MH9143426`, `C-8910:newcastle49`,
  `C-8912:MH8919329`, `C-8913:MH9179827`, `C-8914:MH8919226`,
  `C-8915:MH8919330`, `C-8918:MH9143275`, `C-8921:MH9143427`,
  `C-8922:MH9143425`, `C-8923:MH9179828`, `C-8926:MH8919331`,
  `C-8927:MH9143372`, `C-8928:MH8919178`, `C-8929:MH9143276`,
  `C-8930:MH8919179`, `C-8931:newcastle59`, `C-8933:MH9143373`,
  `C-8934:MH9143326`, `C-8935:MH9143327`, `C-8936:newcastle74`,
  `C-8937:newcastle65`, `C-8938:MH8919176`, `C-8939:MH8919227`,
  `C-8940:MH8919332`, `C-8941:MH8919177`, `C-8942:MH8919282`,
  `C-8943:MH8919283`, `C-8944:MH9143277`, `C-8946:MH8919333`.

Membership is fixed by the formulas above; lexical donor order is used for
materialization, fitting, and reporting. Missing donors, unexpected stimuli,
panel mismatches, insufficient cells, or metadata disagreement cause refusal;
samples are never replaced after the public freeze.

## Cells and entities

For each selected sample, order cells by
`SHA256('STEPHENSON-CITESEQ-CELL-BUDGET-v1' || NUL || donor || NUL || sample || NUL || obs_name)`,
break ties by `obs_name`, and take 512. The author-supplied
`initial_clustering` annotation does not affect cell eligibility or selection.
It is mapped prospectively to eight graph strata: B cell to B; RBC to ERYTH;
HSC to HSC; CD14, CD16, DCs, Mono_prolif, and pDC to MNP; NK_16hi and NK_56hi
to NK; Plasmablast to PB; Platelets to PLT; and CD4, CD8, Lymph_prolif, MAIT,
Treg, and gdT to T.

The ordered markers are CD4, CD7, CD14, CD19, CD33, CD38, CD44, CD47, and
CD52. The deposited `var` table has no Ensembl-ID column, so each RNA feature
is resolved by one exact marker symbol with feature type `Gene Expression`;
canonical reference Ensembl IDs are recorded descriptively but do not enter
matching. Each ADT feature is one exact `AB_<marker>` alias with feature type
`Antibody Capture`. There is no fuzzy fallback. RNA state is one when the raw
Gene Expression count is positive. Within each sample and ADT marker, sort cells by raw ADT count and
then by
`SHA256('STEPHENSON-CITESEQ-ADT-v1' || NUL || donor || NUL || sample || NUL || obs_name || NUL || marker)`;
the first 256 cells are state zero and the rest state one. The 81 ordered RNA
marker by ADT marker entities are 2 by 2 linked-state tables. An entity is
informative when its margins admit more than one integer table; each scored
sample must have at least 64 informative entities. No cell-level vector is
serialized.

## Estimator and controls

For donor (d), entity (e), and fixed margins, let (A_{de}) be the
upper-left count. The primary estimator fits donor log odds
\(\theta_{de}\) and a population log odds \(\mu_e\) by the exact conditional
likelihood

\[
P(A_{de}=a\mid M_{de},\theta_{de}) \propto
  {c_0 \choose a}{c_1 \choose r_0-a}\exp(\theta_{de}a),
\]

with a donor-to-population quadratic penalty, ridge penalty on \(\mu\), and a
product-graph penalty on \(\mu\). Newton steps use the exact conditional score
and information, eliminate donor parameters by a Schur complement, and must
pass the frozen gradient and condition-number certificates. Recipient
prediction is the noncentral-hypergeometric expected table at the recipient's
own margins and the pilot-selected transport log odds \(\alpha\mu_e\); there is
no Haldane surrogate or post-fit clipping.

The frozen grid is the Cartesian product of graph neighbors `{1,2}`, donor
heterogeneity penalty `{0.1,1,10}`, ridge penalty `{0.01,0.1}`, and graph
penalty `{0,0.1,1}`, crossed with transport alpha `{0.5,0.75,1,1.25}` (144
predictive configurations from 36 structural fits). Pilot loss selects the
minimum, with lexicographic
`(loss, neighbors, heterogeneity, ridge, graph, alpha)` tie breaking.
The best graph-zero configuration is also retained as a diagnostic. Graph
superiority is reported but is not a promotion gate.

For each fit set, graph columns are all nonempty `(sample, mapped cell-type)`
strata in frozen sample and label order. RNA profiles are detection prevalence.
ADT profiles are the stratum mean of per-cell
`log1p(100 * marker_count / nine_marker_ADT_total)`, with a zero vector for a
zero-total cell. Each marker profile is centered and scaled across strata with
`ddof=1`; zero variance refuses. Euclidean directed k-nearest-neighbor edges
use marker-order tie breaking, followed by undirected union and lexical edge
order. Unweighted two-endpoint incidence matrices induce normalized
hypergraph Laplacians; the ordered-pair penalty is their Kronecker sum. The
pilot graph uses only the 12 calibration donors. After a passing pilot, the
same selected configuration is refitted without retuning and its graph rebuilt
from all 36 development donors. Held rows never enter graph construction.

The matched conventional comparator pilot-selects the strongest of raw and
exact-null-centered signed Pearson and signed-root Poisson-deviance statistics.
Each statistic is divided by the square root of the sample size and pooled
entity-wise by Paule-Mandel random effects using its exact fixed-margin null
variance. The same `{0.5,0.75,1,1.25}` transport-alpha grid is selected for the
classical family. At recipient margins the scaled pooled statistic is restored
to the target sample size and directly inverted to a fractional 2 by 2 table; an unattainable
value is projected to the nearest feasible interior boundary. The operation is
recorded for every affected entity. This comparator shares the same donors,
tables, margins, sample weighting, and pilot selection opportunity as the
primary.

The destroyed-link control orders the 512 selected cells by
`SHA256('STEPHENSON-CITESEQ-DESTROYED-LINK-v1' || NUL || sample || NUL || obs_name)`
and cyclically shifts complete ADT profiles by one position before fitting the
primary estimator with its selected configuration. It preserves every
univariate margin and destroys RNA-ADT cell linkage. Fixed-margin independence
and the pilot-selected graph-zero hierarchical estimator are reported
diagnostics.

## Development gate

Fit all primary configurations and all four classical residual variants on the
12 calibration donors, and select them on the 24-sample adaptive pilot. The
primary must beat both the strongest residual and destroyed-link controls by
all three criteria: at least 5% reduction in mean loss, upper endpoint below
zero for a 20,000-draw fixed-seed donor bootstrap 95% interval of paired loss
differences, and lower loss in at least 19 of 24 pilot donors. Failure closes
the candidate without any Newcastle matrix access. Passing freezes the chosen
settings and refits them once on all 36 development donors without retuning.

## Held access and score

Held access has two public authorization barriers. After a passing pilot result
is committed, an immutable authorization may permit a separate subprocess to
read only Newcastle RNA values and return nine 2-state margins per donor. ADT
margins remain the frozen `[256,256]`. All five method predictions for all 56
donors are then materialized, hashed, committed, and fetched byte-for-byte from
an immutable public commit. Only a second committed authorization may permit a
single pass that forms held RNA-ADT truth tables.

For observed table (T), nonnegative margin-preserving prediction
\(\widehat T\), and (N=512), entity loss is

\[
D(T,\widehat T)=\frac{2}{N}\sum_{i,j:T_{ij}>0}
T_{ij}\log\frac{T_{ij}}{\widehat T_{ij}}.
\]

Truth-positive and prediction-zero gives infinite loss; truth-zero contributes
zero. Loss is averaged equally over informative entities, then donors are
weighted equally. Against each required comparator, the primary must obtain at
least 5% lower mean loss, a donor-bootstrap upper 95% endpoint below zero, and
lower loss in at least 45 of 56 Newcastle donors. An exact one-sided binomial
sign test, with zeros unfavorable, must also give `p <= 0.025`. Both the
residual and destroyed-link comparisons must pass. Graph-versus-zero is a
reported diagnostic only.

This is a candidate-specific confirmatory test. Its p-value is not a
familywise correction across earlier public candidate searches. Any
post-attempt exception leaves a terminal one-shot record; rerunning, replacing
donors, changing thresholds, or tuning on Newcastle outcomes is forbidden.
