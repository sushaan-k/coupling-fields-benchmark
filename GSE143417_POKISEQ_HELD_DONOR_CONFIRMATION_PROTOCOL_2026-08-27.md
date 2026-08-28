# GSE143417 PoKI-seq held-donor confirmation protocol

**Version:** `gse143417-pokiseq-held-donor/1.0`  
**Frozen locally:** 27 August 2026, before acquisition or inspection of the
deposited expression matrices  
**Status:** prospective local specification; not an external preregistration

Outcome access requires a full Git commit and commit-bound public URL that
contain this protocol, the runner, both sealing tools, all implementation
modules, tests, source designation, and the still-disabled preanalysis lock.
The authorized lock records that public freeze before any expression matrix is
opened.

## Question and sealed unit

The experiment asks whether perturbation-specific dependence between an
endogenous proliferation program and an effector program transfers from one
human donor to another under TGF-beta challenge. GSE143417 contains a 36-
construct pooled knock-in screen in primary T cells from two donors, each
measured after TCR stimulation with or without TGF-beta in three technical
replicates.

Donor1 is the sole development unit. Donor2 is the held confirmation unit.
Technical replicates are pooled within donor and context after retaining cells
with one unique construct assignment. They are not treated as biological
replicates.

The official `GSE143417_RAW.tar` is 492,216,320 bytes with SHA-256
`6bc8bf810fbca8f0585c337ed143d39d8bfbc3f85d623894ebadf4c6f357b632`.
Only GSM4259039 through GSM4259062 are used. Consecutive GEX and TCR samples
are paired by replicate, and terminal 16-base 10x barcodes are joined within
each library.

## Eligibility and programs

A construct is eligible only when it has at least 30 mono-construct cells in
each donor-by-context arm. The metadata-only audit found 35 of 36 constructs
eligible; `CTLA4dn` is excluded. GFP and mCherry form the pooled control and
are not query constructs, leaving 33 queries.

The two prespecified scores are averages of library-size-normalized `log1p`
expression:

- proliferation: `MKI67`, `TOP2A`, `CDK1`;
- effector: `GZMA`, `FASLG`, `IFNG`.

Each score is discretized into three states. The one-third and two-third cuts
are fitted only in Donor1 Stim cells carrying GFP or mCherry. Those four cuts
are then applied unchanged to every construct, context, and Donor2 cell.
Tied cuts, a missing marker, a missing state margin, or a support failure
produces a refusal. Before fitting, every eligible construct-by-donor-by-
context arm must place at least 5% of its cells in each state of both programs;
this margin-only occupancy check cannot remove an individual construct and a
failure refuses the experiment.

## Estimand and centering

For each construct and arm, linked cells form a 3-by-3 proliferation-by-
effector table. The coupling field is the double-centered log table with a 0.5
pseudocount, expressed in four Helmert interaction coordinates. The primary
estimand is the control-centered factorial contrast

`[construct TGFB - construct Stim] - [pooled control TGFB - pooled control Stim]`.

Every arm is centered on 64 fixed-margin permutations plus one disjoint
destroyed-link draw. Canonical row and column state vectors make the null mean
a deterministic function of the two margins, table size, and seed. The same
permutations, rows, contrasts, and donor split are used for the field and both
classical residual families.

## Frozen estimators

The primary estimator applies inverse permutation-variance weights, a nuclear
penalty equal to 0.1 times the leading development singular value, and graph
penalty 5 on a typed construct-architecture hypergraph. Its frozen components
are receptor families BTLA, FAS, PD1, TGFBR2, TIGIT, TIM3, CTLA4, and IL7R,
and shared signaling domains `dn`, CD28, 41BB, ICOS, MYD88, IL7RA, OX40, and
CD3Z. A component becomes a hyperedge only when at least two deposited labels
contain it; unique cargos remain isolates.

Matched field comparators are zero contrast, direct Donor1 contrast,
variance-derived scalar shrinkage, nuclear-only shrinkage, hypergraph-only
shrinkage, fixed endpoint ridge (`alpha=0.1`), the destroyed-link contrast,
and a label-permuted hypergraph. The primary classical baseline retains all
nine entries of signed Pearson or signed Poisson-deviance residual matrices
from the row-plus-column independence model. Direct and identically structured
versions of both residuals are evaluated. Helmert-projected residuals are not
primary because that projection discards residual components.

## Pairing seal and common-table endpoint

Production execution has three irreversible analysis stages.

1. `predict` may use Donor2 target Stim pairs, pooled-control Stim and TGFB
   pairs, and the separate margins of each Donor2 target TGFB table. It writes
   every coordinate and joint-table prediction before target TGFB pairing is
   opened.
2. The prediction JSON is committed to the public benchmark repository.
   `authorize_poki_gse143417_scoring.py` then requires that immutable commit
   and blob URL and seals them with the exact prediction SHA-256,
   cache SHA-256, outcome-lock SHA-256, runner, field, residual, table-scoring,
   and protocol bytes in a separate `SEALED_FOR_SCORING` designation.
3. The designation is published in a later commit. `score` requires its exact
   prediction hash, full 40-character public commit, and commit-bound GitHub
   URL. Only then does it form Donor2 target TGFB joint tables and score them.

For every family, the held target Stim interaction is the calibration anchor.
The predicted factorial contrast and observed pooled-control TGFB-minus-Stim
interaction are added to it. The target TGFB null-mean correction is computed
from margins and frozen seeds. All methods receive the same observed target
TGFB row and column margins.

Coupling coordinates are lifted to log interactions, exponentiated, and fitted
to those margins by iterative proportional fitting. Full Pearson residuals use
`N*=E+r*sqrt(E)`. Full signed-deviance residuals use the sign-compatible inverse
of the Poisson deviance equation. A fixed `1e-3` mean-cell-count floor prevents
nearly decomposable residual seeds before the same margin fit. The primary loss
is multinomial deviance per held cell on the identical target TGFB joint table.

## Uncertainty and pass rule

Exactly 2,000 paired query-construct bootstrap draws use NumPy
`default_rng(20260827)`. Complete construct blocks are resampled across every
method and endpoint. Cells and technical replicates are not resampling units.

The confirmation passes only if all four conditions hold:

1. the lower 95% bootstrap endpoint for pooled held coupling-field correlation
   is above zero;
2. the upper 95% endpoint for primary-minus-destroyed common-table deviance is
   below zero;
3. the upper 95% endpoint for primary-minus-best-classical deviance is below
   zero, where best classical is reselected within each draw from direct and
   structured Pearson and deviance residuals; and
4. the upper 95% endpoint for primary-minus-best-matched-field deviance is
   below zero, with zero, direct, scalar, nuclear-only, hypergraph-only,
   endpoint, and label-permuted predictions re-competed within each draw.

The TGFBR2-derived `TGFbR2dn`, `TGFbR241BB`, and `TGFbR2Myd88` constructs are
the prespecified positive-control class. `tNGFR` is the prespecified
falsification target. Their results are secondary and cannot change the pooled
pass rule.

Any refusal or failed criterion remains in the benchmark. A change after
outcome authorization is labeled post-lock sensitivity analysis and cannot
replace the frozen result.
