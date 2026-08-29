# Kotliarov PBMC binary coupling-field source-development protocol

## Status and scope

This document freezes an outcome-blind replacement candidate after the
multistate Kotliarov campaign ended at RNA lineage support. That terminal
record states that the ADT count dataset was not opened, no RNA--ADT pairing
was formed, and no held endpoint was evaluated. The replacement changes the
estimand before ADT access: it uses binary RNA--protein tables in all selected
PBMCs and does not reuse the failed lineage requirement.

The source-development phase may read paired RNA and ADT values only for the
10 batch-1 development donors. It must stop before held ADT access unless the
source gate below passes. No held donor may enter model selection.

## Frozen cohort

- Development, batch 1: 200, 207, 212, 233, 237, 245, 256, 261, 273, 277.
- Held, batch 2: 201, 205, 215, 229, 234, 236, 250, 268, 279.
- Donor 209 is excluded because it spans both batches.
- The biological replication unit is the donor.

Each donor contributes 512 deposited concordant singlets selected without
RNA or ADT values. Cells are ordered by SHA-256 of the fixed salt
`KOTLIAROV-PBMC-BINARY-CELL-v2`, donor identifier, and deposited barcode;
the first 512 are retained.

## Frozen entities and states

The ordered cognate panel is:

1. CD3D--CD3
2. CD4--CD4
3. CD8A--CD8
4. MS4A1--CD20
5. CD14--CD14
6. FCGR3A--CD16
7. NCAM1--CD56
8. HLA-DRA--HLA-DR
9. IL7R--CD127

All 81 ordered RNA-to-protein pairs are evaluated. RNA state is raw detection.
Within each donor, each protein is divided into 256 low and 256 high cells by
count and the fixed tie salt `KOTLIAROV-PBMC-BINARY-ADT-v2`. The destroyed-link
control circularly shifts complete protein-state profiles after ordering cells
with `KOTLIAROV-PBMC-BINARY-DESTROYED-v2`; every RNA and protein margin is
therefore unchanged.

## Source-only model selection

Leave-one-development-donor-out prediction selects configurations by mean
donor-equal multinomial deviance per cell. Ties use lexicographic configuration
order. Every method receives the same target margins and support mask.

- Hierarchical exact conditional log odds: heterogeneity penalty in
  {0.1, 1, 10}, ridge penalty in {0.01, 0.1}, graph penalty 0, and transport
  multiplier in {0.5, 0.75, 1, 1.25}.
- Classical residual: raw or exact-null-centered signed Pearson or signed-root
  Poisson-deviance coordinate, with the same transport grid.
- Common-effect exact conditional maximum likelihood, with the same transport
  grid.
- Pooled saturated Poisson log-linear interaction, with the same transport
  grid.
- Fixed-margin independence and the destroyed-link fit are mandatory controls.

All 81 coordinates must have at least two informative source donors in every
fold. A fit that misses its numerical certificate is refused; configurations
must complete every fold to be selectable.

## Source promotion gate

The candidate proceeds only if the selected hierarchical estimator satisfies
all of the following in the 10 source-held folds:

1. at least 5% lower mean loss than the selected classical residual;
2. at least 5% lower mean loss than the better of common-effect conditional
   maximum likelihood and pooled saturated Poisson;
3. at least 5% lower mean loss than the destroyed-link fit;
4. for each comparison, the upper endpoint of a 20,000-draw paired donor
   bootstrap interval for primary minus comparator loss is below zero;
5. for each comparison, at least 8 of 10 donors favor the primary estimator;
6. every source, support, pairing, optimization, and reconstruction check
   passes.

Bootstrap draws use seed 20260829. Failure of any condition is a terminal
source-only refusal; held ADT values remain unopened.

## Held phase if promoted

After a source pass, the selected configurations are refitted on all 10
development donors. Held RNA margins and frozen predictions are then published
with checksums before any held ADT value is opened. Held scoring is one-shot.
Confirmation requires at least 5% lower mean loss, a paired-bootstrap upper
endpoint below zero, at least 8 of 9 favorable donors, and an exact one-sided
binomial sign-test P <= 0.025 against each of the selected residual, the better
fitted classical interaction estimator, and destroyed links. Results against
both fitted classical estimators are reported regardless of the gate.

## Public inputs

- RNA matrix: 82,850,550 bytes, MD5
  `8778f578eced043bd993a474c8919139`.
- ADT matrix: 4,708,920 bytes, MD5
  `d74235cc89e0c8edbd9637731e53c8d6`.
- Column metadata SHA-256:
  `9de81ec194efda0a81937d7524e07de327a65a3c510909f21195e633f1af8470`.
- RNA feature metadata SHA-256:
  `a59b42b01d128476b0bca8dd90c62dcd3b0533a4e9c114b457a907b5df716998`.
- ADT feature metadata SHA-256:
  `5122b7c387908552517fa88d39e8c4edb71726aeac3bb082cf4cb1aebd5f1866`.

The matrices are the public `KotliarovPBMCData` ArtifactDB assets associated
with DOI 10.1038/s41591-020-0769-8.
