# GSE342939 longitudinal RA B-cell CITE-seq protocol

## Question and pre-access boundary

This one-shot experiment asks whether RNA--protein dependence learned from
seven physical donors predicts two longitudinal recipient maps in each of six
donor-disjoint individuals. The study contains three healthy controls, five
ACPA-positive individuals who converted to clinical rheumatoid arthritis, and
five ACPA-positive non-converters. Each donor contributes paired pre and post
peripheral-B-cell GEX and CITE-seq libraries.

The candidate designation, metadata-access manifest, and pre-access amendment
form one contract. Before this freeze, the only downloaded bodies were the
official GEO SOFT record, official file inventory, official ADT feature README,
and the 104 deposited nonnumeric feature and barcode axes. Their 107 URLs,
98,414,933 bytes, and SHA-256 digests are recorded in
`metadata_access_manifest_v1.json`. No Matrix Market header or entry, BCR
file, series archive, assay value, association, prediction, or loss was read.

## Donor allocation

The split is made at the physical-donor level within healthy-control,
converter, and non-converter strata. Donors are ordered by SHA-256 of

```text
GSE342939-RA-B-CITESEQ-HELD-v1 NUL stratum NUL physical_donor
```

and the first two in each stratum are held. Both visits and both modalities
remain together. The source donors are `NN3`, `PC3`, `PC4`, `PC5`, `PN1`,
`PN2`, and `PN3`; the held donors are `NN1`, `NN2`, `PC1`, `PC2`, `PN4`, and
`PN5`. A source fold holds out one physical donor and both visits. Bootstrap
and sign-test units are also physical donors.

The candidate binds all 52 numeric Matrix Market filenames, URLs, and official
byte counts. The 28 source matrices total 814,592,401 bytes; the 24 held
matrices total 942,291,845 bytes. Those figures come from the official file
inventory and do not imply body access. The 1,924,362,240-byte series archive
and every BCR file are outside the allowlist.

## Axis pairing and panel

GEX and CITE-seq were deposited as separate MEX libraries. Every GEX feature
axis contains the same 36,617 rows. Every CITE feature axis contains the same
62 ADTs plus an exact `unmapped` sentinel. Each GEX barcode has one terminal
`-1`; CITE axes contain unsuffixed barcode cores. Within a donor-visit, pairing
removes exactly that GEX suffix and takes the exact unique intersection. Fuzzy,
cross-visit, cross-donor, and whitelist-completion matches are forbidden. The
26 observed intersections range from 392 to 19,683 barcodes and are frozen in
the metadata manifest.

The panel contains 45 one-to-one RNA/ADT cognates selected from deposited
feature definitions without assay values. The candidate records each RNA
symbol, ADT name, official ADT identifier and sequence, and complete deposited
ADT axis string. All 2,025 ordered RNA-to-ADT pairs form the coordinate
universe. CD45 isoforms, HLA-DR, CD32, class-level IgA/IgG, IgER, CD289, CCP
probes, and CD3 are excluded because the protein name does not determine one
transcript target. Axis inspection cannot add, remove, replace, or reorder a
marker after the freeze.

## Cells and binary tables

RNA-only QC requires at least 200 detected genes, mitochondrial UMI fraction
at most 0.10, and at most 70,000 GEX UMIs. Among eligible paired barcodes,
salted SHA-256 keys select the first even count up to 512; at least 128 cells
are required. Selection is performed independently within physical donor and
visit, then deposited GEX order is restored.

RNA state is raw detection. Each ADT is divided into equal low and high halves
by within-visit midrank, with a frozen salted key resolving raw-count ties.
The destroyed-link control shifts each complete 45-marker ADT state vector by
one position in salted order within donor-visit. It preserves every margin and
the multivariate ADT profile.

## Longitudinal coupling field

For donor (d), visit code (q=-1/2) before and (q=+1/2) after, and each
RNA/ADT coordinate, the primary conditional log odds are

```text
Theta[d,q] = Mu + q Delta + B[d] + q C[d].
```

`Mu` is the population-average field, `Delta` is the paired visit change,
`B[d]` is donor baseline heterogeneity, and `C[d]` is donor change
heterogeneity. Donor effects sum to zero coordinatewise. The objective is the
exact fixed-margin conditional negative log likelihood plus frozen population,
heterogeneity, and graph penalties. Separate two-nearest-neighbor RNA and ADT
graphs regularize each 45 by 45 field along its rows and columns. Every graph
is reconstructed from training donor-visits only.

For an unseen donor, the two predicted coordinates are

```text
Theta_pre  = alpha0 Mu - 0.5 alpha_delta Delta
Theta_post = alpha0 Mu + 0.5 alpha_delta Delta.
```

The baseline and visit-change multipliers each range over `0, 0.5, 1, 1.5`.
Heterogeneity penalties are `0.1, 1, 10`, population ridges are `0.01, 0.1`,
and graph penalties are `0, 0.03, 0.3`. Recipient tables are exact
noncentral-hypergeometric expectations at recipient margins. A visit-agnostic
fit with `Delta=C[d]=0` is reported as an ablation.

## Training-only masks

Every source fold constructs a new comparison mask from its six training
donors and their 12 visits. A coordinate requires at least two informative
training donors at each visit, at least three distinct donors across both
visits, finite interior common-effect estimates at each visit, and four
positive cells in each visit-specific table pooled over all training donors.
The held-out source donor contributes only recipient margins when its mask is
intersected. Its paired counts cannot change the mask. The final held mask is
rebuilt from all seven source donors only. At least 256 coordinates must be
scorable in each recipient visit, identically for every method.

## Classical head-to-head

Five controls are mandatory:

1. visit-aware selected signed Pearson or signed-root Poisson-deviance
   residual transfer;
2. the margin-preserving destroyed-link refit;
3. visit-specific common-effect exact conditional maximum likelihood;
4. visit-specific pooled saturated-Poisson fixed-interaction prediction; and
5. recipient-margin independence.

The Poisson comparator pools every training-donor table separately at pre and
post, including fixed-margin-degenerate tables. It transports the mean and
post-minus-pre log interactions. At recipient margins, row and column nuisance
parameters are refitted with that interaction fixed. This is the classical
continuous log-linear table, not a noncentral-hypergeometric expectation.
Unit transport at pooled margins must reproduce each pooled table to normalized
maximum cell error at most `1e-8`; zero transport must equal independence.

## Promotion and confirmation

For each physical source donor, loss is averaged equally over its two visit
losses. Source promotion requires complete seven-donor cross-validation, valid
numerical certificates, at least 5% lower donor-equal deviance than residual,
pooled Poisson, destroyed-link, and independence controls, paired-bootstrap
upper 95% endpoints below zero against residual, Poisson, and destroyed-link,
improvement in at least six of seven donors against each, and lower point loss
than common-effect CMLE. Bootstrap inference uses 20,000 physical-donor draws
and seed 20260830.

Held confirmation uses the same donor-level endpoint and requires at least 5%
lower loss, paired-bootstrap upper endpoints below zero, and improvement in
all six held donors against each of residual, Poisson, and destroyed-link. Six
of six gives the frozen exact one-sided sign-test value `1/64 = 0.015625`.
Primary point loss must also be below common-effect CMLE. A supported result
that misses any criterion is a completed negative result. No donor, visit,
panel member, gate, or stage may be replaced or rerun.

## Access firewall

The future runner has nine public stages: `claim-source`, `run-source`,
`claim-held-rna`, `run-held-rna`, `claim-held-adt`, `run-held-adt`,
`predict-held`, `authorize-score`, and `score-held`. Each network stage needs
an exclusive, fsync-complete attempt artifact bound by a public annotated tag.

Source failure atomically publishes its terminal result and keeps every held
matrix URL unreachable. After a public source pass, the held RNA process may
request only 12 held GEX matrices and writes selected-barcode records, sealed
RNA states, and public row margins. A separate process may then request only
12 held CITE matrices; it may read selected barcode hashes but cannot load RNA
states. Both margin processes are forbidden from importing joint-table code.

Prediction reads public source coordinates and public margins only. After its
public artifact is bound by a separate authorization, scoring runs without
network access, loads the two sealed state artifacts, forms held joint tables
once, and publishes one terminal result. Every canonical artifact is written
by exclusive temporary creation, file and directory fsync, and atomic rename.
Partial failures retain the identity and deletion record of every requested
file, including the failing file.

## Freeze state

This task creates the designation, manifest, amendment, protocol, and
adversarial contract tests only. It does not create public tags, numeric-access
attempts, source or held outcomes, predictions, or score authorization. Numeric
matrix access remains forbidden until the complete implementation and its
tests are independently reviewed, checksum-bound, committed, pushed, and
published under the frozen barrier tags.
