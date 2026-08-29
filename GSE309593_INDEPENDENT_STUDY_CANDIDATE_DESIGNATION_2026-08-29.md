# GSE309593 independent-study candidate designation

## Frozen question

Can a coupling field developed entirely in GSE288020 predict RNA--protein joint tables in an independent BMMC CITE-seq study before any GSE309593 RNA--ADT pairing is constructed? GSE309593 is the recipient study; it contributes no model, hyperparameter, comparator, threshold, or marker selection.

The final source fit may use all 23 GSE288020 subjects only after that study's calibration, pilot, and internal test have been publicly closed. The recipient panel is the exact schema intersection of the ordered, unambiguous cognate list in `candidate_designation_v1.json`. Fewer than nine retained cognates is a terminal refusal.

## Frozen recipient cohort

GEO MINiML identifies 23 distinct, non-bridge, bone-marrow samples at `MM Pre-Treatment`, one per subject:

| Batch | Subjects |
|---|---|
| B092 | FH1003, FH1004, FH1007 |
| B099 | FH1001, FH1002, FH1005, FH1009 |
| B110 | FH1006, FH1008, FH1010, FH1011, FH1012 |
| B129 | FH1014, FH1017 |
| B162 | FH1016, FH1020, FH1021 |
| B208 | FH1022, FH1024 |
| B210 | FH1023, FH1026, FH1027, FH1028 |

The designation binds each subject to one RNA H5 and one ADT CSV by GSM, file name, byte count, and the canonical GEO per-sample URL template. These 23 RNA files total 2,201,943,182 bytes; the ADT files total 28,358,247 bytes. Files will be processed one subject at a time.

## Access barrier

1. Publish and independently verify the annotated designation tag.
2. Close GSE288020 development and publish the fixed primary and comparator configurations.
3. Process recipient RNA H5 files without downloading ADT files. Freeze RNA support, selected identifiers, states, and margins.
4. Process recipient ADT files in a program that cannot read recipient RNA states. Freeze ADT ranks and margins.
5. Publish reconstructed predictions from recipient margins before forming any recipient RNA--ADT table.
6. Join states and score once only after independent verification of the prediction tag.

Any failure of the marker, file, subject, batch, common-identifier, informative-table, or numerical gates ends in a terminal refusal. A supported analysis that misses an outcome criterion is published as a negative result.

## Matched comparisons

The held score includes the hierarchical exact conditional estimator, the GSE288020-selected signed-root Poisson-deviance residual, a frozen destroyed-source-link fit, a stratified common exact conditional estimate, and a donor-pooled saturated-Poisson interaction estimate. Every method uses the same source subjects, marker intersection, recipient subjects, cells, margins, and deviance. The estimator-level increment over either fitted classical interaction is claimed only if its point loss is lower and its paired-subject interval excludes zero.

## Metadata evidence and access attestation

| Binding | Bytes | SHA-256 |
|---|---:|---|
| [GEO MINiML](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE309nnn/GSE309593/miniml/GSE309593_family.xml.tgz) | 15,035 | `df4c78cb4da016db22c2d96326aee0d0762807cfe05ef6ca36a057a71cf43f3a` |
| [GEO file inventory](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE309nnn/GSE309593/suppl/filelist.txt) | 21,249 | `2d158e89ac5e15d084ad50f06c3bd9b41ae29f7e730256d582f2d1f0421759b2` |
| [Allen BMMC panel and downloads](https://apps.allenimmunology.org/aifi/insights/ndmm/downloads/scrna/) | 66,182 | `2bf96574a077f7ed440f6c9e806599768154f2448b6864d6c386c6f939e393ab` |

The [GEO series record](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE309593) and [Allen external-repository page](https://apps.allenimmunology.org/aifi/insights/ndmm/downloads/external/) are the descriptive authorities. Before this designation, only the three metadata objects above and public descriptive pages were retrieved. No GSE309593 H5, ADT CSV, assay value, cell identifier, or barcode was downloaded or read.
