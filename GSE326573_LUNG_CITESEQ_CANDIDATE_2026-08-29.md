# GSE326573 lung CITE-seq held-batch candidate

## Scientific question

Can a coupling field fitted to linked RNA and surface-protein states in early acquisition batches reconstruct same-cell joint tables in later, donor-disjoint batches from recipient margins alone?

GSE326573 contains lung CD3-positive T-cell CITE-seq from controls and fibrotic lung disease. The split follows deposited batch order. Batches 1--3 contain 20 source donors. Batches 4--6 contain nine held donors after excluding CTD-ILD_2 and IPF_1, which already occur in source. Each held batch contributes three donors. CTD-ILD_3 has two technical samples; their losses are averaged before donor-level inference.

## Frozen feature map

Every deposited H5 contains the same 36,601 genes and 15 antibody features. Eleven antibodies have an unambiguous one-gene cognate in every file: CD8A--CD8, NCAM1--CD56, CD4--CD4, IL2RA--CD25, PDCD1--PD1, ITGAE--CD103, CCR7--CCR7, CTLA4--CTLA4, LAG3--LAG3, CD28--CD28, and KLRG1--KLRG1. The two deposited TotalSeq suffix styles are exact aliases fixed by metadata and axis inspection.

CD3, CD45RO, HLA-DR, and CD57 are excluded because they do not define a unique gene-level cognate. The reciprocal library-type labels for GSM9634837 and GSM9634838 conflict with their titles and file ownership. GSM9634837 is fixed as the CONTROL_9 paired count container because it owns the H5, is titled RNA, and contains both Gene Expression and Antibody Capture axes; GSM9634838 has no matrix.

## Access boundary

The 327 MB official archive and all 32 member hashes are bound in the candidate JSON. The preflight opened only H5 shape and feature-axis datasets. It did not open sparse values, indices, pointers, or barcodes and did not compute a count, pairing, association, or loss.

The executable protocol will select every estimator setting by leave-one-source-batch-out prediction, publish the final source fit before any held count access, and publish complete held predictions from margins before forming a held joint table. The mandatory comparison set is the hierarchical exact conditional field, signed Pearson and signed-root Poisson-deviance residual transfer, a stratified common-effect conditional estimate, a donor-pooled saturated Poisson interaction, a destroyed-link fit, and independence. Every supported result or refusal enters the public benchmark.
