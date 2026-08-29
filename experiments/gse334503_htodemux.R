#!/usr/bin/env Rscript

# Minimal HTODemux implementation for the GSE334503 source reduction.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("usage: gse334503_htodemux.R COUNTS_TSV CUTOFFS_TSV")
}
if (!requireNamespace("cluster", quietly = TRUE)) {
  stop("R package 'cluster' is required")
}
if (!requireNamespace("MASS", quietly = TRUE)) {
  stop("R package 'MASS' is required")
}

counts <- as.matrix(read.table(
  args[[1]], header = FALSE, sep = "\t", check.names = FALSE,
  colClasses = "integer"
))
storage.mode(counts) <- "integer"
if (nrow(counts) != 12 || ncol(counts) <= 13) {
  stop("HTODemux expects 12 sample HTOs and more than 13 cells")
}
if (anyNA(counts) || any(counts < 0)) {
  stop("HTO counts must be nonnegative integers")
}

# Seurat NormalizeData(..., normalization.method = "CLR", margin = 1).
clr <- t(apply(counts, 1, function(values) {
  denominator <- exp(sum(log1p(values[values > 0])) / length(values))
  log1p(values / denominator)
}))

set.seed(42)
clusters <- cluster::clara(t(clr), k = 13, samples = 100)$clustering
cluster_ids <- sort(unique(clusters))
if (length(cluster_ids) != 13) {
  stop("CLARA did not return all 13 requested clusters")
}

# Seurat AverageExpression exponentiates the log-normalized data before
# averaging. For feature-wise CLR this is proportional to the raw-count mean.
average_expression <- sapply(cluster_ids, function(cluster_id) {
  rowMeans(expm1(clr[, clusters == cluster_id, drop = FALSE]))
})
if (any(average_expression == 0)) {
  stop("Cells with zero counts exist as a cluster")
}

rows <- vector("list", nrow(counts))
for (tag_index in seq_len(nrow(counts))) {
  negative_column <- which.min(average_expression[tag_index, ])
  negative_cluster <- cluster_ids[[negative_column]]
  background <- counts[tag_index, clusters == negative_cluster]
  fit <- suppressWarnings(MASS::fitdistr(
    background, densfun = "Negative Binomial"
  ))
  size <- unname(fit$estimate[["size"]])
  mu <- unname(fit$estimate[["mu"]])
  cutoff <- stats::qnbinom(0.99, size = size, mu = mu)
  if (!is.finite(size) || !is.finite(mu) || !is.finite(cutoff)) {
    stop("negative-binomial HTO fit is non-finite")
  }
  rows[[tag_index]] <- data.frame(
    tag_index = tag_index - 1L,
    negative_cluster = negative_cluster,
    background_cells = length(background),
    size = size,
    mu = mu,
    cutoff = as.integer(cutoff)
  )
}

output <- do.call(rbind, rows)
con <- file(args[[2]], open = "wt")
writeLines(c(
  paste("# r_version", R.version.string, sep = "\t"),
  paste("# cluster_version", as.character(packageVersion("cluster")), sep = "\t"),
  paste("# mass_version", as.character(packageVersion("MASS")), sep = "\t")
), con = con)
write.table(
  output, file = con, sep = "\t", quote = FALSE, row.names = FALSE,
  col.names = TRUE
)
close(con)
