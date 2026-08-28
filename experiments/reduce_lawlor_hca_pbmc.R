#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(Matrix))

expected <- c(
  rna = "c6c1b29e8737085d89f2af35c23fb9ce0e189369e65061f18cfe8532bd1e80c5",
  adt = "bab0c4c2b43bb2c8c942491a370623b0685ef0dd10ce58aa45734371a8fb59e5",
  annotations = "76a6548089867687461603ed0031ba02f9203885cd8a623b7497bdceccaf7229"
)

parse_args <- function(values) {
  if (length(values) %% 2 != 0) stop("arguments must be --name value pairs")
  result <- list()
  for (index in seq(1, length(values), by = 2)) {
    key <- sub("^--", "", values[[index]])
    result[[key]] <- values[[index + 1]]
  }
  required <- c("rna", "adt", "annotations", "aliases", "output")
  missing <- required[!required %in% names(result)]
  if (length(missing)) stop(paste("missing arguments:", paste(missing, collapse = ", ")))
  result
}

sha256 <- function(path) {
  output <- system2("shasum", c("-a", "256", shQuote(path)), stdout = TRUE)
  if (length(output) != 1) stop(paste("failed to hash", path))
  strsplit(output, "[[:space:]]+")[[1]][1]
}

extract_matrix <- function(object, label) {
  if (inherits(object, "Matrix") || is.matrix(object)) return(as(object, "dgCMatrix"))
  if (!is.list(object)) stop(paste(label, "RDS is not a matrix or supported list"))
  preferred <- c("counts", "matrix", "data", "RNA", "ADT")
  candidates <- preferred[preferred %in% names(object)]
  candidates <- candidates[vapply(object[candidates], function(x) inherits(x, "Matrix") || is.matrix(x), logical(1))]
  if (length(candidates) != 1) stop(paste(label, "RDS does not contain exactly one supported matrix"))
  as(object[[candidates]], "dgCMatrix")
}

orient_cells <- function(matrix, cell_ids, label) {
  row_overlap <- if (is.null(rownames(matrix))) 0 else sum(rownames(matrix) %in% cell_ids)
  col_overlap <- if (is.null(colnames(matrix))) 0 else sum(colnames(matrix) %in% cell_ids)
  if (row_overlap == col_overlap) stop(paste(label, "cell axis is ambiguous"))
  if (row_overlap > col_overlap) matrix <- t(matrix)
  if (is.null(colnames(matrix)) || is.null(rownames(matrix))) stop(paste(label, "matrix lacks dimnames"))
  as(matrix, "dgCMatrix")
}

normalize_token <- function(value) {
  token <- trimws(toupper(value))
  token <- sub("^((ANTI[-_. ]*HUMAN|ADT|PROT|CITE)[-_. ]*)+", "", token)
  token <- sub("[-_. ]*TOTALSEQ[A-Z0-9]*$", "", token)
  gsub("[^A-Z0-9]", "", token)
}

write_mtx_gz <- function(matrix, path) {
  plain <- sub("[.]gz$", "", path)
  writeMM(as(matrix, "dgTMatrix"), plain)
  status <- system2("gzip", c("-f", shQuote(plain)))
  if (status != 0) stop(paste("gzip failed for", plain))
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
for (name in names(expected)) {
  if (!file.exists(args[[name]])) stop(paste("missing", name, "file"))
  observed <- sha256(args[[name]])
  if (observed != expected[[name]]) stop(paste(name, "SHA-256 mismatch"))
}
if (!file.exists(args$aliases)) stop("alias table is missing")

annotations <- read.csv(args$annotations, row.names = 1, check.names = FALSE, stringsAsFactors = FALSE)
eligible <- annotations$HTO_Classification %in% c("Baseline", "LPS", "CD3_CD28") &
  annotations$Demuxlet_Classification == "SNG" &
  !is.na(annotations$Celltype_Annotation)
annotations <- annotations[eligible, , drop = FALSE]

grid <- data.frame(
  condition = c("CD3_CD28", "CD3_CD28", "CD3_CD28", "CD3_CD28", "CD3_CD28", "LPS"),
  cell_type = c("B", "CD4T_Naive", "CD4T_Mem", "CD8T_Naive", "CD8T_Mem", "CD14_Mono"),
  stringsAsFactors = FALSE
)
keep <- annotations$HTO_Classification == "Baseline"
for (index in seq_len(nrow(grid))) {
  keep <- keep | (
    annotations$HTO_Classification == grid$condition[[index]] &
      annotations$Celltype_Annotation == grid$cell_type[[index]]
  )
}
keep <- keep & annotations$Celltype_Annotation %in% unique(grid$cell_type)
annotations <- annotations[keep, , drop = FALSE]

rna <- orient_cells(extract_matrix(readRDS(args$rna), "RNA"), rownames(annotations), "RNA")
adt <- orient_cells(extract_matrix(readRDS(args$adt), "ADT"), rownames(annotations), "ADT")
common <- rownames(annotations)[rownames(annotations) %in% colnames(rna) & rownames(annotations) %in% colnames(adt)]
if (length(common) < 0.95 * nrow(annotations)) stop("fewer than 95% of eligible cells match both matrices")
annotations <- annotations[common, , drop = FALSE]
rna <- rna[, common, drop = FALSE]
adt <- adt[, common, drop = FALSE]

aliases <- read.delim(args$aliases, stringsAsFactors = FALSE)
if (anyDuplicated(aliases$adt_token)) stop("alias table has duplicate ADT tokens")
alias_tokens <- vapply(aliases$adt_token, normalize_token, character(1))
if (anyDuplicated(alias_tokens)) stop("alias tokens are ambiguous after normalization")
alias_genes <- toupper(aliases$gene_symbol)
if (anyDuplicated(alias_genes)) stop("alias table maps multiple ADT tokens to one RNA gene")
alias_lookup <- setNames(aliases$gene_symbol, alias_tokens)
rna_names <- toupper(rownames(rna))
if (anyDuplicated(rna_names)) stop("RNA gene symbols are ambiguous after case normalization")
rna_lookup <- setNames(seq_len(nrow(rna)), rna_names)
tokens <- vapply(rownames(adt), normalize_token, character(1))
genes <- unname(alias_lookup[tokens])
matched <- !is.na(genes) & toupper(genes) %in% names(rna_lookup)
if (sum(matched) < 12) stop("fewer than 12 ADT features have a sealed RNA match")

adt_rows <- which(matched)
rna_rows <- unname(rna_lookup[toupper(genes[matched])])
marker <- data.frame(
  marker_id = paste0(rownames(adt)[adt_rows], "::", rownames(rna)[rna_rows]),
  adt_feature = rownames(adt)[adt_rows],
  gene_symbol = rownames(rna)[rna_rows],
  adt_row = adt_rows - 1,
  stringsAsFactors = FALSE
)
if (anyDuplicated(marker$marker_id)) stop("matched marker identifiers are not unique")

dir.create(args$output, recursive = TRUE, showWarnings = FALSE)
rna_total <- Matrix::colSums(rna)
cell <- data.frame(
  cell_id = common,
  donor = annotations$Donor_of_Origin,
  condition = annotations$HTO_Classification,
  cell_type = annotations$Celltype_Annotation,
  rna_total = as.numeric(rna_total),
  stringsAsFactors = FALSE
)
write.table(cell, gzfile(file.path(args$output, "cells.tsv.gz")), sep = "\t", row.names = FALSE, quote = FALSE)
write.table(marker, file.path(args$output, "markers.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
write.table(data.frame(adt_feature = rownames(adt)), file.path(args$output, "adt_features.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
write_mtx_gz(rna[rna_rows, , drop = FALSE], file.path(args$output, "rna_matched.mtx.gz"))
write_mtx_gz(adt, file.path(args$output, "adt_all.mtx.gz"))

outputs <- c("cells.tsv.gz", "markers.tsv", "adt_features.tsv", "rna_matched.mtx.gz", "adt_all.mtx.gz")
manifest <- data.frame(
  path = outputs,
  bytes = vapply(file.path(args$output, outputs), file.size, numeric(1)),
  sha256 = vapply(file.path(args$output, outputs), sha256, character(1)),
  stringsAsFactors = FALSE
)
write.table(manifest, file.path(args$output, "reducer_manifest.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
