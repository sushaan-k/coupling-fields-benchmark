args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("usage: export_gse314416_metadata.R INPUT_RDS OUTPUT_TSV")
}

metadata <- readRDS(args[[1L]])
required <- c("orig.ident", "sample", "timepoint")
if (!is.data.frame(metadata) || !all(required %in% colnames(metadata))) {
  stop("baseline metadata does not contain the required columns")
}
if (anyDuplicated(rownames(metadata)) != 0L) {
  stop("baseline metadata cell identifiers are not unique")
}

output <- data.frame(
  cell_id = rownames(metadata),
  well = metadata[["orig.ident"]],
  donor = metadata[["sample"]],
  timepoint = metadata[["timepoint"]],
  stringsAsFactors = FALSE
)
write.table(
  output,
  file = args[[2L]],
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  col.names = TRUE
)
