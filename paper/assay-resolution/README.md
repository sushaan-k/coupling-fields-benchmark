# Assay resolution governs the transfer of molecular dependence

- `reader.pdf`: two-column paper.
- `main.pdf`: line-numbered submission layout.
- `supplement.pdf`: Additional file 1, with proofs, protocols, and earlier benchmark results.

The PDFs retain the blinded title page and deferred author declarations.
Public author names and contact details appear on the paper page.

## Build

From this directory, with a current TeX Live installation:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error reader.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex
```

`main.tex` is the shared manuscript source. The figures use TikZ and PGFPlots;
their coordinates are embedded in `resolution_figure.tex` and
`relationship_figure.tex`. The supplement includes `resolution_methods.tex`
and preserves the original benchmark protocols and outcomes.

This directory contains the manuscript, not a new executable analysis release.
The resolution and query-ranking analyses are retrospective and are not part
of the [original v2.0.4 benchmark](https://github.com/sushaan-k/coupling-fields-benchmark/releases/tag/coupling-fields-v2.0.4-public-benchmark).
