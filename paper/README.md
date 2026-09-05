# Manuscript

`main.tex` is the manuscript and `supplement.tex` is Additional file 1.
The numerical macros used by these sources are in
`manuscript_results_macros.tex`. The older `results_macros.tex` and
`generated_comparator_tables.tex` remain unchanged for historical result tests.

Build from this directory with a LaTeX installation that includes pdfLaTeX
and BibTeX. For the two-column reader:

```bash
pdflatex -jobname=reader '\def\ReaderVersion{1}\input{main.tex}'
bibtex reader
pdflatex -jobname=reader '\def\ReaderVersion{1}\input{main.tex}'
pdflatex -jobname=reader '\def\ReaderVersion{1}\input{main.tex}'
```

For the line-numbered submission, run pdfLaTeX on `main.tex`, BibTeX on
`main`, and pdfLaTeX twice more. Build `supplement.tex` the same way with
job name `supplement`.

The two vector figures are supplied in `standalone_figures/`. The current
workflow's TikZ source is in `figures/`. Compiled reader and supplement PDFs
are distributed under `docs/assets/papers/`. The manuscript retains its
blinded author and declaration fields; the project website lists the authors.
