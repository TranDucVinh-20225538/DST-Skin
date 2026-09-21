# Compile (review PDF)

The 2027 author kit was not out when this was written; `cvpr.sty` is the official 2026 kit from https://github.com/cvpr-org/author-kit with `\confYear{2027}`.

```bash
export TEXMFHOME="$(pwd)/texmf"   # only needed on this HPC (minimal TeX Live)
pdflatex -interaction=nonstopmode main
bibtex main
pdflatex -interaction=nonstopmode main
pdflatex -interaction=nonstopmode main
```

Or: `make`. Figures: `python make_figures.py` (reads frozen CSVs; does not recompute Kendall W).

Official numbers stay in `outputs/reports/OFFICIAL_W_FREEZE.txt`. Markdown twin: `../draft_v1.md`.
