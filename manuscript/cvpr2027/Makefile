TEXMFHOME ?= $(abspath texmf)
export TEXMFHOME
LATEX = pdflatex -interaction=nonstopmode
BIBTEX = bibtex

.PHONY: all figures main suppl clean

all: figures main suppl

figures:
	../../.conda-env/bin/python make_figures.py

main: main.pdf
main.pdf: main.tex refs.bib cvpr.sty fig/fig1_msp_bars.pdf fig/fig2_rank_heatmaps.pdf fig/fig3_coverage_signflip.pdf
	$(LATEX) main
	$(BIBTEX) main
	$(LATEX) main
	$(LATEX) main

suppl: suppl.pdf
suppl.pdf: suppl.tex cvpr.sty fig/figA_seed_jumps.pdf
	$(LATEX) suppl
	$(LATEX) suppl

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.synctex.gz
