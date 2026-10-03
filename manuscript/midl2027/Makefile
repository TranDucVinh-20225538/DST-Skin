TEXMFHOME ?= $(abspath texmf)
export TEXMFHOME
LATEX = pdflatex -interaction=nonstopmode
BIBTEX = bibtex

.PHONY: all figures main clean

all: figures main

figures:
	../../.conda-env/bin/python make_figures.py

main: main.pdf
main.pdf: main.tex refs.bib midl.cls fig/fig1_hook.pdf fig/fig2_rank_heatmaps.pdf fig/fig3_coverage_signflip.pdf fig/figA_seed_jumps.pdf
	$(LATEX) main </dev/null
	$(BIBTEX) main
	$(LATEX) main </dev/null
	$(LATEX) main </dev/null

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.synctex.gz
