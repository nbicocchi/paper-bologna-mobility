LATEXMK ?= latexmk
LATEXFLAGS ?= -pdf -interaction=nonstopmode -halt-on-error
PYTHON ?= python3

.PHONY: all pdf experiments reproduce
all: pdf
pdf:
	$(LATEXMK) $(LATEXFLAGS) main.tex

experiments:
	$(PYTHON) ../sharing-code/code/run.py

reproduce: experiments
	$(MAKE) pdf
