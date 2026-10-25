LATEXMK ?= latexmk
LATEXFLAGS ?= -pdf -interaction=nonstopmode -halt-on-error

.PHONY: all pdf
all: pdf
pdf:
	$(LATEXMK) $(LATEXFLAGS) main.tex
