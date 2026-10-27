LATEXMK ?= latexmk
PYTHON ?= python3
LATEXFLAGS ?= -pdf -interaction=nonstopmode -halt-on-error

.PHONY: all pdf clean

all: pdf

pdf:
	$(PYTHON) count_words.py
	$(LATEXMK) $(LATEXFLAGS) main.tex

# Rimuove solo i file intermedi, mantenendo il PDF compilato.
clean:
	$(LATEXMK) -c main.tex
