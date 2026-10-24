LATEXMK ?= latexmk
LATEXFLAGS ?= -pdf -interaction=nonstopmode -halt-on-error

.PHONY: all pdf clean

all: pdf

pdf:
	$(LATEXMK) $(LATEXFLAGS) main.tex

# Rimuove solo i file intermedi, mantenendo il PDF compilato.
clean:
	$(LATEXMK) -c main.tex
