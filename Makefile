# Name of the main file
MAIN_FILE=main
# Sets the LaTeX compiler to "xelatex"
TEX_COMPILER=xelatex
# Folder where all the necessary styles are located
# TEX_STYLES=./styles/rwu
# Main TeX File which should be used to compile
MAIN_TEX=$(MAIN_FILE).tex
# Processing files which can be cleared after the compilation
PROC_FILES= \
	*.aux \
	chapters/*.aux \
	*.lof \
	chapters/*.lof \
	*.log \
	chapters/*.log \
	*.lot \
	chapters/*.lot \
	*.nav \
	chapters/*.nav \
	*.out \
	chapters/*.out \
	*.snm \
	chapters/*.snm \
	*.toc \
	chapters/*.toc \
	*.vrb \
	chapters/*.vrb \
	*temp.tex \
	chapters/*temp.tex \
	*.run.xml \
	*-blx.bib \
	*.bbl \
	*.bcf \
	*.blg \
	*.fls \
	*.fdb_latexmk \

# Define non file targets
.PHONY: all clean clearscr build

# Default target: Calls other targets
all: build clean clearscr

# Clean target: Clean the dump files
clean:
	@echo "Cleaning Up ... "
	for i in $(PROC_FILES); do [ -f "$$i" ] && rm -f "$$i"; done; exit 0

clearscr:
	clear

# Build target: Runs first the clean target and than compiles the LaTeX files into one pdf file
# It's necessary to run the compilation process 3 times, to make sure all files are compiled accurate
build:
	@echo "Building document!"
	$(TEX_COMPILER) -interaction nonstopmode -halt-on-error $(MAIN_TEX)
	bibtex $(MAIN_FILE)
	$(TEX_COMPILER) -interaction nonstopmode -halt-on-error $(MAIN_TEX)
	$(TEX_COMPILER) -interaction nonstopmode -halt-on-error $(MAIN_TEX)
