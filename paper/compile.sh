#!/bin/bash
# LaTeX Compilation Script for DynaTree Paper

set -e  # Exit on error

echo "=========================================="
echo "  DynaTree Paper Compilation"
echo "=========================================="

cd "$(dirname "$0")"

# Clean old auxiliary files
echo "[1/6] Cleaning old files..."
rm -f *.aux *.log *.out *.toc *.bbl *.blg *.synctex.gz *.fdb_latexmk *.fls

# First compilation
echo "[2/6] First compilation pass..."
pdflatex -interaction=nonstopmode dynatree.tex > /dev/null 2>&1 || {
    echo "⚠️  First pass completed with warnings (this is normal)"
}

# BibTeX (for bibliography)
echo "[3/6] Running bibtex..."
bibtex neurips_2025 > /dev/null 2>&1 || {
    echo "⚠️  BibTeX completed with warnings"
}

# Second compilation (for references)
echo "[4/6] Second compilation pass (resolving references)..."
pdflatex -interaction=nonstopmode dynatree.tex > /dev/null 2>&1 || {
    echo "⚠️  Second pass completed with warnings"
}

# Third compilation (finalize references)
echo "[5/6] Third compilation pass (finalizing references)..."
pdflatex -interaction=nonstopmode dynatree.tex > /dev/null 2>&1 || {
    echo "⚠️  Third pass completed with warnings"
}

# Check result
echo "[6/6] Checking output..."
if [ -f dynatree.pdf ]; then
    FILE_SIZE=$(ls -lh dynatree.pdf | awk '{print $5}')
    PAGE_COUNT=$(pdfinfo dynatree.pdf 2>/dev/null | grep "Pages:" | awk '{print $2}')
    echo ""
    echo "=========================================="
    echo "✓ Compilation successful!"
    echo "=========================================="
    echo "  File: dynatree.pdf"
    echo "  Size: $FILE_SIZE"
    echo "  Pages: $PAGE_COUNT"
    echo ""
    echo "To view the PDF:"
    echo "  xdg-open dynatree.pdf"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "✗ Compilation failed!"
    echo "=========================================="
    echo "Check the log file for errors:"
    echo "  less neurips_2025.log"
    echo ""
    exit 1
fi

