#!/bin/bash

# Generate SVG, PDF, and PNG diagrams from D2 files
# Usage: ./generate_d2_diagrams.sh

cd "$(dirname "$0")/diagrams"

# Create output directories
mkdir -p svg pdf png

for d2file in *.d2; do
    if [ -f "$d2file" ]; then
        name="${d2file%.d2}"
        echo "Generating $name..."

        # SVG (vector - best for web)
        d2 --theme 200 --layout elk "$d2file" "svg/$name.svg"

        # PDF (vector - best for print/slides)
        d2 --theme 200 --layout elk "$d2file" "pdf/$name.pdf"

        # PNG (raster - fallback)
        d2 --theme 200 --layout elk "$d2file" "png/$name.png"
    fi
done

echo ""
echo "Done! Generated diagrams in svg/, pdf/, png/ directories."
echo "SVG count: $(ls -1 svg/*.svg 2>/dev/null | wc -l)"
echo "PDF count: $(ls -1 pdf/*.pdf 2>/dev/null | wc -l)"
echo "PNG count: $(ls -1 png/*.png 2>/dev/null | wc -l)"
