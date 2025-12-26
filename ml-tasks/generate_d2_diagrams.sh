#!/bin/bash
# Generate SVG, PDF, and PNG from D2 diagram files

cd "$(dirname "$0")/diagrams"

# Create output directories
mkdir -p svg pdf png

# Generate diagrams in all formats
for d2file in *.d2; do
    if [ -f "$d2file" ]; then
        name="${d2file%.d2}"
        echo "Generating: $name"
        d2 --theme 200 --layout elk "$d2file" "svg/$name.svg"
        d2 --theme 200 --layout elk "$d2file" "pdf/$name.pdf"
        d2 --theme 200 --layout elk "$d2file" "png/$name.png"
    fi
done

echo "Done! Generated diagrams in svg/, pdf/, and png/ directories"
