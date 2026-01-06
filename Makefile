# Makefile for building Marp slides with custom theme
# Usage: make all       - build all slides
#        make lectures  - build 8-lecture series
#        make diagrams  - generate all diagrams
#        make clean     - remove built files

THEME = themes/iitgn-modern.css
MARP = marp
MARP_NPX = npx @marp-team/marp-cli

.PHONY: all clean lectures ml-tasks ntp objdet diagrams help stats
.PHONY: l01 l02 l03 l04 l05 l06 l07 l08
.PHONY: diagrams-l02 diagrams-l03 diagrams-l04 diagrams-l05 diagrams-l06 diagrams-l08
.PHONY: diagrams-ml-tasks diagrams-object-detection diagrams-next-token
.PHONY: watch-l01 watch-l02 watch-l03 watch-l04 watch-l05 watch-l06 watch-l07 watch-l08
.PHONY: serve-l01 serve-l02 serve-l03 serve-l04 serve-l05 serve-l06 serve-l07 serve-l08

all: lectures diagrams

# ============================================================================
# SLIDE BUILDING
# ============================================================================

# Build all 8 lectures
lectures: l01 l02 l03 l04 l05 l06 l07 l08

# Individual lecture builds (using marp directly)
l01:
	@echo "Building Lecture 01: What is AI..."
	cd lectures/01-what-is-ai && $(MARP) slides.md --theme-set ../../$(THEME) --allow-local-files -o slides.pdf
	cd lectures/01-what-is-ai && $(MARP) slides.md --theme-set ../../$(THEME) --allow-local-files -o slides.html

l02:
	@echo "Building Lecture 02: Data Foundation..."
	cd lectures/02-data-foundation && $(MARP) slides.md --theme-set ../../$(THEME) --allow-local-files -o slides.pdf
	cd lectures/02-data-foundation && $(MARP) slides.md --theme-set ../../$(THEME) --allow-local-files -o slides.html

l03:
	@echo "Building Lecture 03: Supervised Learning..."
	cd lectures/03-supervised-learning && $(MARP) slides.md --theme-set ../../$(THEME) --allow-local-files -o slides.pdf
	cd lectures/03-supervised-learning && $(MARP) slides.md --theme-set ../../$(THEME) --allow-local-files -o slides.html

l04:
	@echo "Building Lecture 04: Model Selection..."
	cd lectures/04-model-selection && $(MARP) slides.md --theme-set ../../$(THEME) --allow-local-files -o slides.pdf
	cd lectures/04-model-selection && $(MARP) slides.md --theme-set ../../$(THEME) --allow-local-files -o slides.html

l05:
	@echo "Building Lecture 05: Neural Networks..."
	cd lectures/05-neural-networks && $(MARP) slides.md --theme-set ../../$(THEME) --allow-local-files -o slides.pdf
	cd lectures/05-neural-networks && $(MARP) slides.md --theme-set ../../$(THEME) --allow-local-files -o slides.html

l06:
	@echo "Building Lecture 06: Computer Vision..."
	cd lectures/06-computer-vision && $(MARP) slides.md --theme-set ../../$(THEME) --allow-local-files -o slides.pdf
	cd lectures/06-computer-vision && $(MARP) slides.md --theme-set ../../$(THEME) --allow-local-files -o slides.html

l07:
	@echo "Building Lecture 07: Language Models..."
	cd lectures/07-language-models && $(MARP) slides.md --theme-set ../../$(THEME) --allow-local-files -o slides.pdf
	cd lectures/07-language-models && $(MARP) slides.md --theme-set ../../$(THEME) --allow-local-files -o slides.html

l08:
	@echo "Building Lecture 08: Generative AI..."
	cd lectures/08-generative-ai && $(MARP) slides.md --theme-set ../../$(THEME) --allow-local-files -o slides.pdf
	cd lectures/08-generative-ai && $(MARP) slides.md --theme-set ../../$(THEME) --allow-local-files -o slides.html

# ============================================================================
# DIAGRAM GENERATION
# ============================================================================

# Generate all diagrams
diagrams: diagrams-l02 diagrams-l03 diagrams-l04 diagrams-l05 diagrams-l06 diagrams-l08 diagrams-ml-tasks diagrams-object-detection diagrams-next-token

# Lecture-specific diagram generation
diagrams-l02:
	@echo "Generating L02 diagrams..."
	cd lectures/02-data-foundation && python generate_diagrams.py

diagrams-l03:
	@echo "Generating L03 diagrams..."
	cd lectures/03-supervised-learning && python generate_diagrams.py

diagrams-l04:
	@echo "Generating L04 diagrams..."
	cd lectures/04-model-selection && python generate_diagrams.py

diagrams-l05:
	@echo "Generating L05 diagrams..."
	cd lectures/05-neural-networks && python generate_diagrams.py

diagrams-l06:
	@echo "Generating L06 diagrams..."
	cd lectures/06-computer-vision && python generate_diagrams.py

diagrams-l08:
	@echo "Generating L08 diagrams..."
	cd lectures/08-generative-ai && python generate_diagrams.py

# External diagram generation
diagrams-ml-tasks:
	@echo "Generating ML Tasks diagrams..."
	cd ml-tasks && python generate_realistic_examples.py

diagrams-object-detection:
	@echo "Generating Object Detection diagrams..."
	cd object-detection && python generate_realistic_examples.py

diagrams-next-token:
	@echo "Generating Next Token Prediction diagrams..."
	cd next-token-prediction && python generate_ntp_diagrams.py

# Legacy diagram targets
diagrams-legacy:
	cd ml-tasks && python generate_ml_diagrams.py
	cd next-token-prediction && python generate_ntp_diagrams.py
	cd object-detection && python generate_objdet_diagrams.py

# ============================================================================
# WATCH MODE (for development)
# ============================================================================

watch-l01:
	cd lectures/01-what-is-ai && $(MARP) slides.md --theme-set ../../$(THEME) --allow-local-files -w -o slides.html

watch-l02:
	cd lectures/02-data-foundation && $(MARP) slides.md --theme-set ../../$(THEME) --allow-local-files -w -o slides.html

watch-l03:
	cd lectures/03-supervised-learning && $(MARP) slides.md --theme-set ../../$(THEME) --allow-local-files -w -o slides.html

watch-l04:
	cd lectures/04-model-selection && $(MARP) slides.md --theme-set ../../$(THEME) --allow-local-files -w -o slides.html

watch-l05:
	cd lectures/05-neural-networks && $(MARP) slides.md --theme-set ../../$(THEME) --allow-local-files -w -o slides.html

watch-l06:
	cd lectures/06-computer-vision && $(MARP) slides.md --theme-set ../../$(THEME) --allow-local-files -w -o slides.html

watch-l07:
	cd lectures/07-language-models && $(MARP) slides.md --theme-set ../../$(THEME) --allow-local-files -w -o slides.html

watch-l08:
	cd lectures/08-generative-ai && $(MARP) slides.md --theme-set ../../$(THEME) --allow-local-files -w -o slides.html

# ============================================================================
# SERVE MODE (for preview)
# ============================================================================

serve-l01:
	cd lectures/01-what-is-ai && $(MARP) slides.md --theme-set ../../$(THEME) --allow-local-files -s

serve-l02:
	cd lectures/02-data-foundation && $(MARP) slides.md --theme-set ../../$(THEME) --allow-local-files -s

serve-l03:
	cd lectures/03-supervised-learning && $(MARP) slides.md --theme-set ../../$(THEME) --allow-local-files -s

serve-l04:
	cd lectures/04-model-selection && $(MARP) slides.md --theme-set ../../$(THEME) --allow-local-files -s

serve-l05:
	cd lectures/05-neural-networks && $(MARP) slides.md --theme-set ../../$(THEME) --allow-local-files -s

serve-l06:
	cd lectures/06-computer-vision && $(MARP) slides.md --theme-set ../../$(THEME) --allow-local-files -s

serve-l07:
	cd lectures/07-language-models && $(MARP) slides.md --theme-set ../../$(THEME) --allow-local-files -s

serve-l08:
	cd lectures/08-generative-ai && $(MARP) slides.md --theme-set ../../$(THEME) --allow-local-files -s

# ============================================================================
# LEGACY BUILDS
# ============================================================================

ml-tasks:
	cd ml-tasks && $(MARP) ml-tasks.md --theme-set ../$(THEME) --allow-local-files -o ml-tasks.pdf
	cd ml-tasks && $(MARP) ml-tasks.md --theme-set ../$(THEME) --allow-local-files -o ml-tasks.html

ntp:
	cd next-token-prediction && $(MARP) next-token-prediction.md --theme-set ../$(THEME) --allow-local-files -o next-token-prediction.pdf
	cd next-token-prediction && $(MARP) next-token-prediction.md --theme-set ../$(THEME) --allow-local-files -o next-token-prediction.html

objdet:
	cd object-detection && $(MARP) object-detection-basics.md --theme-set ../$(THEME) --allow-local-files -o object-detection-basics.pdf
	cd object-detection && $(MARP) object-detection-basics.md --theme-set ../$(THEME) --allow-local-files -o object-detection-basics.html

# ============================================================================
# STATISTICS
# ============================================================================

stats:
	@echo "========================================"
	@echo "Principles of AI - Course Statistics"
	@echo "========================================"
	@echo ""
	@echo "Lectures: $$(ls -d lectures/*/ 2>/dev/null | wc -l | tr -d ' ')"
	@echo ""
	@echo "Slides Built:"
	@echo "  HTML:  $$(find lectures -name 'slides.html' 2>/dev/null | wc -l | tr -d ' ')"
	@echo "  PDF:   $$(find lectures -name 'slides.pdf' 2>/dev/null | wc -l | tr -d ' ')"
	@echo ""
	@echo "Diagrams Generated:"
	@echo "  SVG:   $$(find lectures -name '*.svg' 2>/dev/null | wc -l | tr -d ' ')"
	@echo "  PNG:   $$(find lectures -name '*.png' 2>/dev/null | wc -l | tr -d ' ')"
	@echo ""

# ============================================================================
# CLEANING
# ============================================================================

clean:
	@echo "Cleaning built files..."
	rm -f ml-tasks/*.pdf ml-tasks/*.html
	rm -f next-token-prediction/*.pdf next-token-prediction/*.html
	rm -f object-detection/*.pdf object-detection/*.html
	rm -f lectures/*/*.pdf lectures/*/*.html
	@echo "✓ Cleaned slide builds"

clean-all: clean
	@echo "Cleaning all generated files..."
	@find lectures -type d -name "diagrams" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "✓ Cleaned all generated files"

# ============================================================================
# HELP
# ============================================================================

help:
	@echo "========================================"
	@echo "Principles of AI - Makefile Commands"
	@echo "========================================"
	@echo ""
	@echo "Building Slides:"
	@echo "  make lectures        - Build all 8 lecture slides (HTML + PDF)"
	@echo "  make l01-l08         - Build specific lecture"
	@echo "  make ml-tasks        - Build ML Tasks slides"
	@echo "  make ntp             - Build Next Token Prediction slides"
	@echo "  make objdet          - Build Object Detection slides"
	@echo ""
	@echo "Diagram Generation:"
	@echo "  make diagrams        - Generate all diagrams"
	@echo "  make diagrams-l02-l08 - Generate specific lecture diagrams"
	@echo "  make diagrams-ml-tasks       - ML Tasks diagrams"
	@echo "  make diagrams-object-detection - Object Detection diagrams"
	@echo "  make diagrams-next-token     - Next Token diagrams"
	@echo ""
	@echo "Development:"
	@echo "  make watch-l01-l08   - Watch & auto-rebuild specific lecture"
	@echo "  make serve-l01-l08   - Serve slides for preview"
	@echo ""
	@echo "Utilities:"
	@echo "  make stats           - Show course statistics"
	@echo "  make clean           - Remove built slides"
	@echo "  make clean-all       - Remove all generated files"
	@echo "  make help            - Show this help message"
	@echo ""
