# Makefile for building Marp slides with custom theme
# Usage: make all       - build all slides
#        make lectures  - build new 8-lecture series
#        make ml-tasks  - build ML Tasks slides
#        make ntp       - build Next Token Prediction slides
#        make objdet    - build Object Detection slides
#        make clean     - remove built files

THEME = themes/iitgn-modern.css
MARP = npx @marp-team/marp-cli

.PHONY: all clean lectures ml-tasks ntp objdet diagrams l01 l02 l03 l04 l05 l06 l07 l08

all: lectures ml-tasks ntp objdet

# Build all 8 lectures
lectures: l01 l02 l03 l04 l05 l06 l07 l08

l01:
	@echo "Building Lecture 01: What is AI..."
	# L01 uses existing PDF from ml-teaching, just has a companion slides.md

l02:
	@echo "Building Lecture 02: Data Foundation..."
	cd lectures/02-data-foundation && $(MARP) slides.md --theme-set ../../$(THEME) --allow-local-files --html --pdf -o slides.pdf
	cd lectures/02-data-foundation && $(MARP) slides.md --theme-set ../../$(THEME) --allow-local-files --html -o slides.html

l03:
	@echo "Building Lecture 03: Supervised Learning..."
	cd lectures/03-supervised-learning && $(MARP) slides.md --theme-set ../../$(THEME) --allow-local-files --html --pdf -o slides.pdf
	cd lectures/03-supervised-learning && $(MARP) slides.md --theme-set ../../$(THEME) --allow-local-files --html -o slides.html

l04:
	@echo "Building Lecture 04: Model Selection..."
	cd lectures/04-model-selection && $(MARP) slides.md --theme-set ../../$(THEME) --allow-local-files --html --pdf -o slides.pdf
	cd lectures/04-model-selection && $(MARP) slides.md --theme-set ../../$(THEME) --allow-local-files --html -o slides.html

l05:
	@echo "Building Lecture 05: Neural Networks..."
	cd lectures/05-neural-networks && $(MARP) slides.md --theme-set ../../$(THEME) --allow-local-files --html --pdf -o slides.pdf
	cd lectures/05-neural-networks && $(MARP) slides.md --theme-set ../../$(THEME) --allow-local-files --html -o slides.html

l06:
	@echo "Building Lecture 06: Computer Vision..."
	cd lectures/06-computer-vision && $(MARP) slides.md --theme-set ../../$(THEME) --allow-local-files --html --pdf -o slides.pdf
	cd lectures/06-computer-vision && $(MARP) slides.md --theme-set ../../$(THEME) --allow-local-files --html -o slides.html

l07:
	@echo "Building Lecture 07: Language Models..."
	cd lectures/07-language-models && $(MARP) slides.md --theme-set ../../$(THEME) --allow-local-files --html --pdf -o slides.pdf
	cd lectures/07-language-models && $(MARP) slides.md --theme-set ../../$(THEME) --allow-local-files --html -o slides.html

l08:
	@echo "Building Lecture 08: Generative AI..."
	cd lectures/08-generative-ai && $(MARP) slides.md --theme-set ../../$(THEME) --allow-local-files --html --pdf -o slides.pdf
	cd lectures/08-generative-ai && $(MARP) slides.md --theme-set ../../$(THEME) --allow-local-files --html -o slides.html

# Legacy builds
diagrams:
	cd ml-tasks && python generate_ml_diagrams.py
	cd next-token-prediction && python generate_ntp_diagrams.py
	cd object-detection && python generate_objdet_diagrams.py

ml-tasks:
	cd ml-tasks && $(MARP) ml-tasks.md --theme-set ../$(THEME) --allow-local-files --html --pdf -o ml-tasks.pdf
	cd ml-tasks && $(MARP) ml-tasks.md --theme-set ../$(THEME) --allow-local-files --html -o ml-tasks.html

ntp:
	cd next-token-prediction && $(MARP) next-token-prediction.md --theme-set ../$(THEME) --allow-local-files --html --pdf -o next-token-prediction.pdf
	cd next-token-prediction && $(MARP) next-token-prediction.md --theme-set ../$(THEME) --allow-local-files --html -o next-token-prediction.html

objdet:
	cd object-detection && $(MARP) object-detection-basics.md --theme-set ../$(THEME) --allow-local-files --html --pdf -o object-detection-basics.pdf
	cd object-detection && $(MARP) object-detection-basics.md --theme-set ../$(THEME) --allow-local-files --html -o object-detection-basics.html

clean:
	rm -f ml-tasks/*.pdf ml-tasks/*.html
	rm -f next-token-prediction/*.pdf next-token-prediction/*.html
	rm -f object-detection/*.pdf object-detection/*.html
	rm -f lectures/*/*.pdf lectures/*/*.html
