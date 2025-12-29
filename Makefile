# Makefile for building Marp slides with custom theme
# Usage: make all      - build all slides
#        make ml-tasks - build ML Tasks slides
#        make ntp      - build Next Token Prediction slides
#        make objdet   - build Object Detection slides
#        make clean    - remove built files

THEME = themes/iitgn-modern.css

.PHONY: all clean ml-tasks ntp objdet diagrams

all: diagrams ml-tasks ntp objdet

diagrams:
	cd ml-tasks && python generate_ml_diagrams.py
	cd next-token-prediction && python generate_ntp_diagrams.py
	cd object-detection && python generate_objdet_diagrams.py

ml-tasks:
	cd ml-tasks && marp ml-tasks.md --theme-set ../$(THEME) --allow-local-files --html --pdf -o ml-tasks.pdf
	cd ml-tasks && marp ml-tasks.md --theme-set ../$(THEME) --allow-local-files --html -o ml-tasks.html

ntp:
	cd next-token-prediction && marp next-token-prediction.md --theme-set ../$(THEME) --allow-local-files --html --pdf -o next-token-prediction.pdf
	cd next-token-prediction && marp next-token-prediction.md --theme-set ../$(THEME) --allow-local-files --html -o next-token-prediction.html

objdet:
	cd object-detection && marp object-detection-basics.md --theme-set ../$(THEME) --allow-local-files --html --pdf -o object-detection-basics.pdf
	cd object-detection && marp object-detection-basics.md --theme-set ../$(THEME) --allow-local-files --html -o object-detection-basics.html

clean:
	rm -f ml-tasks/*.pdf ml-tasks/*.html
	rm -f next-token-prediction/*.pdf next-token-prediction/*.html
	rm -f object-detection/*.pdf object-detection/*.html
