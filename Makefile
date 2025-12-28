# Makefile for building Marp slides with custom theme
# Usage: make all     - build all slides
#        make lecture2 - build just lecture2
#        make clean    - remove built files

THEME = themes/iitgn-modern.css

.PHONY: all clean lecture2 ntp objdet

all: lecture2 ntp objdet

lecture2:
	cd ml-tasks && marp lecture2-ml-tasks.md --theme-set ../$(THEME) --allow-local-files --pdf -o lecture2-ml-tasks.pdf
	cd ml-tasks && marp lecture2-ml-tasks.md --theme-set ../$(THEME) --allow-local-files -o lecture2-ml-tasks.html

ntp:
	cd next-token-prediction && marp next-token-prediction.md --theme-set ../$(THEME) --allow-local-files --pdf -o next-token-prediction.pdf
	cd next-token-prediction && marp next-token-prediction.md --theme-set ../$(THEME) --allow-local-files -o next-token-prediction.html

objdet:
	cd object-detection && marp object-detection-basics.md --theme-set ../$(THEME) --allow-local-files --pdf -o object-detection-basics.pdf
	cd object-detection && marp object-detection-basics.md --theme-set ../$(THEME) --allow-local-files -o object-detection-basics.html

clean:
	rm -f ml-tasks/*.pdf ml-tasks/*.html
	rm -f next-token-prediction/*.pdf next-token-prediction/*.html
	rm -f object-detection/*.pdf object-detection/*.html
