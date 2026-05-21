.PHONY: install webapp format

# Detect OS
OS := $(shell uname 2>/dev/null || echo Windows)
ifeq ($(OS), Linux)
	VENVNAME=venv_linux
    PYTHONVENV=$(VENVNAME)/bin/python
    TENSORBOARD=$(VENVNAME)/bin/tensorboard
	PYTHON=python3.10
else ifeq ($(OS), Darwin)
	VENVNAME=venv_mac
    PYTHONVENV=$(VENVNAME)/bin/python
    TENSORBOARD=$(VENVNAME)/bin/tensorboard
	PYTHON=python3.10
else
	VENVNAME=venv_win
    PYTHONVENV=$(VENVNAME)/Scripts/python
    TENSORBOARD=$(VENVNAME)/Scripts/tensorboard
	PYTHON=python
endif

## Install environment
install:
	@echo ">> Delete previous venv"
	@$(PYTHON) -c "import shutil; shutil.rmtree('$(VENVNAME)', ignore_errors=True)"
	@echo ">> Create venv"
	$(PYTHON) -m venv $(VENVNAME)
	@$(PYTHONVENV) -m pip install -U pip
	@echo ">> Installing dependencies"
ifeq ($(OS), Linux)
	@$(PYTHONVENV) -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
	@$(PYTHONVENV) -m pip install torch torchvision torchaudio
endif
	@$(PYTHONVENV) -m pip install -r requirements.txt

## Run webapp
train:
	@$(PYTHONVENV) -m train

clean_train:
	@$(PYTHONVENV) -m train -cleanrun

## Run Tensorboard
tensorboard:
	@$(TENSORBOARD) --logdir output/ray_results

## Format files with ruff
format:
	$(PYTHONVENV) -m ruff format . || exit 0
	$(PYTHONVENV) -m ruff check . --fix --unsafe-fixes --exit-zero

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available commands:$$(tput sgr0)"
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
