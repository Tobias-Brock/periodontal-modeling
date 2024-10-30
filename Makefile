#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = pa-modeling
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python Dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Format source code with black
.PHONY: black
black:
	black --config pyproject.toml pamod

## Format source code with ruff
.PHONY: ruff
ruff:
	ruff format .

## Lint source code with ruff
.PHONY: lint
lint:
	ruff check .

## Run pre-commit hooks
.PHONY: pre-commit
pre-commit:
	pre-commit run --all-files

## Create docs
.PHONY: docs
docs:
	mkdocs serve

## Set up python interpreter environment
.PHONY: create_environment
create_environment:

	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y

	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


.PHONY: preprocess
preprocess:
	$(PYTHON_INTERPRETER) pamod/data/_preprocessing.py ${ARGS}

.PHONY: benchmark
benchmark:
	$(PYTHON_INTERPRETER) pamod/benchmarking/_benchmark.py ${ARGS}

.PHONY: app
app:
	$(PYTHON_INTERPRETER) pamod/app/app.py

## Build Docker image
.PHONY: docker-build
docker-build:
	docker build -f docker/app.dockerfile -t pamod-image .

## Run Docker container
.PHONY: docker-run
docker-run:
	docker run -p 7880:7880 pamod-image
