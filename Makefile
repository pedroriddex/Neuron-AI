PYTHON=python3.11
VENV=.venv

.PHONY: dev test lint docker run bench clean

## Create virtualenv and install deps
$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install -U pip wheel
	$(VENV)/bin/pip install -r requirements.txt

## Activate dev environment
dev: $(VENV)/bin/activate
	@echo "Environment ready. Activate with: source $(VENV)/bin/activate"

## Run tests
test:
	$(VENV)/bin/python -m pytest -q

## Run linters
lint:
	$(VENV)/bin/pre-commit run --all-files

## Build docker image
docker:
	docker build -t neuron/base:0.1 -f Dockerfile .

## Run API locally
run:
	$(VENV)/bin/uvicorn api.main:app --reload --port 8000

## Run benchmark (requires API closed)
bench:
	$(VENV)/bin/python bench/bench.py

## Clean caches
clean:
	rm -rf $(VENV) __pycache__ .pytest_cache
