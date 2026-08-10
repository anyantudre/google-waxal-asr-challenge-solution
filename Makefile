PYTHON := python
PROJECT := waxal_asr

.PHONY: help
help:
	@echo "Setup"
	@echo "  make requirements     Install Python dependencies"
	@echo "  make dev              Install dependencies plus test and lint tools"
	@echo ""
	@echo "Inference (see README section 'Run inference')"
	@echo "  make lid              Language identification over the test audio"
	@echo "  make predict          One arm, about 2 GB and 20 minutes, verifies the pipeline"
	@echo "  make submission       Full 26-member ensemble, the best result"
	@echo "  make recipes          List every ensemble recipe in configs/ensembles.yaml"
	@echo ""
	@echo "Training (see README section 'Run training')"
	@echo "  make data             Build every training corpus from its public source"
	@echo "  make train ARM=s43    Train one arm; ARM is any config in configs/"
	@echo ""
	@echo "Quality"
	@echo "  make test             Run the test suite"
	@echo "  make analysis         Write reports/data_insights.md from the data present"
	@echo "  make lint             Check formatting and imports"
	@echo "  make clean            Remove caches and compiled files"

.PHONY: requirements
requirements:
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -r requirements.txt

.PHONY: dev
dev: requirements
	$(PYTHON) -m pip install -e ".[dev]"

.PHONY: test
test:
	$(PYTHON) -m pytest tests/ -v

.PHONY: lint
lint:
	$(PYTHON) -m ruff check $(PROJECT) tests

.PHONY: format
format:
	$(PYTHON) -m ruff format $(PROJECT) tests
	$(PYTHON) -m ruff check --fix $(PROJECT) tests

.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache .ruff_cache

# Inference. LID must run first: the Sunbird arm is routed per clip by its output.
.PHONY: lid
lid:
	$(PYTHON) -m $(PROJECT).lid --audio-dir data/raw/test_audio --out data/interim/lid.json

.PHONY: predict
predict:
	$(PYTHON) -m $(PROJECT).modeling.predict --recipe p2n_distil_nl_f

.PHONY: submission
submission:
	$(PYTHON) -m $(PROJECT).modeling.predict --recipe p2n_mbr

.PHONY: recipes
recipes:
	$(PYTHON) -m $(PROJECT).modeling.predict --list

.PHONY: analysis
analysis:
	$(PYTHON) -m $(PROJECT).analysis

# Training. `make data` downloads and converts roughly 450 hours of audio.
# The exact training corpus is published, so reproducing it is a download rather than a rebuild.
# docs/dataset_card.md documents how every subset was derived from its upstream source.
.PHONY: data
data:
	huggingface-cli download anyantudre/waxal-linsna --repo-type dataset --local-dir data/external

ARM ?= s43
.PHONY: train
train:
	$(PYTHON) -m $(PROJECT).modeling.train --config configs/w2vbert_$(ARM).yaml --name $(ARM)
