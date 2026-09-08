.PHONY: install dev-install ocr-install lint format test validate-schema eval-smoke \n	corpus corpus-full ablation ablation-smoke figure phase1 \n	extract extract-ablation phase2 docker-build docker-up

install:
	uv pip install --system -e .

dev-install:
	uv pip install --system -e ".[dev]"
	pre-commit install

# Phase 1 needs an OCR engine. PaddleOCR is pure pip (no system binary) and is
# the primary engine per the plan; Tesseract additionally needs its own binary.
ocr-install:
	uv pip install --system -e ".[ocr]"

paddle-install:
	uv pip install --system -e ".[paddle]"

lint:
	ruff check .
	black --check .

format:
	ruff check --fix .
	black .

test:
	pytest --cov=src/invoice_extract --cov-report=term-missing

validate-schema:
	python scripts/validate_examples.py

eval-smoke:
	python -m invoice_extract.eval.harness \
		--gold tests/fixtures/gold.jsonl \
		--pred tests/fixtures/pred.jsonl \
		--schema schema/invoice_schema.json

docker-build:
	docker compose build

docker-up:
	docker compose up -d app

# ---------------------------------------------------------------- phase 1

CORPUS ?= data/processed/gst_in_synthetic
DPI ?= 200
ENGINE ?= tesseract

# Small corpus for a fast loop; `corpus-full` renders every canonical doc.
corpus:
	python scripts/build_image_corpus.py 		--canonical $(CORPUS)/canonical --out $(CORPUS) --dpi $(DPI) --limit 25

corpus-full:
	python scripts/build_image_corpus.py 		--canonical $(CORPUS)/canonical --out $(CORPUS) --dpi $(DPI)

# The Phase 1 headline result. Needs a real OCR engine installed.
ablation:
	python scripts/run_preprocess_ablation.py 		--corpus $(CORPUS) --engine $(ENGINE) --out reports/phase1_ablation

# Verifies the whole chain end to end with no OCR engine installed. The numbers
# are a plumbing check, never a result -- see docs/phase1_preprocessing.md.
ablation-smoke:
	python scripts/run_preprocess_ablation.py 		--corpus $(CORPUS) --engine oracle --allow-oracle --limit 5 		--configs raw gray+border+deskew full+sauvola --out reports/_plumbing

# Visual QA / report figure: degraded input vs each pipeline stage.
figure:
	python scripts/make_pipeline_figure.py --corpus $(CORPUS) --profiles medium heavy

phase1: corpus ablation figure

# ---------------------------------------------------------------- phase 2

EXTRACTOR ?= rules
PROFILE ?= clean
# 'gold' feeds the renderer's ground-truth text: a perfect-OCR CEILING, not an
# end-to-end result. Switch to 'engine' once an OCR engine is installed.
OCR_SOURCE ?= gold

# Run one extractor over the corpus and score it with the Phase 0 harness.
extract:
	python scripts/run_extraction.py 		--corpus $(CORPUS) --extractor $(EXTRACTOR) 		--ocr-source $(OCR_SOURCE) --profile $(PROFILE) 		--out reports/phase2/$(EXTRACTOR)
	python -m invoice_extract.eval.harness 		--gold reports/phase2/$(EXTRACTOR)/gold.jsonl 		--pred reports/phase2/$(EXTRACTOR)/pred.jsonl

# The Phase 2 headline table. Add vlm:* extractors on a GPU box.
extract-ablation:
	python scripts/run_extraction_ablation.py 		--corpus $(CORPUS) --extractors null rules 		--ocr-source $(OCR_SOURCE) --out reports/phase2_ablation

phase2: extract-ablation
