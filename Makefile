# EASS Student Project Evaluator
# ===============================

.PHONY: help run run-all run-limit run-codex clean clean-all setup lint format test

help:
	@echo "EASS Student Project Evaluator"
	@echo "=============================="
	@echo ""
	@echo "Usage:"
	@echo "  make run           Run with Gemini (first 5 submissions)"
	@echo "  make run-codex     Run with Codex (first 5 submissions)"
	@echo "  make run-all       Run on all submissions"
	@echo "  make run-limit N=3 Run on first N submissions"
	@echo "  make clean         Remove work directory"
	@echo "  make clean-all     Remove work, results, and logs"
	@echo "  make setup         Install dependencies"
	@echo "  make format        Format code with ruff"
	@echo "  make lint          Lint scripts"
	@echo ""

run:
	./run.sh --ai gemini --limit 5

run-codex:
	./run.sh --ai codex --limit 5

run-all:
	./run.sh --ai gemini

N ?= 3
run-limit:
	./run.sh --ai gemini --limit $(N)

clean:
	@echo "Cleaning work directory..."
	rm -rf work/
	@echo "Done."

clean-all:
	@echo "Cleaning all generated files..."
	rm -rf work/ results/ logs/ __pycache__/ scripts/__pycache__/
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Done."

setup:
	@echo "Setting up Python environment..."
	uv venv
	uv sync
	@echo "Done."

lint:
	@echo "Linting..."
	uv run ruff check scripts/
	uv run ruff format --check scripts/
	@echo "Done."

format:
	@echo "Formatting..."
	uv run ruff format scripts/
	@echo "Done."

test:
	./run.sh --limit 1
