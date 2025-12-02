# EASS Student Project Evaluator - Makefile
# ==========================================

.PHONY: help run run-all run-limit run-codex clean clean-all setup lint format test

# Default target
help:
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║           🎓 EASS Student Project Evaluator                  ║"
	@echo "╠══════════════════════════════════════════════════════════════╣"
	@echo "║  make run          - Run with Gemini (first 5 submissions)  ║"
	@echo "║  make run-codex    - Run with Codex (first 5 submissions)   ║"
	@echo "║  make run-all      - Run on ALL submissions (Gemini)        ║"
	@echo "║  make run-limit N=3 - Run on first N submissions            ║"
	@echo "║  make clean        - Remove work directory (temp files)     ║"
	@echo "║  make clean-all    - Remove work, results, and logs         ║"
	@echo "║  make setup        - Install dependencies with uv           ║"
	@echo "║  make format       - Format Python code with ruff           ║"
	@echo "║  make lint         - Lint the evaluation scripts            ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"

# Run with Gemini (default), limit of 5
run:
	./run.sh --ai gemini --limit 5

# Run with Codex, limit of 5
run-codex:
	./run.sh --ai codex --limit 5

# Run on all submissions (no limit)
run-all:
	./run.sh --ai gemini

# Run with custom limit: make run-limit N=3
N ?= 3
run-limit:
	./run.sh --ai gemini --limit $(N)

# Clean only the work directory (cloned repos and temp artifacts)
clean:
	@echo "🧹 Cleaning work directory..."
	rm -rf work/
	@echo "✅ Work directory cleaned"

# Clean everything - work, results, and logs
clean-all:
	@echo "🧹 Cleaning all generated files..."
	rm -rf work/
	rm -rf results/
	rm -rf logs/
	rm -rf __pycache__/
	rm -rf scripts/__pycache__/
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ All generated files cleaned"

# Setup the environment
setup:
	@echo "📦 Setting up Python environment with uv..."
	uv venv
	uv sync
	@echo "✅ Environment ready"

# Lint the scripts
lint:
	@echo "🔍 Linting evaluation scripts..."
	uv run ruff check scripts/
	uv run ruff format --check scripts/
	@echo "✅ Linting complete"

# Format the scripts
format:
	@echo "🎨 Formatting evaluation scripts..."
	uv run ruff format scripts/
	@echo "✅ Formatting complete"

# Quick test with 1 submission
test:
	./run.sh --limit 1
