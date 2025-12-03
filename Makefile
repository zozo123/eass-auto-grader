# Load .env if it exists
-include .env
export

WORKERS ?= 4
N ?= 3

.PHONY: help run run-all run-parallel run-limit run-codex clean clean-all setup lint format test compare qdrant-up qdrant-down

help:
	@echo "EASS Student Project Evaluator"
	@echo "=============================="
	@echo ""
	@echo "Usage:"
	@echo "  make run           Run with Gemini (first 5 submissions)"
	@echo "  make run-codex     Run with Codex (first 5 submissions)"
	@echo "  make run-all       Run on all submissions (parallel, $(WORKERS) workers)"
	@echo "  make run-parallel  Run on all submissions in parallel"
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
	./run.sh --ai gemini --parallel --workers $(WORKERS)

run-parallel:
	./run.sh --ai gemini --parallel --workers $(WORKERS)
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

compare:
	uv run python scripts/compare_repos.py --root work --repo-subdir repo

compare-v2:
	uv run python scripts/compare_repos_v2.py --root work --repo-subdir repo --reset

compare-v2-quick:
	uv run python scripts/compare_repos_v2.py --root work --repo-subdir repo --top-k 3 --max-code-files 15

detect-plagiarism:
	uv run python scripts/detect_plagiarism.py --root work --repo-subdir repo --verbose
	uv run python scripts/generate_plagiarism_html.py

detect-plagiarism-quiet:
	uv run python scripts/detect_plagiarism.py --root work --repo-subdir repo
	uv run python scripts/generate_plagiarism_html.py

open-plagiarism-report:
	open results/plagiarism_report.html

plot-similarity:
	uv run python scripts/plot_similarity.py --input results/repo_similarity.csv --output results/repo_similarity.png

plot-similarity-v2:
	uv run python scripts/plot_similarity.py --input results/repo_similarity_v2.csv --output results/repo_similarity_v2.png

visualize:
	uv run python scripts/visualize_similarity.py --output-dir results/plots

deep-analysis:
	uv run python scripts/deep_analysis.py --root work --repo-subdir repo --reset

deep-analysis-quick:
	uv run python scripts/deep_analysis.py --root work --repo-subdir repo

deep-visualize:
	uv run python scripts/visualize_deep_analysis.py

deep-all:
	uv run python scripts/deep_analysis.py --root work --repo-subdir repo --reset
	uv run python scripts/visualize_deep_analysis.py

combined-report:
	uv run python scripts/generate_combined_report.py
	open results/combined_analysis_report.html

full-analysis:
	uv run python scripts/deep_analysis.py --root work --repo-subdir repo --reset
	uv run python scripts/visualize_deep_analysis.py
	uv run python scripts/detect_plagiarism.py --root work --repo-subdir repo
	uv run python scripts/generate_combined_report.py
	open results/combined_analysis_report.html

qdrant-up:
	docker compose -f docker-compose.qdrant.yml up -d

qdrant-down:
	docker compose -f docker-compose.qdrant.yml down

qdrant-logs:
	docker compose -f docker-compose.qdrant.yml logs -f
