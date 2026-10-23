# EASS auto-grader

Course tooling by [Yossi Eliaz](https://zozo123.github.io/) for the EASS course at Holon Institute of Technology (HIT). `run.sh` grades student FastAPI repositories against the EX1–EX3 rubric in `prompt.txt` using Gemini CLI, the Gemini API, or Codex CLI; `make compare` flags cross-repo code similarity. Outputs are for instructor review.

## Features

Grading (`run.sh`):

- **Rubric-based LLM grading** of each submission repo for EX1 (backend), EX2 (fullstack) or EX3 (MVP)
- **Providers**: Gemini CLI (default), Gemini API, or Codex CLI, with an optional fallback provider (`AI_FALLBACK_CHAIN`)
- **Pydantic validation** of the model's JSON output (`scripts/models.py`)
- **Parallel runs** across submissions (`--parallel --workers N`)

Similarity analysis (`make compare`):

- **Vector-based similarity analysis** using Qdrant and local embeddings (no API keys needed)
- **AST-based code analysis** for structural comparison
- **Multiple similarity metrics**: vector, Jaccard, n-gram/winnowing, structural
- **Deep multi-dimensional analysis** covering 17 aspects of code
- **Clustering and visualization** with heatmaps and dendrograms
- **Responsible similarity analysis** that accounts for expected patterns
- **Creativity/originality assessment** for fair evaluation

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- [GitHub CLI](https://cli.github.com/) (`gh`), used to clone submissions during grading
- For grading: [Gemini CLI](https://github.com/google-gemini/gemini-cli) (`npm install -g @google/gemini-cli`) and/or [Codex CLI](https://github.com/openai/codex)
- For the format check: [ruff](https://github.com/astral-sh/ruff) (`run.sh` calls `/opt/homebrew/bin/ruff`)
- For similarity analysis: Docker (for the Qdrant vector database)

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/zozo123/eass-auto-grader.git
cd eass-auto-grader

# Install dependencies using uv
uv sync

# Or using pip
pip install -e .
```

### 2. Configure

```bash
cp .env.example .env
```

Set `AI_PROVIDER` (`gemini-cli`, `gemini-api` or `codex`), the model for that provider, and `AI_FALLBACK_CHAIN`. `.env.example` also has local LLM settings, but `run.sh` does not accept a `local` provider yet.

### 3. Prepare Your Data

Create a `submission.csv` in the same format as `submission.csv.example` (a form export with seven columns):

```csv
timestamp,email,student_name,q1,q2,q3,repo_url
01/01/2025 10:00:00,student1@example.com,Student One,Yes,Yes,Yes,https://github.com/your-org/project-1
```

`make clone` reads these columns by position. `submission.csv` is gitignored; do not commit real student data.

### 4. Grade

```bash
make run          # Gemini CLI, first 5 submissions
make run-codex    # Codex CLI, first 5 submissions
make run-all      # all submissions, in parallel ($WORKERS workers)

# Or call run.sh directly
./run.sh --ai gemini-api --limit 5
./run.sh --ai codex --model o3
./run.sh --help
```

`run.sh` sends `gemini_prompt.txt` (Gemini) or `codex_prompt.txt` (Codex) to the model; both currently match `prompt.txt`. Results are written to `results/`: `index.md` (ranking), `ranked_list.csv` and `evaluation.json`. Raw scores (0-10) are rescaled across the batch: the lowest score above 1 maps to `SCORE_SCALE_MIN`, the highest to `SCORE_SCALE_MAX`, and scores of 1 or below (or failed clones) get 0.

### 5. Compare Repositories

```bash
make qdrant-up    # start Qdrant
make clone        # clone every repo in submission.csv into work/
make compare      # full pipeline; writes results/combined_analysis_report.html
```

Individual steps:

```bash
make deep-analysis       # 17-dimension analysis
make deep-visualize      # clustered heatmaps and dendrograms
make detect-plagiarism   # fingerprint-based report (use responsibly)
uv run python scripts/responsible_similarity_analysis.py --root work --repo-subdir repo
uv run python scripts/creativity_assessment.py --root work --repo-subdir repo
```

## Analysis Scripts

| Script | Description |
|--------|-------------|
| `run_evaluation.py` | Sequential evaluation runner (not used by `run.sh`); also writes `REPORT.md` |
| `run_evaluation_parallel.py` | Evaluation runner used by `run.sh` (sequential or parallel) |
| `models.py` | Pydantic schema for the grading JSON |
| `generate_prompt.py` | Builds a grading prompt with the embedded JSON schema |
| `grade_with_pydantic.py` | Grades one repo with Gemini or Codex and validates the output |
| `smart_code_embeddings.py` | Multi-level code embeddings in Qdrant |
| `compare_repos.py` | Basic similarity comparison |
| `compare_repos_v2.py` | Advanced multi-metric comparison |
| `deep_analysis.py` | 17-dimension comprehensive analysis |
| `visualize_similarity.py` | Basic visualization |
| `visualize_deep_analysis.py` | Clustered heatmaps and dendrograms |
| `detect_plagiarism.py` | Plagiarism detection (use responsibly!) |
| `responsible_similarity_analysis.py` | Fair analysis with context |
| `creativity_assessment.py` | Originality and effort scoring |
| `generate_plagiarism_html.py` | HTML plagiarism report |
| `generate_combined_report.py` | Combined analysis report |

## Similarity Metrics

The tool uses multiple complementary metrics:

1. **Vector Similarity**: Semantic similarity using embeddings
2. **Jaccard Similarity**: Token-based overlap
3. **N-gram/Winnowing**: Code fingerprinting for copy detection
4. **Structural Similarity**: AST-based structure comparison

## Deep Analysis Dimensions

The deep analysis covers 17 aspects grouped into categories:

### Structural (4 dimensions)
- File organization patterns
- Class/function architecture
- Import structure
- Configuration patterns

### Semantic (3 dimensions)
- Variable naming conventions
- Comment style and content
- String literals and messages

### Behavioral (4 dimensions)
- API endpoint patterns
- Database operations
- Error handling approaches
- CRUD implementation style

### Style (3 dimensions)
- Code formatting
- Type hints usage
- Documentation patterns

### Testing (2 dimensions)
- Test structure and coverage
- Mock patterns

### Documentation (1 dimension)
- README and docs content

## Responsible Use

**Important:** High similarity scores don't always indicate plagiarism!

Common legitimate reasons for high similarity:
- Students following the same tutorials
- Using standard library patterns
- Same assignment requirements
- Common framework boilerplate

Always:
1. Check if subjects are from the same course/assignment
2. Look for shared tutorials or reference materials
3. Consider framework-specific patterns
4. Review the actual code in context

## Configuration

### Environment Variables

Grading settings (provider, models, timeout, workers, fallback, score scale) live in `.env`; see `.env.example`.

```bash
# Optional: Customize the Qdrant connection
QDRANT_URL=http://localhost:6333   # most comparison scripts
QDRANT_HOST=localhost              # smart_code_embeddings.py
QDRANT_PORT=6333
```

### Makefile Targets

```bash
make help              # Show the main targets
make run               # Grade the first 5 submissions with Gemini CLI
make run-codex         # Grade the first 5 submissions with Codex CLI
make run-all           # Grade all submissions in parallel
make run-limit N=3     # Grade the first N submissions
make test              # Smoke run: grade one submission
make clone             # Clone all repositories from submission.csv
make compare           # Full similarity pipeline
make compare-quick     # Basic comparison and plots
make deep-analysis     # 17-dimension analysis
make deep-visualize    # Visualize the deep analysis
make detect-plagiarism # Fingerprint-based report
make combined-report   # Build and open the combined HTML report
make qdrant-up         # Start Qdrant container
make qdrant-down       # Stop Qdrant container
make lint              # Lint scripts with ruff
make clean             # Remove work/
```

## Output Files

Results are saved in the `results/` directory:
- `index.md`, `ranked_list.csv`, `evaluation.json` - Grading results
- `repo_similarity*.csv`, `similarity_matrix.csv`, `smart_similarity.csv` - Similarity scores
- `deep_analysis/` - Deep analysis (`deep_analyses.json`, `deep_similarity.csv`, `clusters.json`) and its plots
- `plots/` - Heatmap and top-pair plots from `make compare-quick`
- `plagiarism_report.html`, `combined_analysis_report.html` - HTML reports

## Project Structure

```
.
├── scripts/                 # Analysis scripts
│   ├── run_evaluation.py
│   ├── compare_repos_v2.py
│   ├── deep_analysis.py
│   └── ...
├── examples/                # Example inputs (placeholders only)
├── work/                    # Cloned repositories (gitignored)
├── results/                 # Analysis outputs (gitignored)
├── logs/                    # Log files (gitignored)
├── submission.csv           # Your data file (gitignored)
├── submission.csv.example   # Example format
├── run.sh                   # Grading entry point
├── prompt.txt               # EX1-EX3 grading rubric prompt
├── .env.example             # Provider and grading settings
├── docker-compose.qdrant.yml
├── Makefile
├── pyproject.toml
└── README.md
```

## Extending for Your Use Case

1. **Custom Assignment Context**: Modify `responsible_similarity_analysis.py` to add expected patterns for your specific framework/tutorials
2. **Custom Metrics**: Add new dimension categories in `deep_analysis.py`
3. **Custom Scoring**: Adjust weights in `creativity_assessment.py`
4. **Custom Reports**: Modify HTML generators for your branding

## License

MIT. See [LICENSE](LICENSE).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
