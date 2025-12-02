# EASS Student Project Ranker

Automated evaluation pipeline for FastAPI student projects using AI code analysis.

## Overview

This tool clones student repositories, runs AI-powered code analysis (Gemini or Codex), and generates ranked evaluation reports with detailed scoring across 10 dimensions.

## Requirements

```
Python 3.12+    brew install python@3.12
uv              brew install uv
ruff            brew install ruff
gh              brew install gh
gemini-cli      brew install gemini-cli
```

**Authentication:**
```bash
gemini auth login                      # For Gemini
export OPENAI_API_KEY="your-key"       # For Codex
```

## Quick Start

```bash
git clone https://github.com/zozo123/ranker-eass.git
cd ranker-eass
uv sync
cp submission.csv.example submission.csv   # Add your data
./run.sh --limit 5                         # Test run
./run.sh                                   # Full run
```

## Input Format

`submission.csv` (you create this):
```csv
student_name,email,repo_url
"John Doe",john@example.com,https://github.com/org/project
```

## Usage

```bash
./run.sh [OPTIONS]

--ai <gemini|codex>   AI provider (default: gemini)
--limit N             Process first N submissions
--timeout S           Timeout per command (default: 300s)
--no-clean            Keep work directory
```

**Makefile shortcuts:**
```bash
make run              # 5 submissions
make run-all          # All submissions  
make run-codex        # Use Codex
make clean            # Clean work/
make format           # Format code
```

## Project Structure

```
ranker-eass/
├── run.sh                 # Entry point
├── Makefile               # Build commands
├── scripts/
│   ├── run_evaluation.py  # Main pipeline
│   └── run_evaluation_parallel.py
├── gemini_prompt.txt      # AI prompt
├── codex_prompt.txt       # AI prompt (Codex)
├── submission.csv.example # Template
└── pyproject.toml         # Dependencies
```

**Generated (gitignored):**
```
work/{student}/artifacts/  # AI output, ruff stats
work/{student}/repo/       # Cloned code
results/REPORT.md          # Final report
results/evaluation.json    # Raw scores
results/ranked_list.csv    # Rankings
logs/pipeline.log          # Execution log
```

## Scoring

| Dimension              | Weight |
|------------------------|--------|
| Functional Correctness | 15%    |
| API Design             | 15%    |
| Test Quality           | 15%    |
| Architecture           | 10%    |
| Code Quality           | 10%    |
| Error Handling         | 10%    |
| Security               | 10%    |
| Documentation          | 5%     |
| Docker                 | 5%     |
| Dependencies           | 5%     |

**Grades:** A (9-10), B (8-9), C (7-8), D (6-7), F (<6)

## Output

Each submission receives:
- Numeric scores (0-10) per dimension
- Deep analysis evidence with code references
- Issues categorized by severity
- Actionable improvement recommendations

## Security

**Gitignored (contains PII):**
- `submission.csv` - student names/emails
- `work/` - cloned repositories
- `results/` - evaluation data
- `logs/` - execution logs

**Never commit:** API keys, `.env` files, student data.

## License

MIT
