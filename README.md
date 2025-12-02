# EASS Auto Grader

Automated grading pipeline for FastAPI student projects.

## Setup

```bash
brew install python@3.12 uv ruff gh
brew install gemini-cli && gemini auth login    # Gemini
brew install codex && codex login               # Codex (alternative)
```

## Usage

```bash
git clone https://github.com/zozo123/eass-auto-grader.git && cd eass-auto-grader
uv sync
cp submission.csv.example submission.csv   # Add student data
./run.sh --limit 5                         # Test run
./run.sh                                   # Full run
./run.sh --ai codex                        # Use Codex instead
```

## Input

`submission.csv`:
```csv
student_name,email,repo_url
"John Doe",john@example.com,https://github.com/org/repo
```

## Grading Criteria

| Dimension | Weight |
|-----------|--------|
| Functional Correctness | 15% |
| API Design | 15% |
| Test Quality | 15% |
| Architecture | 10% |
| Code Quality | 10% |
| Error Handling | 10% |
| Security | 10% |
| Documentation | 5% |
| Docker | 5% |
| Dependencies | 5% |

**Requirements:** FastAPI backend (not Flask). Frontend: JavaScript or Streamlit.

## Output

`results/REPORT.md` - Evaluation report with scores, grades, and recommendations.
