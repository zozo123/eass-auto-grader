# EASS Auto Grader

Automated grading pipeline for FastAPI student projects.

## Requirements

- Python 3.12+, uv, ruff, gh CLI, gemini-cli
- `gemini auth login` for authentication

## Usage

```bash
git clone https://github.com/zozo123/eass-auto-grader.git && cd eass-auto-grader
uv sync
cp submission.csv.example submission.csv  # Add student data
./run.sh --limit 5                        # Test run
./run.sh                                  # Full run
```

## Input: submission.csv

```csv
student_name,email,repo_url
"John Doe",john@example.com,https://github.com/org/repo
```

## Grading Criteria

| Dimension | Weight | Required |
|-----------|--------|----------|
| Functional Correctness | 15% | FastAPI backend |
| API Design | 15% | RESTful, validation |
| Test Quality | 15% | pytest |
| Architecture | 10% | Separation of concerns |
| Code Quality | 10% | Type hints, docstrings |
| Error Handling | 10% | HTTPException |
| Security | 10% | No secrets in code |
| Documentation | 5% | README |
| Docker | 5% | Dockerfile |
| Dependencies | 5% | requirements.txt |

**Constraints:** Backend must use FastAPI (not Flask). Frontend (if any) must use JavaScript or Streamlit.

## Output

- `results/REPORT.md` - Full evaluation report
- `results/ranked_list.csv` - Rankings spreadsheet

## MCP Servers (Gemini)

Configured in `.gemini/settings.json`: filesystem, github
