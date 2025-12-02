# 🎓 EASS Student Project Ranker

An automated pipeline for evaluating student FastAPI projects using AI-powered code analysis (Gemini or Codex CLI).

## ✨ Features

- **AI-Powered Analysis**: Uses Gemini CLI (default) or OpenAI Codex CLI for deep code evaluation
- **10-Dimension Scoring**: Evaluates functional correctness, architecture, code quality, API design, error handling, security, testing, documentation, Docker, dependencies
- **Deep Analysis Evidence**: Provides specific code-level findings for each score
- **Detailed Reports**: Generates comprehensive REPORT.md with insights, rankings, and recommendations
- **Code Formatting Check**: Uses Ruff to analyze code style compliance
- **Weighted Scoring**: Prioritizes critical dimensions (functional, API, testing at 15% each)

## 📋 Prerequisites

### Required Tools

| Tool | Installation | Purpose |
|------|-------------|---------|
| **Python 3.12+** | `brew install python@3.12` | Runtime |
| **uv** | `brew install uv` | Fast Python package manager |
| **Ruff** | `brew install ruff` | Code formatting checker |
| **GitHub CLI** | `brew install gh` | Clone student repos |
| **Gemini CLI** | `brew install gemini-cli` | AI analysis (default) |
| **Codex CLI** | `npm install -g @openai/codex` | AI analysis (alternative) |

### Authentication

#### For Gemini (default):
```bash
# Authenticate with Google
gemini auth login
```

#### For Codex (alternative):
```bash
# Set OpenAI API key
export OPENAI_API_KEY="your-api-key"
```

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/zozo123/ranker-eass.git
cd ranker-eass
```

### 2. Install dependencies
```bash
uv sync
```

### 3. Prepare your submission file
Create a `submission.csv` with student data:
```csv
student_name,email,repo_url
"John Doe",john@example.com,https://github.com/org/student-project
"Jane Smith",jane@example.com,https://github.com/org/another-project
```

### 4. Run the evaluation

**Using Gemini (default):**
```bash
./run.sh --limit 5      # Test with 5 submissions
./run.sh                # Run all submissions
```

**Using Codex:**
```bash
./run.sh --ai codex --limit 5
```

## 📁 Project Structure

```
ranker-eass/
├── run.sh                    # Main entry point
├── Makefile                  # Convenience commands
├── scripts/
│   ├── run_evaluation.py     # Core evaluation pipeline
│   └── run_evaluation_parallel.py  # Parallel execution
├── gemini_prompt.txt         # Gemini AI evaluation prompt
├── codex_prompt.txt          # Codex AI evaluation prompt
├── evaluation_schema.json    # JSON schema for evaluation
├── pyproject.toml            # Python dependencies
└── submission.csv            # Student submissions (YOU CREATE THIS)
```

### Generated Outputs (gitignored)

```
work/                         # Cloned repos & artifacts
├── {student_slug}/
│   ├── artifacts/
│   │   ├── gemini.json      # AI evaluation results
│   │   ├── ruff_stats.json  # Code formatting stats
│   │   └── tree.txt         # Repo structure
│   └── repo/                # Cloned student repo

results/                      # Final outputs
├── REPORT.md                # Comprehensive report
├── evaluation.json          # Raw JSON data
├── ranked_list.csv          # Sortable rankings
└── index.md                 # Quick overview

logs/
└── pipeline.log             # Execution logs
```

## 🎯 CLI Options

```bash
./run.sh [OPTIONS]

Options:
  --ai <gemini|codex>   AI provider to use (default: gemini)
  --limit N             Process only first N submissions
  --timeout S           Timeout per command in seconds (default: 300)
  --no-clean            Don't clean work directory before starting
  -h, --help            Show help
```

## 📊 Evaluation Dimensions

| Dimension | Weight | What It Measures |
|-----------|--------|------------------|
| **Functional Correctness** | 15% | Does the API work? CRUD complete? |
| **API Design** | 15% | RESTful? Proper status codes? Validation? |
| **Test Quality** | 15% | Test coverage? Isolation? Error cases? |
| **Architecture** | 10% | Separation of concerns? Patterns? |
| **Code Quality** | 10% | Type hints? Clean code? DRY? |
| **Error Handling** | 10% | HTTPException? Edge cases? |
| **Security** | 10% | Secrets exposed? Env vars? |
| **Documentation** | 5% | README? Setup instructions? |
| **Docker** | 5% | Dockerfile? Best practices? |
| **Dependencies** | 5% | Pinned versions? Lock file? |

## 🔧 Makefile Commands

```bash
make run          # Run evaluation (5 submissions)
make run-all      # Run all submissions
make run-limit N=10  # Run with custom limit
make clean        # Clean work directory
make clean-all    # Clean work + results + logs
make format       # Format Python code with ruff
make lint         # Lint Python code
```

## 📝 Sample Output

### REPORT.md Preview

```markdown
## 🏆 Final Rankings

| Rank | Student | Score | Grade | Format | Docker | Tests |
|------|---------|-------|-------|--------|--------|-------|
| 1 | 🥇 Alice Smith | 8.5 | A | 100% | ✅ | ✅ |
| 2 | 🥈 Bob Jones | 7.8 | B | 85% | ✅ | ✅ |
```

### Deep Analysis Evidence

Each submission gets detailed evidence for scores:
```
**Functional Correctness:** 8/10
- Endpoints Found: 5
- CRUD Completeness: full
- Would Run: ✅ Yes
- Evidence: "Complete CRUD with proper status codes (201, 204)"
```

## 🔒 Security Notes

⚠️ **Important**: The following files contain sensitive data and are **gitignored**:

- `submission.csv` - Contains student PII (names, emails)
- `work/` - Contains cloned student repositories
- `results/` - Contains evaluation data with student info
- `logs/` - May contain sensitive paths

**Never commit:**
- API keys or credentials
- `.env` files
- Student personal information

## 📄 License

MIT License

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Run `make format && make lint`
4. Submit a pull request
