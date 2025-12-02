# EASS 8 SCORE – Automated Student Repo Evaluation

Short description:  
This repository defines an automated pipeline and LLM-based rubric to clone, analyze, test, and rank student EX1 projects (primarily FastAPI/HTTP APIs) from the EASS-HIT-PART-A-2025-CLASS-VIII cohort.

---

## 1. Goal and Scope

The goal of this project is to **automatically evaluate and rank all submitted student repositories** according to a detailed rubric that focuses on:

- Functional correctness and HTTP/API semantics  
- Code quality, style, and complexity  
- Test quality and coverage  
- Performance and resource usage  
- Documentation and developer experience  
- Maintainability, modularity, and architecture  
- Robustness, edge cases, and error handling  
- Security and secrets hygiene  
- License compliance and dependency health  

The evaluation will be performed by:

1. A **local automation agent** that uses:
   - GitHub CLI (`gh`)
   - Command-line tools (e.g., `cloc`, `pytest`, `coverage`, linters, type-checkers, security scanners)
2. A **coding LLM** that:
   - Reads generated artifacts (JSON, logs, coverage reports, README, etc.)
   - Applies the rubric
   - Produces detailed feedback and numeric scores per criterion
   - Produces a final ranking across all repos

---

## 2. Expected Inputs and Outputs

### 2.1 Input: `submission.csv`

The pipeline expects a CSV file named **`submission.csv`** at the root of this repo.

**Expected header:**

```csv
timestamp,email,student_name,consent_1,consent_2,consent_3,repo_url
```

Each row corresponds to **one student submission**. Example rows (based on actual data):

```csv
11/25/2025 20:01:48,haimlt1995@gmail.com,Haim Lev Tov,Yes,Yes,Yes,https://github.com/EASS-HIT-PART-A-2025-CLASS-VIII/BookWorm
11/26/2025 15:29:55,levigal50@gmail.com,gal levy,Yes,Yes,Yes,https://github.com/EASS-HIT-PART-A-2025-CLASS-VIII/video-games-api
11/27/2025 13:55:47,hadararazi12301@gmail.com,Hadar Tzemach Harazi,Yes,Yes,Yes,https://github.com/HadarTzemach/task-manager-api
11/28/2025 13:31:25,elioffri@gmail.com,Eli Offri,Yes,Yes,Yes,https://github.com/EASS-HIT-PART-A-2025-CLASS-VIII/fastapi-movies/tree/main
11/28/2025 16:47:21,Ofir272@gmail.com,Ofir Tasa,Yes,Yes,Yes,https://github.com/EASS-HIT-PART-A-2025-CLASS-VIII/appointment-manager.git
11/28/2025 20:40:20,adihabasovya@gmail.com,Adi Habasov,Yes,Yes,Yes,https://github.com/EASS-HIT-PART-A-2025-CLASS-VIII/Movie_Catalogue.git
11/28/2025 21:09:54,yuribon55@gmail.com,Yuri Bondar,Yes,Yes,Yes,https://github.com/EASS-HIT-PART-A-2025-CLASS-VIII/Gym-PR-Tracker.git
11/29/2025 18:30:27,Matan60003@gmail.com,Matan Owadeyah,Yes,Yes,Yes,https://github.com/EASS-HIT-PART-A-2025-CLASS-VIII/http-api-demo-matan
11/29/2025 18:48:44,Yahavbenhur@gmail.com,Yahav Ben Hur,Yes,Yes,Yes,https://github.com/EASS-HIT-PART-A-2025-CLASS-VIII/cookwithme
11/29/2025 22:12:24,kesem.grubman@gmail.com,Kesem Grubman,Yes,Yes,Yes,https://github.com/EASS-HIT-PART-A-2025-CLASS-VIII/wise-budget
11/30/2025 16:52:03,Benlavi7496@gmail.com,Ben lavi,Yes,Yes,Yes,https://github.com/EASS-HIT-PART-A-2025-CLASS-VIII/movie-service-fastapi
11/30/2025 19:28:19,chelle.ov@gmail.com,Michelle Borisov,Yes,Yes,Yes,https://github.com/EASS-HIT-PART-A-2025-CLASS-VIII/sim-device-control
11/30/2025 20:46:14,talinca123@gmail.com,Tal Alayoff,Yes,Yes,Yes,https://github.com/EASS-HIT-PART-A-2025-CLASS-VIII/books-catalogue-api
12/1/2025 18:48:23,ziva9529@gmail.com,Ziv Ashkenazi,Yes,Yes,Yes,https://github.com/EASS-HIT-PART-A-2025-CLASS-VIII/football-players-service
12/1/2025 19:34:23,yuvaloren25@hotmail.com,Yuval Oren,Yes,Yes,Yes,https://github.com/EASS-HIT-PART-A-2025-CLASS-VIII/CalloCount
12/1/2025 19:45:15,amitakerman100@gmail.com,Amit Akerman,Yes,Yes,Yes,https://github.com/EASS-HIT-PART-A-2025-CLASS-VIII/Wardrobe-Catalog
12/1/2025 20:11:52,anat.ash44@gmail.com,Anat Ashkinezer,Yes,Yes,Yes,https://github.com/EASS-HIT-PART-A-2025-CLASS-VIII/to-do-manager/tree/main
12/1/2025 21:12:14,e4guycohen@outlook.com,Guy Cohen,Yes,Yes,Yes,https://github.com/EASS-HIT-PART-A-2025-CLASS-VIII/CinemaPlanet
12/2/2025 10:09:40,levigal50@gmail.com,gal levy,Yes,Yes,Yes,https://github.com/EASS-HIT-PART-A-2025-CLASS-VIII/video-games-api
12/2/2025 11:15:17,avivhakak535@gmail.com,Aviv Hakak,Yes,Yes,Yes,https://github.com/EASS-HIT-PART-A-2025-CLASS-VIII/EX1---fastapi
12/2/2025 12:27:32,dudi.levy25@gmail.com,David levy,Yes,Yes,Yes,https://github.com/EASS-HIT-PART-A-2025-CLASS-VIII/movie-catalogue-api
```

> **Note:** If the real CSV does not have a header, the automation agent should assume the above column order.

### 2.2 Per-repository working directory layout

For each repo, the automation agent will create a directory:

```text
work/
  <student_slug>/
    repo/                # cloned repo
    artifacts/           # tool outputs (JSON, logs, coverage, cloc, etc.)
    reports/             # feedback.md, evaluation.json
```

`student_slug` can be derived from email or repo name (e.g., `haim_lev_tov_bookworm`).

### 2.3 Global outputs

At the root of this project:

* `results/evaluation.json` – **array** of per-repo evaluations (full rubric, raw scores & weighted scores)
* `results/ranked_list.csv` – ranking table across all repos
* `results/index.md` – overview of ranking + links to per-repo feedback
* `logs/` – aggregated logs, failures, and tool outputs

---

## 3. Required Tooling and Environment

The local automation agent is expected to have:

* **GitHub CLI**: `gh`
* **Git**: for cloning
* **Python 3.10+** (assumed main target for FastAPI projects)
* **Virtualenv** or equivalent
* **cloc**: for line counts and language breakdown
* **pytest** and **coverage.py**: for test execution and coverage
* **ruff** or **flake8**: for style and linting (Python)
* **mypy** (optional but preferred): for static typing analysis
* **bandit**: for Python security scanning (where appropriate)
* Optionally: **gitleaks** or similar tool for secret scanning

The agent may adapt to other languages if the repo is not Python, but Python/FastAPI is the default assumption.

---

## 4. Instructions for Running Tests and Evaluation

### 4.1 High-level procedure

For **each row** in `submission.csv`:

1. Parse the row (timestamp, email, student_name, repo_url).
2. Create a new working directory under `work/<student_slug>/`.
3. Clone the repository using GH CLI.
4. Detect primary language (using GitHub API + `cloc`).
5. For Python projects:

   * Create and activate a virtual environment.
   * Install dependencies.
   * Run tests, coverage, linting, static analysis, and security scan.
6. Collect artifacts into `artifacts/`.
7. Invoke the **coding LLM** with:

   * A summary prompt (see rubric & ranking sections)
   * All relevant artifacts (or distilled summaries)
8. Receive back:

   * Detailed scores per criterion
   * Written feedback per criterion
9. Persist:

   * `reports/<student_slug>.feedback.md`
   * `reports/<student_slug>.evaluation.json` (single-repo)
10. Aggregate all evaluations into global `results/evaluation.json` and `results/ranked_list.csv`.

### 4.2 Precise GH CLI and local commands

Assuming a row with `repo_url`:

```bash
# 0. Set variables
STUDENT_SLUG="<student_slug>"           # e.g. haim_lev_tov_bookworm
REPO_URL="<repo_url>"                   # from submission.csv
WORK_DIR="work/${STUDENT_SLUG}"
REPO_DIR="${WORK_DIR}/repo"
ARTIFACTS_DIR="${WORK_DIR}/artifacts"
REPORTS_DIR="${WORK_DIR}/reports"

mkdir -p "${REPO_DIR}" "${ARTIFACTS_DIR}" "${REPORTS_DIR}"

# 1. Clone repo using GH CLI (supports SSH/HTTPS)
gh repo clone "${REPO_URL#https://github.com/}" "${REPO_DIR}"

# 2. Basic info & language detection
cd "${REPO_DIR}"
gh repo view --json name,description,primaryLanguage,languages,defaultBranchRef > "${ARTIFACTS_DIR}/gh_repo_view.json" || true

# 3. Run cloc for code metrics
cloc . --json --out="${ARTIFACTS_DIR}/cloc.json" || true

# 4. Python-specific setup (if primary language or structure suggests Python/FastAPI)
python -m venv .venv || virtualenv .venv || true
# shell: Linux/macOS
source .venv/bin/activate 2>/dev/null || . .venv/bin/activate 2>/dev/null || true

if [ -f "requirements.txt" ]; then
  pip install -r requirements.txt || echo "requirements_install_failed" >> "${ARTIFACTS_DIR}/errors.log"
elif [ -f "pyproject.toml" ]; then
  pip install . || echo "pyproject_install_failed" >> "${ARTIFACTS_DIR}/errors.log"
fi

# 5. Run tests with coverage
pytest --maxfail=1 --disable-warnings --junitxml="${ARTIFACTS_DIR}/pytest.xml" \
  --cov=. --cov-report=xml:"${ARTIFACTS_DIR}/coverage.xml" || echo "pytest_failed" >> "${ARTIFACTS_DIR}/errors.log"

# 6. Linting and style
ruff . --format json > "${ARTIFACTS_DIR}/ruff.json" 2>>"${ARTIFACTS_DIR}/errors.log" || true
# or:
# flake8 . --format=html --htmldir="${ARTIFACTS_DIR}/flake8_html" || true

# 7. Static typing (optional)
mypy . --junit-xml "${ARTIFACTS_DIR}/mypy.xml" 2>>"${ARTIFACTS_DIR}/errors.log" || true

# 8. Security scanning
bandit -r . -f json -o "${ARTIFACTS_DIR}/bandit.json" || true
# optional secrets scan
# gitleaks detect --source . --report-format json --report-path "${ARTIFACTS_DIR}/gitleaks.json" || true

# 9. Capture README and docs
cp README* "${ARTIFACTS_DIR}/" 2>/dev/null || true
find . -maxdepth 2 -type f -iname "openapi*.json" -o -iname "openapi*.yaml" -print0 \
  | xargs -0 -I{} cp "{}" "${ARTIFACTS_DIR}/" 2>/dev/null || true

# 10. Deactivate venv (if needed)
deactivate 2>/dev/null || true
```

> The automation agent should log any failures to `errors.log` and pass that context to the LLM to interpret partial results.

---

## 5. Evaluation Rubric

All scores are on a **0–10** scale per criterion before weighting.

### 5.1 Functional Correctness (Weight: 0.25)

**What to check:**

* API endpoints implement the described domain (books, movies, tasks, etc.).
* CRUD operations behave correctly (create/read/update/delete).
* HTTP semantics: proper use of methods, status codes, and error responses.
* Data validation (e.g., Pydantic models for FastAPI).
* OpenAPI/Swagger is valid and reasonably documented.

**Example tests (for a typical FastAPI EX1 service):**

* `POST /items` with valid body → `201 Created`, response includes new ID.
* `POST /items` with invalid body → `422 Unprocessable Entity` or similar.
* `GET /items` → returns list, stable structure.
* `GET /items/{id}` for existing and non-existing items → `200 OK` or `404 Not Found`.
* `PUT/PATCH /items/{id}` updates fields correctly; missing fields handled clearly.
* `DELETE /items/{id}` → correct status and idempotence.

**Scoring guidelines:**

* **0–2**: Fails to run, or endpoints mostly non-functional.
* **3–5**: Some endpoints work, but many bugs or broken flows.
* **6–8**: Core functionality works; minor issues or missing edge cases.
* **9–10**: Fully functional with correct HTTP semantics and validation.

---

### 5.2 Code Quality & Style (Weight: 0.15)

**Signals:**

* Folder structure (`app/`, `routers/`, `models/`, `schemas/`, `tests/`, etc.).
* Clean, readable code: meaningful names, small functions, low nesting depth.
* Lint results (ruff/flake8): low number of warnings/errors, no critical smells.
* Limited code duplication.

**Use `cloc.json` and linter output to judge noise vs. signal.**

---

### 5.3 Complexity and Architecture (Weight: 0.10)

**Signals:**

* Separation of concerns (routers/controllers, models/schemas, services, data access).
* Layering (API, domain, persistence) rather than “everything in `main.py`”.
* No gigantic “god functions” or mega files with thousands of LOC.

---

### 5.4 Performance & Resource Use (Weight: 0.05)

**Signals:**

* No obviously quadratic or worse behavior on typical operations.
* No unnecessary blocking I/O in request handlers when async patterns are expected.
* Tests complete within a reasonable time (e.g., under 60 seconds total by default).

> No micro-benchmarks are required; this is a sanity check for clearly bad patterns.

---

### 5.5 Tests and Coverage (Weight: 0.15)

**Signals:**

* Presence of `tests/` or `test_*.py` files.
* `pytest` passes without flakiness.
* Coverage report (`coverage.xml`) shows:

  * ≥ 60%: acceptable
  * ≥ 80%: good
  * ≥ 90%: excellent

**Scoring:**

* **0–2**: No tests, or tests do not run.
* **3–5**: Few tests; coverage < 50%.
* **6–8**: Solid tests; coverage 60–80%.
* **9–10**: Extensive tests, including edge cases; coverage 80%+.

---

### 5.6 Documentation & Developer Experience (Weight: 0.10)

**Signals:**

* Clear `README` with:

  * Project description (what problem it solves).
  * Setup instructions (dependencies, virtualenv, commands).
  * How to run the app and tests.
  * Example API calls or `curl`/HTTPie snippets.
* Optional: diagrams, screenshots, or detailed usage examples.

**Scoring:**

* **0–2**: Missing or useless README.
* **3–5**: Minimal instructions; hard to get started.
* **6–8**: Clear, mostly complete.
* **9–10**: Great onboarding experience with examples and troubleshooting hints.

---

### 5.7 Maintainability & Modularity (Weight: 0.10)

**Signals:**

* Modules and packages logically grouped.
* Low coupling and reasonable cohesion.
* Reuse of helpers/services instead of repeated logic.

---

### 5.8 Robustness & Edge Cases (Weight: 0.05)

**Signals:**

* Handling non-existent resources, invalid IDs, and malformed body payloads.
* Validation errors produce structured responses.
* Guards against empty lists, nulls, and invalid enum values.

---

### 5.9 Security & Secrets (Weight: 0.05)

**Signals:**

* No committed secrets (API keys, passwords, tokens).
* Basic input validation to prevent naive injection scenarios.
* Safe default configurations in code samples (no `DEBUG=True` in production configs, etc.).
* Bandit (and optional secret scanner) output.

---

### 5.10 License & Dependency Health (Weight: 0.05)

**Signals:**

* Presence of a **LICENSE** file.
* Clear indication of license in README (MIT, Apache-2.0, etc.).
* Reasonable and up-to-date dependencies (no obvious abandoned or insecure libraries).
* For Python: optional use of `pip-licenses` or similar tool to generate a dependency license report.

---

## 6. Scoring and Ranking Procedure

### 6.1 Per-criterion scoring

For each repo, the coding LLM will:

1. Read:

   * `artifacts/` (cloc.json, pytest.xml, coverage.xml, ruff.json, bandit.json, errors.log, README, etc.).
   * Any auto-generated summaries by the automation agent (optional).
2. Assign a **0–10 integer or half-integer score** for each criterion:

   * Functional Correctness
   * Code Quality & Style
   * Complexity & Architecture
   * Performance & Resource Use
   * Tests & Coverage
   * Documentation & DX
   * Maintainability & Modularity
   * Robustness & Edge Cases
   * Security & Secrets
   * License & Dependencies

### 6.2 Weights

| Criterion                    | Weight   |
| ---------------------------- | -------- |
| Functional Correctness       | 0.25     |
| Code Quality & Style         | 0.15     |
| Complexity & Architecture    | 0.10     |
| Performance & Resource Use   | 0.05     |
| Tests & Coverage             | 0.15     |
| Documentation & DX           | 0.10     |
| Maintainability & Modularity | 0.10     |
| Robustness & Edge Cases      | 0.05     |
| Security & Secrets           | 0.05     |
| License & Dependencies       | 0.05     |
| **Total**                    | **1.00** |

### 6.3 Aggregation

For each repo:

```
final_score = Σ (criterion_score[i] * weight[i])
```

* `final_score` is in the range **0.0–10.0**.
* The LLM should also compute a **0–100 percentage**:

```
final_percentage = final_score * 10
```

### 6.4 Acceptance Criteria and Pass/Fail Thresholds

* **Pass**: `final_percentage ≥ 60` (i.e., `final_score ≥ 6.0`)
* **Good**: `final_percentage ≥ 75` (i.e., `final_score ≥ 7.5`)
* **Excellent**: `final_percentage ≥ 90` (i.e., `final_score ≥ 9.0`)

Repos that cannot be installed or run at all (e.g., broken dependencies, cannot start tests) should receive:

* Strongly reduced scores in **Functional Correctness** and **Tests & Coverage**.
* Explicit note in `feedback.md` under “Blocking Issues”.

### 6.5 Tie-breaking Rules

If two or more repos have the same `final_score` (within 0.05):

1. Higher **Functional Correctness** score wins.
2. If still tied, higher **Tests & Coverage** score wins.
3. If still tied, higher **Maintainability & Modularity** score wins.
4. If still tied, prefer the repo with:

   * Better **Documentation & DX** (subjective LLM judgment).
5. If *still* tied, break by:

   * Alphabetical order of `student_name`.

---

## 7. Detailed Feedback per Criterion

For each repo, the LLM must produce `feedback.md` containing:

* **Header**:

  * Student name, email, repo URL, date of evaluation.
  * Final score (0–10) and percentage.

* **Sections** (one per criterion):

  For each:

  * A one-line **score summary** (e.g., `Functional Correctness: 7.5 / 10`).
  * 2–5 bullet points:

    * **What is good**
    * **What is weak**
    * **Concrete suggestions for improvement**
  * Reference to specific files or functions where helpful (`app/main.py`, `app/routers/items.py`, etc.).

* **Appendix**:

  * Key metrics (LOC, coverage %, number of tests, number of lint warnings).
  * Any blocking setup or runtime issues encountered.

---

## 8. Repository Candidates Section

The automation agent should use **`submission.csv` as the source of truth**. For each candidate, it should try to gather:

1. **Repo URL** – from `submission.csv`.

2. **Primary language** – via GH CLI:

   ```bash
gh repo view "${REPO}" --json primaryLanguage,languages > "${ARTIFACTS_DIR}/languages.json"
```

3. **Quick summary / README excerpt** – first 10–20 lines of README:

   ```bash
head -n 20 README* > "${ARTIFACTS_DIR}/readme_excerpt.txt" 2>/dev/null || true
```

4. **Known constraints or notes** – any comments from `submission.csv` (if extra columns exist) or manual notes added later.

The LLM should treat the actual list of repos from `submission.csv` as dynamic; the examples in section 2.1 are just illustrative.

---

## 9. Output Artifacts Specification

For **each repo** (`work/<student_slug>/`):

* `artifacts/`

  * `gh_repo_view.json`
  * `cloc.json`
  * `pytest.xml`
  * `coverage.xml`
  * `ruff.json` (or other linter output)
  * `mypy.xml` (optional)
  * `bandit.json`
  * `gitleaks.json` (optional)
  * `README*` copy
  * `openapi*.json` / `openapi*.yaml` (if found)
  * `errors.log` (if any)
* `reports/`

  * `<student_slug>.evaluation.json` – structured rubric scores, weights, and final score
  * `<student_slug>.feedback.md` – narrative feedback

At the **global level**:

* `results/evaluation.json` – array of:

```json
{
  "student_name": "...",
  "email": "...",
  "repo_url": "...",
  "scores": {
    "functional_correctness": 7.5,
    "code_quality": 8.0,
    "complexity_architecture": 7.0,
    "performance": 6.5,
    "tests_coverage": 8.0,
    "documentation_dx": 7.0,
    "maintainability_modularity": 7.5,
    "robustness_edge_cases": 6.0,
    "security_secrets": 7.0,
    "license_dependencies": 5.5
  },
  "weights": {
    "functional_correctness": 0.25,
    "code_quality": 0.15,
    "complexity_architecture": 0.10,
    "performance": 0.05,
    "tests_coverage": 0.15,
    "documentation_dx": 0.10,
    "maintainability_modularity": 0.10,
    "robustness_edge_cases": 0.05,
    "security_secrets": 0.05,
    "license_dependencies": 0.05
  },
  "final_score": 7.36,
  "final_percentage": 73.6,
  "status": "pass"
}
```

* `results/ranked_list.csv` – at minimum:

```csv
rank,student_name,email,repo_url,final_score,final_percentage,status
```

* `results/index.md` – summary of:

  * Top N repos
  * Histogram or buckets of scores
  * Links to each `feedback.md` file

* `logs/` – optional additional logs per run (e.g., orchestrator logs).

---

## 10. Security, Licensing, and Contribution Notes

* **Security & Privacy**

  * Do **not** push cloned student repos or artifacts to any public remote.
  * All evaluations and artifacts should remain within your controlled environment.
  * Do not upload student code directly to external LLM APIs unless this is explicitly allowed by course policy.

* **Secrets**

  * If any committed secrets are found (API keys, passwords, tokens), the LLM should:

    * Note this explicitly in `feedback.md`.
    * Reduce the **Security & Secrets** score accordingly.
  * The automation agent may optionally produce a separate internal report of secrets for instructors.

* **Licensing**

  * Respect the license of each student repository.
  * This evaluation pipeline itself should have an explicit license (e.g., MIT) specified in its own `LICENSE` file.
  * Do not redistribute student code outside the scope of teaching and grading.

* **Contributions**

  * Changes to this evaluation repo should be via pull requests.
  * Any modifications to the rubric, weights, or tooling should update:

    * This `prompt.md`
    * Any scripts that rely on these settings
  * Keep a `CHANGELOG.md` summarizing tweaks between cohorts.

---

## 11. Summary Instructions for the Coding LLM

When invoked for a particular `student_slug`, you (the coding LLM) will:

1. Receive:

   * Paths or direct contents for `artifacts/` and `reports/` (if pre-populated).
   * The rubric and weights (from this document).
   * Basic metadata (student_name, email, repo_url).
2. Analyze:

   * Test results, coverage, linting and static analysis outputs.
   * Code structure and README (via excerpt or full text).
3. Produce:

   * A structured JSON object matching the `evaluation.json` schema above.
   * A `feedback.md` body with:

     * Scores per criterion
     * Detailed comments
     * Concrete improvement suggestions
4. Return:

   * The JSON and Markdown artifacts so the automation agent can write them to disk.
5. At the end (for the global ranking step):

   * Take all `evaluation.json` objects.
   * Compute ranks, break ties as specified.
   * Generate `results/ranked_list.csv` and `results/index.md`.

Use the scoring rules strictly, and prefer **evidence from artifacts** over assumptions.

## 12. Local runner script (Gemini/Codex orchestration)

Use `python scripts/run_evaluation.py` to drive the cloning/analysis loop without wiring up your own orchestration. The script:

* Parses `submission.csv` and derives a slug from each student name/email plus the repo URL.
* Runs `gh repo clone` inside a temporary directory so you can control workspace noise, then optionally copies the repo back into `work/<student_slug>/repo` with `--keep-clones`.
* Executes the default `gemini evaluate {repo_dir} --output {artifacts_dir}/gemini.json` command (customize via `--gemini-command` or add `--codex-command`/`--extra-command` templates).
* Optional `--format-check` runs a formatting gate first (default: `python -m ruff format --check .`; override with `--format-command`).
* Captures all stdout/stderr to `work/<student_slug>/artifacts/errors.log` so partial output survives failures.
* Reads the scored JSON (default `artifacts/gemini.json`) via `--score-file`/`--score-key` and builds `results/evaluation.json`, `results/ranked_list.csv`, and `results/index.md`. Raw scores are capped to [0,10], converted to percentages, and then normalized: non-terrible submissions are spread across 85–100; clone failures or raw ≤ 1.0 are assigned 0 to surface broken repos.
* Logs activity to `logs/pipeline.log` and supports `--dry-run`/`--limit` flags for incremental testing.

Example:

```bash
python scripts/run_evaluation.py \
  --submission-csv submission.csv \
  --gemini-command "gemini evaluate {repo_dir} --prompt prompt.md --output {artifacts_dir}/gemini.json" \
  --codex-command "codex review {repo_dir} --json {artifacts_dir}/codex.json" \
  --score-file gemini.json \
  --score-key final_score
```

Drop this script into your automation agent after the artifacts are ready so the LLM can focus on `reports/` generation.

### Parallel option

If you prefer a threaded run to speed up evaluation across many repos, use the parallel runner:

```bash
python scripts/run_evaluation_parallel.py \
  --workers 4 \
  --format-check \
  --gemini-command "gemini evaluate {repo_dir} --prompt prompt.md --output {artifacts_dir}/gemini.json" \
  --score-key final_score
```

It produces the same outputs (`results/` plus per-repo artifacts) while executing clones and commands concurrently.
