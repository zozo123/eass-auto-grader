#!/usr/bin/env python3
"""
Generate grading prompts with embedded Pydantic JSON schema.

This ensures the LLM output matches our validated schema.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from models import get_json_schema_str


PROMPT_TEMPLATE = """You are grading a FastAPI CRUD assignment. Analyze the repository thoroughly before scoring. Output valid JSON only - no markdown, code fences, or commentary.

Grading rules:
- Backend MUST be FastAPI. If Flask is used or FastAPI is missing: set framework_compliance.penalty_applied=true, provide penalty_reason, cap functional_correctness and api_design at 3.
- Frontend allowed only if JavaScript or Python Streamlit; penalize other stacks.
- Require full CRUD with correct HTTP verbs/status codes; reflect in crud_completeness (full/partial/minimal/none).
- Require pytest tests; if missing, test_quality <= 2.
- Require Dockerfile; if missing, docker_containerization <= 2.
- Require git hygiene (.gitignore) and dependency file (requirements.txt or pyproject.toml).
- Detect files anywhere in the tree (backend/, app/, src/, etc.).
- Check if README.md is properly formatted with Markdown (headers, lists, code blocks). Set documentation_analysis.readme_formatted=true if so.
- Compute final_score as weighted average (round to 2 decimals).

Scoring weights (0-10 each):
1. functional_correctness (15%) - CRUD works, correct status codes
2. api_design (15%) - RESTful, validation, response models
3. test_quality (15%) - pytest coverage, isolation, error cases
4. architecture_design (10%) - separation of concerns, repository pattern
5. code_quality (10%) - type hints, docstrings, DRY
6. error_handling (10%) - HTTPException, edge cases
7. security_practices (10%) - no secrets in code, env vars
8. documentation (5%) - README, setup instructions
9. docker_containerization (5%) - Dockerfile, compose, best practices
10. dependency_management (5%) - requirements.txt/pyproject.toml, pinned versions

CRITICAL: Your response must be ONLY valid JSON matching this exact schema (no markdown, no explanation, no code fences):

{json_schema}

REMEMBER: Output ONLY the JSON object. No other text before or after."""


def generate_prompt() -> str:
    """Generate the full prompt with JSON schema."""
    return PROMPT_TEMPLATE.format(json_schema=get_json_schema_str())


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate grading prompt with schema")
    parser.add_argument("-o", "--output", type=Path, help="Output file (default: stdout)")
    parser.add_argument("--gemini", action="store_true", help="Output to gemini_prompt.txt")
    parser.add_argument("--codex", action="store_true", help="Output to codex_prompt.txt")
    parser.add_argument("--both", action="store_true", help="Output to both prompt files")
    
    args = parser.parse_args()
    
    prompt = generate_prompt()
    
    if args.both or args.gemini:
        gemini_path = Path(__file__).parent.parent / "gemini_prompt.txt"
        gemini_path.write_text(prompt)
        print(f"✅ Written to {gemini_path}")
    
    if args.both or args.codex:
        codex_path = Path(__file__).parent.parent / "codex_prompt.txt"
        codex_path.write_text(prompt)
        print(f"✅ Written to {codex_path}")
    
    if args.output:
        args.output.write_text(prompt)
        print(f"✅ Written to {args.output}")
    
    if not (args.both or args.gemini or args.codex or args.output):
        print(prompt)


if __name__ == "__main__":
    main()
