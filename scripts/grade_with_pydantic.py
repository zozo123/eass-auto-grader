#!/usr/bin/env python3
"""
Unified grading script with Pydantic structured output.

Uses 3-tier fallback: Gemini CLI → Codex CLI → Local LLM
All providers return validated Pydantic models for guaranteed JSON format.
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from models import GradingResponse, get_json_schema_str

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Prompt template with Pydantic schema
# =============================================================================

SYSTEM_PROMPT = """You are grading a FastAPI CRUD assignment. Analyze the repository thoroughly before scoring. Output valid JSON only - no markdown, code fences, or commentary.

Grading rules:
- Backend MUST be FastAPI. If Flask is used or FastAPI is missing: set framework_compliance.penalty_applied=true, provide penalty_reason, cap functional_correctness and api_design at 3.
- Frontend allowed only if JavaScript or Python Streamlit; penalize other stacks.
- Require full CRUD with correct HTTP verbs/status codes; reflect in crud_completeness (full/partial/minimal/none).
- Require pytest tests; if missing, test_quality <= 2.
- Require Dockerfile; if missing, docker_containerization <= 2.
- Require git hygiene (.gitignore) and dependency file (requirements.txt or pyproject.toml).
- Detect files anywhere in the tree (backend/, app/, src/, etc.).
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

IMPORTANT: Return ONLY valid JSON matching this exact schema:
"""


def get_prompt_with_schema() -> str:
    """Generate the full prompt with embedded JSON schema."""
    return SYSTEM_PROMPT + "\n" + get_json_schema_str()


# =============================================================================
# JSON extraction and repair
# =============================================================================

def extract_json_from_response(text: str) -> str:
    """Extract JSON from potentially messy LLM output."""
    if not text:
        return ""
    
    # Remove markdown code fences
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'```\s*$', '', text, flags=re.MULTILINE)
    
    # Find JSON object
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    
    if first_brace == -1 or last_brace == -1:
        return text.strip()
    
    json_str = text[first_brace:last_brace + 1]
    return json_str


def repair_json(json_str: str) -> str:
    """Attempt to repair malformed JSON."""
    # Balance braces
    open_braces = json_str.count('{')
    close_braces = json_str.count('}')
    open_brackets = json_str.count('[')
    close_brackets = json_str.count(']')
    
    # Add missing closing braces/brackets
    if open_braces > close_braces:
        json_str += '}' * (open_braces - close_braces)
    if open_brackets > close_brackets:
        json_str += ']' * (open_brackets - close_brackets)
    
    # Remove extra closing braces/brackets
    if close_braces > open_braces:
        for _ in range(close_braces - open_braces):
            last_brace = json_str.rfind('}')
            if last_brace > 0:
                json_str = json_str[:last_brace] + json_str[last_brace + 1:]
    
    # Fix trailing commas
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    
    return json_str


def parse_and_validate(raw_output: str) -> Optional[GradingResponse]:
    """Parse LLM output and validate with Pydantic."""
    json_str = extract_json_from_response(raw_output)
    
    # Try parsing as-is first
    try:
        data = json.loads(json_str)
        return GradingResponse.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as e:
        logger.warning(f"First parse attempt failed: {e}")
    
    # Try with repairs
    repaired = repair_json(json_str)
    try:
        data = json.loads(repaired)
        return GradingResponse.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(f"Parse failed after repair: {e}")
        return None


# =============================================================================
# Provider implementations
# =============================================================================

def run_gemini(repo_path: Path, prompt_file: Path, timeout: int = 300) -> Optional[str]:
    """Run Gemini CLI with structured output."""
    logger.info("Trying Gemini CLI...")
    
    cmd = [
        "gemini", "-p", str(prompt_file),
        "--sandbox", "none",
        "--output-format", "json",
        str(repo_path)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=repo_path
        )
        
        if result.returncode == 0 and result.stdout.strip():
            logger.info("Gemini succeeded")
            return result.stdout
        else:
            logger.warning(f"Gemini failed: {result.stderr}")
            return None
    except subprocess.TimeoutExpired:
        logger.warning("Gemini timed out")
        return None
    except FileNotFoundError:
        logger.warning("Gemini CLI not found")
        return None
    except Exception as e:
        logger.warning(f"Gemini error: {e}")
        return None


def run_codex(repo_path: Path, prompt_file: Path, timeout: int = 300) -> Optional[str]:
    """Run Codex CLI with structured output."""
    logger.info("Trying Codex CLI...")
    
    # Read prompt content
    prompt_content = prompt_file.read_text()
    
    cmd = [
        "codex",
        "--full-auto",
        "--output-last-message",
        prompt_content
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=repo_path
        )
        
        if result.returncode == 0 and result.stdout.strip():
            logger.info("Codex succeeded")
            return result.stdout
        else:
            logger.warning(f"Codex failed: {result.stderr}")
            return None
    except subprocess.TimeoutExpired:
        logger.warning("Codex timed out")
        return None
    except FileNotFoundError:
        logger.warning("Codex CLI not found")
        return None
    except Exception as e:
        logger.warning(f"Codex error: {e}")
        return None


# =============================================================================
# Main grading function
# =============================================================================

def grade_repo(
    repo_path: Path,
    output_dir: Path,
    providers: list[str] = None
) -> Optional[GradingResponse]:
    """
    Grade a repository using 3-tier fallback.
    
    Args:
        repo_path: Path to the student repository
        output_dir: Directory to save results
        providers: List of providers to try (default: gemini, codex, local)
    
    Returns:
        Validated GradingResponse or None if all providers fail
    """
    if providers is None:
        providers = ["gemini", "codex"]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create prompt file with Pydantic schema
    prompt_content = get_prompt_with_schema()
    prompt_file = output_dir / "prompt.txt"
    prompt_file.write_text(prompt_content)
    
    provider_funcs = {
        "gemini": run_gemini,
        "codex": run_codex
    }
    
    for provider in providers:
        if provider not in provider_funcs:
            logger.warning(f"Unknown provider: {provider}")
            continue
        
        raw_output = provider_funcs[provider](repo_path, prompt_file)
        
        if raw_output:
            # Save raw output
            raw_file = output_dir / f"{provider}_raw.txt"
            raw_file.write_text(raw_output)
            
            # Parse and validate
            result = parse_and_validate(raw_output)
            
            if result:
                # Save validated JSON
                json_file = output_dir / f"{provider}.json"
                json_file.write_text(result.model_dump_json(indent=2))
                
                # Also save as the canonical result
                final_file = output_dir / "grading_result.json"
                final_file.write_text(result.model_dump_json(indent=2))
                
                logger.info(f"✅ Successfully graded with {provider}")
                logger.info(f"   Final score: {result.final_score}")
                logger.info(f"   Grade: {result.pass_fail.grade.value}")
                
                return result
            else:
                logger.warning(f"❌ {provider} output failed validation")
        else:
            logger.warning(f"❌ {provider} returned no output")
    
    logger.error("All providers failed to grade the repository")
    return None


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Grade a student repository with Pydantic-validated output"
    )
    parser.add_argument(
        "repo_path",
        type=Path,
        help="Path to the student repository"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output directory (default: repo_path/../reports)"
    )
    parser.add_argument(
        "-p", "--providers",
        nargs="+",
        choices=["gemini", "codex"],
        default=["gemini", "codex"],
        help="Providers to try in order"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    repo_path = args.repo_path.resolve()
    if not repo_path.exists():
        logger.error(f"Repository not found: {repo_path}")
        sys.exit(1)
    
    output_dir = args.output or repo_path.parent / "reports"
    
    result = grade_repo(repo_path, output_dir, args.providers)
    
    if result:
        print(f"\n{'='*60}")
        print(f"Repository: {result.repo_name}")
        print(f"Summary: {result.summary}")
        print(f"Final Score: {result.final_score}/10")
        print(f"Grade: {result.pass_fail.grade.value}")
        print(f"{'='*60}\n")
        
        # Print score breakdown
        print("Score Breakdown:")
        for field, value in result.scores.model_dump().items():
            print(f"  {field.replace('_', ' ').title()}: {value}/10")
        
        print(f"\nResults saved to: {output_dir}")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
