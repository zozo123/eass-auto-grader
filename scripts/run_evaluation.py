#!/usr/bin/env python3
"""Orchestrate GitHub clones and Gemini/Codex evaluations, then summarize scores."""

import argparse
import csv
import json
import logging
import re
import os
import shutil
import signal
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse


def slugify(student_name: str, repo_url: str) -> str:
    """Turn metadata into a filesystem-safe slug."""
    base = student_name or repo_url
    if not base:
        base = "submission"
    base = base.lower().strip()
    base = re.sub(r"[^a-z0-9]+", "_", base)
    base = base.strip("_")
    repo_tail = repo_url.rstrip("/").split("/")[-1] if repo_url else ""
    repo_tail = repo_tail.replace(".git", "")
    parts = [base, repo_tail] if repo_tail else [base]
    slug = "_".join(p for p in parts if p)
    return slug[:64] or "submission"


def get_field(row: Dict[str, str], keys: Iterable[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value:
            return value.strip()
    return ""


def normalize_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        rows = []
        for raw in reader:
            normalized = {
                k.strip(): (v.strip() if v is not None else "")
                for k, v in raw.items()
                if k
            }
            if any(normalized.values()):
                rows.append(normalized)
        return rows


def derive_repo_ref(repo_url: str) -> str:
    if not repo_url:
        return ""
    parsed = urlparse(repo_url)
    if parsed.netloc.endswith("github.com") and parsed.path:
        parts = [segment for segment in parsed.path.split("/") if segment]
        if len(parts) >= 2:
            owner, name = parts[0], parts[1]
            name = name.replace(".git", "")
            return f"{owner}/{name}"
    return repo_url.rstrip("/").replace(".git", "")


def run_command(
    command: str, cwd: Path, dry_run: bool, timeout: Optional[float]
) -> subprocess.CompletedProcess:
    logging.info("Executing: %s", command)
    if dry_run:
        return subprocess.CompletedProcess(
            command, returncode=0, stdout="(dry run)", stderr=""
        )
    try:
        proc = subprocess.Popen(
            command,
            cwd=cwd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            return subprocess.CompletedProcess(
                command, returncode=proc.returncode, stdout=stdout, stderr=stderr
            )
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                stdout, stderr = proc.communicate()
            return subprocess.CompletedProcess(
                command,
                returncode=124,
                stdout=stdout,
                stderr=(stderr or "") + "\nTIMEOUT",
            )
    except Exception as exc:
        logging.exception("Command failed to start: %s", command)
        return subprocess.CompletedProcess(
            command, returncode=1, stdout="", stderr=str(exc)
        )


def append_to_log(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(contents)
        handle.write("\n")


def validate_json_file(path: Path) -> Dict[str, Any]:
    """Quickly validate that a file exists and contains valid JSON."""
    status: Dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "valid": False,
        "error": "",
        "quota_exceeded": False,
        "rate_limited": False,
    }
    if not status["exists"]:
        status["error"] = "missing"
        return status
    try:
        with path.open("r", encoding="utf-8") as handle:
            content = handle.read()
            if not content.strip():
                status["error"] = "empty file"
                return status
            data = json.loads(content)
            
        # Check for quota/rate limit errors in the response
        status["quota_exceeded"], status["rate_limited"] = detect_quota_error(data, content)
        
        if status["quota_exceeded"] or status["rate_limited"]:
            status["error"] = "quota_exceeded" if status["quota_exceeded"] else "rate_limited"
        else:
            status["valid"] = True
    except json.JSONDecodeError as exc:
        status["error"] = f"JSONDecodeError: {exc}"
    except Exception as exc:
        status["error"] = str(exc)
    return status


def detect_quota_error(data: Any, raw_content: str = "") -> tuple[bool, bool]:
    """Detect quota/rate limit errors from AI provider responses.
    
    Handles errors from:
    - Gemini CLI (OAuth-based, outputs JSON with error field)
    - Codex CLI (outputs JSON with status/message)
    - Gemini API (HTTP errors in JSON response)
    - Local LLM (connection errors, model errors)
    
    Returns:
        tuple: (quota_exceeded, rate_limited)
    """
    quota_exceeded = False
    rate_limited = False
    
    # === Quota/billing error patterns ===
    quota_patterns = [
        "quota exceeded",
        "quota_exceeded", 
        "resource exhausted",
        "resourceexhausted",
        "billing",
        "payment required",
        "insufficient_quota",
        "out of quota",
        "api key not valid",
        "api_key_invalid",
        "invalid api key",
        "permission denied",
        "access denied",
        "unauthorized",
        "authentication failed",
    ]
    
    # === Rate limit patterns ===
    rate_limit_patterns = [
        "rate limit",
        "rate_limit",
        "ratelimit",
        "too many requests",
        "429",
        "throttl",
        "slow down",
        "retry after",
        "requests per minute",
        "rpm limit",
        "tpm limit",
        "token limit",
        "context length exceeded",
        "max.* tokens",
    ]
    
    # === CLI-specific error patterns ===
    cli_error_patterns = [
        # Gemini CLI errors
        "error when talking to gemini",
        "failed to connect",
        "connection refused",
        "network error",
        "timeout",
        "timed out",
        "could not resolve",
        "ssl error",
        "certificate",
        # Codex CLI errors  
        "openai error",
        "api error",
        "model not found",
        "model_not_found",
        "service unavailable",
        "internal server error",
        "bad gateway",
        "502", "503", "504",
        # Local LLM errors
        "connection reset",
        "broken pipe",
        "server not responding",
    ]
    
    # Check raw content (case-insensitive)
    content_lower = raw_content.lower()
    
    for pattern in quota_patterns:
        if pattern in content_lower:
            quota_exceeded = True
            break
    
    for pattern in rate_limit_patterns:
        if pattern in content_lower:
            rate_limited = True
            break
    
    # CLI errors often mean we should try another provider
    for pattern in cli_error_patterns:
        if pattern in content_lower:
            # Treat connection/API errors as rate-limit-like (recoverable with different provider)
            rate_limited = True
            break
    
    # Check structured error responses
    if isinstance(data, dict):
        # === Gemini CLI error format ===
        # {"error": {"code": 429, "message": "...", "status": "RESOURCE_EXHAUSTED"}}
        error = data.get("error", {})
        if isinstance(error, dict):
            error_code = str(error.get("code", "")).lower()
            error_status = str(error.get("status", "")).lower()
            error_message = str(error.get("message", "")).lower()
            
            all_error_text = f"{error_code} {error_status} {error_message}"
            
            if any(p in all_error_text for p in ["resource_exhausted", "quota", "billing", "permission"]):
                quota_exceeded = True
            if any(p in all_error_text for p in ["429", "rate", "throttl", "too many"]):
                rate_limited = True
        
        # === Simple error string at top level ===
        if "error" in data and isinstance(data["error"], str):
            error_str = data["error"].lower()
            if any(p in error_str for p in quota_patterns):
                quota_exceeded = True
            if any(p in error_str for p in rate_limit_patterns + cli_error_patterns):
                rate_limited = True
                
        # === Codex CLI error format ===
        # {"status": "error", "message": "..."}
        if data.get("status") == "error":
            msg = str(data.get("message", "")).lower()
            if any(p in msg for p in quota_patterns):
                quota_exceeded = True
            if any(p in msg for p in rate_limit_patterns + cli_error_patterns):
                rate_limited = True
        
        # === Check if response has expected structure ===
        # Valid evaluation should have scores or final_score
        has_valid_structure = (
            "final_score" in data or 
            "scores" in data or
            ("response" in data and len(str(data.get("response", ""))) > 100)
        )
        
        # If we got JSON but it's just an error wrapper, mark as rate limited
        if not has_valid_structure and ("error" in data or data.get("status") == "error"):
            rate_limited = True
    
    return quota_exceeded, rate_limited


def parse_ruff_output(stdout: str) -> Dict[str, Any]:
    """Parse ruff format --check output to extract detailed stats."""
    result = {
        "files_need_formatting": 0,
        "files_ok": 0,
        "total_files": 0,
        "format_ratio": 0.0,
        "files_needing_format": [],
        "status": "unknown",
    }

    if not stdout:
        return result

    lines = stdout.strip().split("\n")
    needs_format = []

    for line in lines:
        line = line.strip()
        if line.startswith("Would reformat:"):
            filename = line.replace("Would reformat:", "").strip()
            needs_format.append(filename)
        # Handle combined line: "4 files would be reformatted, 7 files already formatted"
        # Or single: "7 files already formatted"
        if "would be reformatted" in line:
            match = re.search(r"(\d+)\s+files? would be reformatted", line)
            if match:
                result["files_need_formatting"] = int(match.group(1))
        if "already formatted" in line:
            match = re.search(r"(\d+)\s+files? already formatted", line)
            if match:
                result["files_ok"] = int(match.group(1))

    result["files_needing_format"] = needs_format
    if needs_format and not result["files_need_formatting"]:
        result["files_need_formatting"] = len(needs_format)

    result["total_files"] = result["files_need_formatting"] + result["files_ok"]

    if result["total_files"] > 0:
        result["format_ratio"] = round(
            result["files_ok"] / result["total_files"] * 100, 1
        )

    if result["files_need_formatting"] == 0 and result["total_files"] > 0:
        result["status"] = "perfect"
    elif result["format_ratio"] >= 80:
        result["status"] = "good"
    elif result["format_ratio"] >= 50:
        result["status"] = "needs_work"
    elif result["total_files"] > 0:
        result["status"] = "poor"

    return result


def capture_tree(repo_dir: Path, artifacts_dir: Path, max_depth: int = 4) -> str:
    """Capture repository tree structure and save to artifacts."""
    try:
        result = subprocess.run(
            [
                "tree",
                "-L",
                str(max_depth),
                "-I",
                "__pycache__|.git|node_modules|.venv|venv|*.pyc",
            ],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )
        tree_output = (
            result.stdout
            if result.returncode == 0
            else f"tree command failed: {result.stderr}"
        )
    except FileNotFoundError:
        # tree not installed, use find as fallback
        try:
            result = subprocess.run(
                [
                    "find",
                    ".",
                    "-type",
                    "f",
                    "-name",
                    "*.py",
                    "-o",
                    "-name",
                    "Dockerfile*",
                    "-o",
                    "-name",
                    "*.md",
                    "-o",
                    "-name",
                    "requirements*.txt",
                    "-o",
                    "-name",
                    "pyproject.toml",
                ],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )
            tree_output = "Key files:\n" + result.stdout
        except Exception as e:
            tree_output = f"Could not capture tree: {e}"
    except Exception as e:
        tree_output = f"Could not capture tree: {e}"

    # Save tree to artifacts
    tree_path = artifacts_dir / "tree.txt"
    tree_path.write_text(tree_output, encoding="utf-8")

    return tree_output


def parse_gemini_evaluation(
    artifacts_dir: Path, logger: logging.Logger, filename: str = "gemini.json"
) -> Dict[str, Any]:
    """Parse the comprehensive Gemini evaluation response."""
    gemini_path = artifacts_dir / filename
    if not gemini_path.exists():
        return {}

    try:
        with gemini_path.open("r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except json.JSONDecodeError:
        logger.warning("Could not parse gemini.json")
        return {}

    # Handle Gemini CLI wrapper format
    if isinstance(raw_data, dict) and "response" in raw_data:
        response_str = raw_data["response"]
        if isinstance(response_str, str):
            # Find ALL JSON blocks in the response (model may output multiple iterations)
            # We want the LAST complete one with a valid final_score
            json_blocks = []
            
            # Find all ```json ... ``` blocks
            json_pattern = r'```(?:json)?\s*(\{[\s\S]*?\})\s*```'
            matches = re.findall(json_pattern, response_str)
            
            for match in matches:
                try:
                    parsed = json.loads(match)
                    if isinstance(parsed, dict) and "scores" in parsed:
                        json_blocks.append(parsed)
                except json.JSONDecodeError:
                    continue
            
            # If we found valid JSON blocks, use the last one with best data
            if json_blocks:
                # Prefer the last block with a non-zero final_score
                best_block = None
                for block in reversed(json_blocks):
                    final_score = block.get("final_score", 0)
                    if final_score and float(final_score) > 0:
                        best_block = block
                        break
                # If no block has final_score, take the last one (most complete)
                if best_block is None:
                    best_block = json_blocks[-1]
                raw_data = best_block
            else:
                # Fallback: try to parse the whole response
                clean_response = response_str.strip()
                if clean_response.startswith("```"):
                    clean_response = re.sub(r"^```(?:json)?\s*", "", clean_response)
                    clean_response = re.sub(r"\s*```$", "", clean_response)
                try:
                    raw_data = json.loads(clean_response)
                except json.JSONDecodeError:
                    # Try to find any JSON object in response
                    json_match = re.search(r"\{[\s\S]*\}", response_str)
                    if json_match:
                        try:
                            raw_data = json.loads(json_match.group(0))
                        except json.JSONDecodeError:
                            return {}
                    else:
                        return {}

    # Validate it has the expected structure
    if not isinstance(raw_data, dict):
        return {}

    # Extract and normalize scores
    eval_data = {
        "repo_name": raw_data.get("repo_name", ""),
        "summary": raw_data.get("summary", ""),
        "project_type": raw_data.get("project_type", ""),
        "final_score": float(raw_data.get("final_score", 0)),
        "scores": {},
        "deep_analysis": raw_data.get(
            "deep_analysis", {}
        ),  # New: detailed analysis per dimension
        "tech_stack": raw_data.get("tech_stack", {}),
        "file_inventory": raw_data.get("file_inventory", {}),
        "code_metrics": raw_data.get("code_metrics", {}),
        "issues": raw_data.get("issues", []),
        "strengths": raw_data.get("strengths", []),
        "improvements": raw_data.get("improvements", []),
        "pass_fail": raw_data.get("pass_fail", {}),
    }

    # Score weights for weighted average (must sum to 1.0)
    SCORE_WEIGHTS = {
        "functional_correctness": 0.15,  # 15% - Does it work?
        "architecture_design": 0.10,  # 10% - Is it well-structured?
        "code_quality": 0.10,  # 10% - Is the code clean?
        "api_design": 0.15,  # 15% - Is the API well-designed?
        "error_handling": 0.10,  # 10% - Are errors handled?
        "security_practices": 0.10,  # 10% - Is it secure?
        "test_quality": 0.15,  # 15% - Are there good tests?
        "documentation": 0.05,  # 5% - Is it documented?
        "docker_containerization": 0.05,  # 5% - Is Docker set up?
        "dependency_management": 0.05,  # 5% - Are deps managed?
    }

    # Extract individual scores from both 'scores' and 'deep_analysis'
    score_keys = list(SCORE_WEIGHTS.keys())

    # First try 'scores' field
    if "scores" in raw_data and isinstance(raw_data["scores"], dict):
        for key in score_keys:
            val = raw_data["scores"].get(key)
            if val is not None:
                try:
                    eval_data["scores"][key] = float(val)
                except (ValueError, TypeError):
                    pass

    # Also extract from deep_analysis if present (for backup/validation)
    if "deep_analysis" in raw_data and isinstance(raw_data["deep_analysis"], dict):
        for key in score_keys:
            if key in raw_data["deep_analysis"]:
                da = raw_data["deep_analysis"][key]
                if isinstance(da, dict) and "score" in da:
                    # If score not in main scores, use deep_analysis score
                    if key not in eval_data["scores"]:
                        try:
                            eval_data["scores"][key] = float(da["score"])
                        except (ValueError, TypeError):
                            pass

    # Calculate WEIGHTED average if final_score missing or zero
    if (
        eval_data["final_score"] == 0 or eval_data["final_score"] is None
    ) and eval_data["scores"]:
        weighted_sum = 0.0
        total_weight = 0.0
        for key, weight in SCORE_WEIGHTS.items():
            if key in eval_data["scores"]:
                weighted_sum += eval_data["scores"][key] * weight
                total_weight += weight
        if total_weight > 0:
            eval_data["final_score"] = round(weighted_sum / total_weight, 2)

    return eval_data


def check_project_files(repo_dir: Path) -> Dict[str, Any]:
    """Check for essential project files and return stats."""
    result = {
        "has_readme": False,
        "has_gitignore": False,
        "has_dockerfile": False,
        "has_tests": False,
        "has_requirements": False,
        "readme_file": None,
        "dockerfile_file": None,
        "tests_location": None,
        "score": 0,
        "missing": [],
    }

    # Check for README (case-insensitive)
    for name in ["README.md", "README.rst", "README.txt", "README", "readme.md"]:
        readme_path = repo_dir / name
        if readme_path.exists():
            result["has_readme"] = True
            result["readme_file"] = name
            break
    if not result["has_readme"]:
        result["missing"].append("README")

    # Check for .gitignore
    if (repo_dir / ".gitignore").exists():
        result["has_gitignore"] = True
    else:
        result["missing"].append(".gitignore")

    # Check for Dockerfile (various patterns)
    dockerfile_patterns = [
        "Dockerfile",
        "dockerfile",
        "*.Dockerfile",
        "docker/Dockerfile",
    ]
    for pattern in dockerfile_patterns:
        if pattern.startswith("*"):
            matches = list(repo_dir.glob(pattern))
            if matches:
                result["has_dockerfile"] = True
                result["dockerfile_file"] = matches[0].name
                break
        else:
            docker_path = repo_dir / pattern
            if docker_path.exists():
                result["has_dockerfile"] = True
                result["dockerfile_file"] = pattern
                break
    if not result["has_dockerfile"]:
        result["missing"].append("Dockerfile")

    # Check for tests directory or test files
    test_locations = ["tests", "test", "Tests", "spec"]
    for loc in test_locations:
        if (repo_dir / loc).is_dir():
            result["has_tests"] = True
            result["tests_location"] = loc
            break
    if not result["has_tests"]:
        # Check for test_*.py files in any location
        test_files = list(repo_dir.rglob("test_*.py")) + list(
            repo_dir.rglob("*_test.py")
        )
        if test_files:
            result["has_tests"] = True
            result["tests_location"] = str(test_files[0].relative_to(repo_dir).parent)
    if not result["has_tests"]:
        result["missing"].append("tests")

    # Check for requirements.txt or pyproject.toml
    if (repo_dir / "requirements.txt").exists() or (
        repo_dir / "pyproject.toml"
    ).exists():
        result["has_requirements"] = True

    # Calculate score (out of 5)
    result["score"] = sum(
        [
            result["has_readme"],
            result["has_gitignore"],
            result["has_dockerfile"],
            result["has_tests"],
            result["has_requirements"],
        ]
    )

    return result


def parse_score(
    artifacts_dir: Path, score_file: str, score_key: str, logger: logging.Logger
) -> Optional[float]:
    if not score_file:
        return None
    candidate = artifacts_dir / score_file
    if not candidate.exists():
        logger.debug("Score file %s missing", candidate)
        return None
    try:
        with candidate.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError:
        logger.warning("Unable to decode %s as JSON", candidate)
        return None

    # Handle Gemini CLI output format where response is a JSON string
    if isinstance(payload, dict) and "response" in payload:
        response_str = payload["response"]
        if isinstance(response_str, str):
            # Strip markdown code fences if present
            clean_response = response_str.strip()
            if clean_response.startswith("```"):
                # Remove ```json or ``` at start and ``` at end
                clean_response = re.sub(r"^```(?:json)?\s*", "", clean_response)
                clean_response = re.sub(r"\s*```$", "", clean_response)

            try:
                payload = json.loads(clean_response)
            except json.JSONDecodeError:
                # Try to extract JSON object from response text
                json_match = re.search(
                    r'\{\s*"final_score"\s*:\s*(\d+(?:\.\d+)?)\s*\}', response_str
                )
                if json_match:
                    try:
                        payload = json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        pass
                else:
                    # Try to extract just a number after "final_score"
                    score_match = re.search(
                        r'final_score["\s:]+(\d+(?:\.\d+)?)',
                        response_str,
                        re.IGNORECASE,
                    )
                    if score_match:
                        return float(score_match.group(1))
                    score_match = re.search(
                        r"(?:score|rating)[:\s]+(\d+(?:\.\d+)?)",
                        response_str,
                        re.IGNORECASE,
                    )
                    if score_match:
                        return float(score_match.group(1))
                    logger.warning(
                        "Unable to extract score from response in %s", candidate
                    )
                    return None

    value = payload
    for segment in score_key.split("."):
        if isinstance(value, dict) and segment in value:
            value = value[segment]
        else:
            return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def summarize_notes(notes: List[str]) -> str:
    return " | ".join(notes) if notes else ""


def generate_report(entries: List[Dict[str, Any]], results_dir: Path) -> None:
    """Generate a comprehensive markdown report with insights."""
    now = datetime.now(UTC)
    report_path = results_dir / "REPORT.md"

    # Calculate statistics
    total = len(entries)
    if total == 0:
        return

    passed = sum(1 for e in entries if e.get("status") == "pass")
    clone_failed = sum(1 for e in entries if e.get("status") == "clone_failed")

    scores = [e["score"] for e in entries if e["score"] > 0]
    avg_score = sum(scores) / len(scores) if scores else 0
    max_score = max(scores) if scores else 0
    min_score = min(scores) if scores else 0

    # Format stats
    format_stats = [e.get("ruff_stats", {}) for e in entries if e.get("ruff_stats")]
    perfect_format = sum(1 for s in format_stats if s.get("status") == "perfect")
    good_format = sum(1 for s in format_stats if s.get("status") == "good")
    needs_work_format = sum(1 for s in format_stats if s.get("status") == "needs_work")
    poor_format = sum(1 for s in format_stats if s.get("status") == "poor")

    # Score distribution
    score_10 = sum(1 for s in scores if s >= 9.5)
    score_9 = sum(1 for s in scores if 8.5 <= s < 9.5)
    score_8 = sum(1 for s in scores if 7.5 <= s < 8.5)
    score_7 = sum(1 for s in scores if 6.5 <= s < 7.5)
    score_6 = sum(1 for s in scores if 5.5 <= s < 6.5)
    score_low = sum(1 for s in scores if s < 5.5)

    sorted_entries = sorted(
        entries, key=lambda x: (-x["normalized_score"], x["student_name"])
    )

    lines = [
        "# EASS Student Project Evaluation Report",
        "",
        f"**Generated:** {now.strftime('%Y-%m-%d %H:%M:%S')} UTC",
        f"**Total Submissions:** {total}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total Submissions | {total} |",
        f"| Passed (score ≥6) | {passed} ({passed / total * 100:.1f}%) |",
        f"| Average Score | {avg_score:.2f}/10 |",
        f"| Score Range | {min_score:.1f} - {max_score:.1f} |",
        f"| Clone Failures | {clone_failed} |",
        "",
        "---",
        "",
        "## Code Formatting Analysis (Ruff)",
        "",
        "| Status | Count | Description |",
        "|--------|-------|-------------|",
        f"| Perfect | {perfect_format} | All files properly formatted |",
        f"| Good | {good_format} | >=80% files formatted |",
        f"| Needs Work | {needs_work_format} | 50-79% files formatted |",
        f"| Poor | {poor_format} | <50% files formatted |",
        "",
        "### Detailed Format Stats",
        "",
        "| Student | Formatted | Total | Ratio | Status |",
        "|---------|-----------|-------|-------|--------|",
    ]

    for entry in sorted_entries:
        stats = entry.get("ruff_stats", {})
        if stats and stats.get("total_files", 0) > 0:
            ok = stats.get("files_ok", 0)
            total_f = stats.get("total_files", 0)
            ratio = stats.get("format_ratio", 0)
            status_icon = {
                "perfect": "OK",
                "good": "Good",
                "needs_work": "Fair",
                "poor": "Poor",
            }.get(stats.get("status", ""), "-")
            lines.append(
                f"| {entry['student_name']} | {ok} | {total_f} | {ratio}% | {status_icon} |"
            )

    # Project Files Analysis - use Gemini's file_inventory as source of truth
    # This correctly detects files in nested directories (e.g., backend/Dockerfile)
    pf_total = 0
    has_readme = 0
    has_gitignore = 0
    has_dockerfile = 0
    has_tests = 0
    has_requirements = 0

    for e in entries:
        ge = e.get("gemini_eval", {})
        file_inv = ge.get("file_inventory", {})
        if file_inv:
            pf_total += 1
            if file_inv.get("has_readme"):
                has_readme += 1
            if file_inv.get("has_gitignore"):
                has_gitignore += 1
            if file_inv.get("has_dockerfile"):
                has_dockerfile += 1
            if file_inv.get("has_tests_dir"):
                has_tests += 1
            if file_inv.get("has_requirements") or file_inv.get("has_pyproject"):
                has_requirements += 1
        else:
            # Fallback to naive local check if Gemini data unavailable
            pf = e.get("project_files", {})
            if pf:
                pf_total += 1
                if pf.get("has_readme"):
                    has_readme += 1
                if pf.get("has_gitignore"):
                    has_gitignore += 1
                if pf.get("has_dockerfile"):
                    has_dockerfile += 1
                if pf.get("has_tests"):
                    has_tests += 1
                if pf.get("has_requirements"):
                    has_requirements += 1

    pf_total = pf_total if pf_total > 0 else 1  # avoid division by zero

    lines.extend(
        [
            "",
            "---",
            "",
            "## Project Structure Analysis",
            "",
            "Essential files and directories check:",
            "",
            "| File | Present | Missing |",
            "|------|---------|---------|",
            f"| README.md | {has_readme} ({has_readme / pf_total * 100:.0f}%) | {pf_total - has_readme} |",
            f"| .gitignore | {has_gitignore} ({has_gitignore / pf_total * 100:.0f}%) | {pf_total - has_gitignore} |",
            f"| Dockerfile | {has_dockerfile} ({has_dockerfile / pf_total * 100:.0f}%) | {pf_total - has_dockerfile} |",
            f"| tests/ | {has_tests} ({has_tests / pf_total * 100:.0f}%) | {pf_total - has_tests} |",
            f"| requirements.txt/pyproject.toml | {has_requirements} ({has_requirements / pf_total * 100:.0f}%) | {pf_total - has_requirements} |",
            "",
            "### Students Missing Essential Files",
            "",
        ]
    )

    # Use Gemini's file_inventory (authoritative) instead of naive check
    # This properly detects files in nested directories (e.g., backend/Dockerfile)
    missing_files_entries = []
    for e in sorted_entries:
        ge = e.get("gemini_eval", {})
        file_inv = ge.get("file_inventory", {})
        # Fall back to local check only if Gemini didn't analyze
        if file_inv:
            missing = []
            if not file_inv.get("has_readme"):
                missing.append("README")
            if not file_inv.get("has_gitignore"):
                missing.append(".gitignore")
            if not file_inv.get("has_dockerfile"):
                missing.append("Dockerfile")
            if not file_inv.get("has_tests_dir"):
                missing.append("tests")
            if not file_inv.get("has_requirements") and not file_inv.get("has_pyproject"):
                missing.append("requirements.txt/pyproject.toml")
            if missing:
                missing_files_entries.append((e["student_name"], missing))
        else:
            # Fallback to naive local check if Gemini data unavailable
            local_missing = e.get("project_files", {}).get("missing", [])
            if local_missing:
                missing_files_entries.append((e["student_name"], local_missing))

    if missing_files_entries:
        for name, missing in missing_files_entries:
            lines.append(f"- **{name}**: Missing {', '.join(missing)}")
    else:
        lines.append("*All students have all essential files.*")

    lines.extend(
        [
            "",
            "---",
            "",
            "## Score Distribution",
            "",
            "```",
            f"10 (Excellent) : {'█' * score_10} {score_10}",
            f" 9 (Great)     : {'█' * score_9} {score_9}",
            f" 8 (Good)      : {'█' * score_8} {score_8}",
            f" 7 (Fair)      : {'█' * score_7} {score_7}",
            f" 6 (Pass)      : {'█' * score_6} {score_6}",
            f"<6 (Needs Work): {'█' * score_low} {score_low}",
            "```",
            "",
            "---",
            "",
            "## Detailed Score Breakdown by Category",
            "",
            "Average scores across all submissions:",
            "",
        ]
    )

    # Calculate average scores per category
    score_categories = [
        ("functional_correctness", "Functional Correctness"),
        ("architecture_design", "Architecture & Design"),
        ("code_quality", "Code Quality"),
        ("api_design", "API Design"),
        ("error_handling", "Error Handling"),
        ("security_practices", "Security Practices"),
        ("test_quality", "Test Quality"),
        ("documentation", "Documentation"),
        ("docker_containerization", "Docker/Containerization"),
        ("dependency_management", "Dependency Management"),
    ]

    category_averages = {}
    for key, label in score_categories:
        values = [e.get("gemini_eval", {}).get("scores", {}).get(key) for e in entries]
        values = [v for v in values if v is not None]
        if values:
            category_averages[key] = sum(values) / len(values)

    if category_averages:
        lines.append("| Category | Avg Score | Visual |")
        lines.append("|----------|-----------|--------|")
        for key, label in score_categories:
            avg = category_averages.get(key, 0)
            bar = "█" * int(avg) + "░" * (10 - int(avg))
            lines.append(f"| {label} | {avg:.1f}/10 | {bar} |")
        lines.append("")

    lines.extend(
        [
            "---",
            "",
            "## Final Rankings",
            "",
            "| Rank | Student | Score | Grade | Format | Docker | Tests | Status |",
            "|------|---------|-------|-------|--------|--------|-------|--------|",
        ]
    )

    for rank, entry in enumerate(sorted_entries, start=1):
        ruff = entry.get("ruff_stats", {})
        ge = entry.get("gemini_eval", {})
        pf_grade = ge.get("pass_fail", {})
        file_inv = ge.get("file_inventory", {})

        format_str = (
            f"{ruff.get('format_ratio', 0):.0f}%"
            if ruff and ruff.get("total_files", 0) > 0
            else "N/A"
        )
        status_icon = {
            "pass": "OK",
            "tool_failed": "WARN",
            "clone_failed": "FAIL",
            "needs_attention": "ATTN",
        }.get(entry["status"], "-")
        docker_icon = "Y" if file_inv.get("has_dockerfile") else "N"
        tests_icon = "Y" if file_inv.get("has_tests_dir") else "N"

        medal = ""

        lines.append(
            f"| {rank} | {medal}{entry['student_name']} | {entry['score']:.1f} | {pf_grade.get('grade', '-')} | {format_str} | {docker_icon} | {tests_icon} | {status_icon} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## Detailed Results",
            "",
        ]
    )

    for rank, entry in enumerate(sorted_entries, start=1):
        ruff = entry.get("ruff_stats", {})
        ge = entry.get("gemini_eval", {})
        scores = ge.get("scores", {})
        tech = ge.get("tech_stack", {})

        lines.extend(
            [
                f"### {rank}. {entry['student_name']}",
                "",
                f"- **Email:** {entry['email']}",
                f"- **Repository:** [{entry['repo_url']}]({entry['repo_url']})",
                f"- **Final Score:** {entry['score']:.1f}/10 (Normalized: {entry['normalized_score']:.1f}%)",
                f"- **Status:** {entry['status']}",
            ]
        )

        # Add project summary if available
        if ge.get("summary"):
            lines.append(f"- **Summary:** {ge['summary']}")
        if ge.get("project_type"):
            lines.append(f"- **Project Type:** {ge['project_type']}")

        # Tech stack
        if tech:
            tech_items = []
            if tech.get("framework"):
                tech_items.append(tech["framework"])
            if tech.get("database") and tech["database"] != "none":
                tech_items.append(tech["database"])
            if tech.get("orm") and tech["orm"] != "none":
                tech_items.append(tech["orm"])
            if tech_items:
                lines.append(f"- **Tech Stack:** {', '.join(tech_items)}")

        # Detailed scores
        if scores:
            lines.append("")
            lines.append("**Score Breakdown:**")
            lines.append("")
            lines.append("| Category | Score |")
            lines.append("|----------|-------|")
            for key, label in score_categories:
                if key in scores:
                    lines.append(f"| {label} | {scores[key]:.1f}/10 |")

        # Deep analysis evidence (if available)
        deep_analysis = ge.get("deep_analysis", {})
        if deep_analysis:
            lines.append("")
            lines.append("<details>")
            lines.append(
                "<summary><b>Deep Analysis Evidence</b> (click to expand)</summary>"
            )
            lines.append("")
            for key, label in score_categories:
                if key in deep_analysis and isinstance(deep_analysis[key], dict):
                    da = deep_analysis[key]
                    lines.append(f"**{label}:**")
                    # Show key findings
                    for field, value in da.items():
                        if field != "score" and field != "evidence":
                            if isinstance(value, bool):
                                value = "Yes" if value else "No"
                            lines.append(
                                f"- {field.replace('_', ' ').title()}: {value}"
                            )
                    if da.get("evidence"):
                        lines.append(f"- Evidence: _{da['evidence']}_")
                    lines.append("")
            lines.append("</details>")
            lines.append("")

        # Code formatting
        if ruff and ruff.get("total_files", 0) > 0:
            lines.append("")
            lines.append(
                f"**Code Formatting:** {ruff.get('files_ok', 0)}/{ruff.get('total_files', 0)} files formatted ({ruff.get('format_ratio', 0):.1f}%)"
            )
            if ruff.get("files_needing_format"):
                files_list = ruff["files_needing_format"][:5]
                lines.append(
                    f"  - Files needing format: `{', '.join(files_list)}`{'...' if len(ruff['files_needing_format']) > 5 else ''}"
                )

        # Strengths and improvements
        if ge.get("strengths"):
            lines.append("")
            lines.append("**Strengths:**")
            for s in ge["strengths"][:3]:
                lines.append(f"- {s}")

        if ge.get("improvements"):
            lines.append("")
            lines.append("**Areas for Improvement:**")
            for i in ge["improvements"][:3]:
                lines.append(f"- {i}")

        # Issues
        if ge.get("issues"):
            critical = [i for i in ge["issues"] if i.get("severity") == "critical"]
            major = [i for i in ge["issues"] if i.get("severity") == "major"]
            if critical or major:
                lines.append("")
                lines.append("**Issues Found:**")
                for issue in (critical + major)[:3]:
                    severity = issue.get("severity", "major").upper()
                    lines.append(
                        f"- [{severity}] [{issue.get('category', 'general')}] {issue.get('description', '')}"
                    )

        if entry.get("notes"):
            lines.append("")
            lines.append(f"**Notes:** {entry['notes']}")

        lines.append("")
        lines.append("---")
        lines.append("")

    lines.extend(
        [
            "",
            "## Methodology",
            "",
            "### Evaluation Criteria",
            "Projects were evaluated on 10 dimensions:",
            "",
            "1. **Functional Correctness** - Does the code work? API endpoints functional?",
            "2. **Architecture & Design** - Project structure, separation of concerns",
            "3. **Code Quality** - Readability, naming, DRY, type hints",
            "4. **API Design** - RESTful practices, proper HTTP methods, status codes",
            "5. **Error Handling** - Exception handling, validation, error messages",
            "6. **Security Practices** - Input validation, secrets management",
            "7. **Test Quality** - Test presence, coverage, quality of tests",
            "8. **Documentation** - README quality, docstrings, comments",
            "9. **Docker/Containerization** - Dockerfile quality, compose files",
            "10. **Dependency Management** - requirements.txt/pyproject.toml, versions",
            "",
            "### Additional Checks",
            "- **Code Formatting:** Compliance with Ruff formatter (PEP 8 style)",
            "- **Project Files:** Presence of README, Dockerfile, .gitignore, tests, requirements",
            "- **Tree Structure:** Repository layout analysis",
            "",
            "### Scoring",
            "- Raw scores (0-10) from AI evaluation across 10 dimensions",
            "- Final score is weighted average of all dimensions",
            "- Normalized scores spread across 85-100% for passing submissions",
            "- Grade: A (9-10), B (8-9), C (7-8), D (6-7), F (<6)",
            "",
            "---",
            "",
            f"*Report generated by EASS Ranker Pipeline v2.0*",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def write_index(entries: List[Dict[str, Any]], results_dir: Path) -> None:
    index_path = results_dir / "index.md"
    now = datetime.now(UTC).isoformat()
    lines = [
        "# Ranking Overview",
        "",
        f"Generated on {now} UTC",
        "",
        "See [REPORT.md](REPORT.md) for detailed analysis",
        "",
        "| Rank | Student | Email | Score | Normalized | Format | Files | Status | Notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for rank, entry in enumerate(
        sorted(
            entries,
            key=lambda item: (-item["normalized_score"], item["student_name"] or ""),
        ),
        start=1,
    ):
        ruff = entry.get("ruff_stats", {})
        pf = entry.get("project_files", {})
        format_str = (
            f"{ruff.get('format_ratio', 0):.0f}%"
            if ruff and ruff.get("total_files", 0) > 0
            else "-"
        )
        files_str = f"{pf.get('score', 0)}/5" if pf else "-"
        notes = entry.get("notes") or ""
        lines.append(
            "| {} | {} | {} | {:.1f} | {:.1f}% | {} | {} | {} | {} |".format(
                rank,
                entry["student_name"] or "unknown",
                entry["email"] or "n/a",
                entry["score"],
                entry["normalized_score"],
                format_str,
                files_str,
                entry["status"],
                notes,
            )
        )
    index_path.write_text("\n".join(lines), encoding="utf-8")


def apply_normalization(entries: List[Dict[str, Any]]) -> None:
    """Scale scores into [85, 100] for non-terrible submissions; mark terrible as 0."""
    non_terrible = [
        entry
        for entry in entries
        if entry.get("score", 0) > 1.0 and entry.get("status") not in {"clone_failed"}
    ]
    if non_terrible:
        min_score = min(entry["score"] for entry in non_terrible)
        max_score = max(entry["score"] for entry in non_terrible)
    else:
        min_score = max_score = 0.0

    for entry in entries:
        raw = float(entry.get("score", 0.0))
        status = entry.get("status", "")
        is_terrible = raw <= 1.0 or status == "clone_failed"
        if is_terrible:
            normalized = 0.0
        elif max_score == min_score:
            normalized = 100.0
        else:
            ratio = (raw - min_score) / (max_score - min_score)
            normalized = 85.0 + ratio * 15.0
            normalized = max(85.0, min(100.0, normalized))
        entry["normalized_score"] = round(normalized, 2)
        entry["normalized_percentage"] = round(normalized, 2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Gemini/Codex evaluations for student repos."
    )
    parser.add_argument(
        "--submission-csv", default="submission.csv", help="Path to the CSV."
    )
    parser.add_argument("--work-dir", default="work", help="Base path for artifacts.")
    parser.add_argument(
        "--results-dir", default="results", help="Directory for outputs."
    )
    parser.add_argument("--logs-dir", default="logs", help="Directory for logs.")
    parser.add_argument("--score-file", default="gemini.json", help="Score file name.")
    parser.add_argument(
        "--score-key", default="final_score", help="Key for score in JSON."
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Process only first N submissions."
    )
    parser.add_argument("--dry-run", action="store_true", help="Log without executing.")
    parser.add_argument(
        "--keep-clones", action="store_true", help="Copy repos to work dir."
    )
    parser.add_argument(
        "--extra-command", action="append", default=[], help="Additional commands."
    )
    parser.add_argument(
        "--format-check", action="store_true", help="Run formatting check."
    )
    parser.add_argument(
        "--no-clean", action="store_true", help="Skip cleaning work dir."
    )
    parser.add_argument(
        "--format-command",
        default="python -m ruff format --check .",
        help="Format command.",
    )
    parser.add_argument(
        "--gemini-command",
        default="",
        help="Gemini command.",
    )
    parser.add_argument("--codex-command", default="", help="Codex command.")
    parser.add_argument(
        "--fallback-codex",
        action="store_true",
        help="If Gemini fails, fall back to Codex command.",
    )
    parser.add_argument(
        "--timeout-seconds", type=float, default=300.0, help="Timeout per command."
    )

    args = parser.parse_args()
    submission_path = Path(args.submission_csv)
    if not submission_path.exists():
        raise SystemExit(f"{submission_path} does not exist")

    work_base = Path(args.work_dir)
    results_dir = Path(args.results_dir)
    logs_dir = Path(args.logs_dir)

    if not args.no_clean and work_base.exists():
        shutil.rmtree(work_base)
        logging.info("Cleaned work directory: %s", work_base)

    work_base.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / "pipeline.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    logger = logging.getLogger("ranker")
    logger.setLevel(logging.INFO)
    logger.handlers[:] = []
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    rows = normalize_rows(submission_path)
    if not rows:
        raise SystemExit("No rows found in the submission CSV")
    if args.limit:
        rows = rows[: args.limit]

    analysis_commands: List[tuple[str, str]] = []
    if args.format_check:
        analysis_commands.append(("format", args.format_command))

    primary_label = "gemini" if args.gemini_command else "codex"
    primary_command = args.gemini_command or args.codex_command
    if primary_command:
        analysis_commands.append((primary_label, primary_command))

    fallback_command = (
        args.codex_command if args.gemini_command and args.fallback_codex else None
    )

    analysis_commands.extend([("extra", cmd) for cmd in args.extra_command])

    if not analysis_commands:
        raise SystemExit("At least one analysis command must be specified")

    results: List[Dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        student_name = get_field(
            row, ["student_name", "What is your full name?", "student", "name"]
        )
        email = get_field(row, ["email", "Email Address"])
        repo_url = get_field(
            row,
            [
                "repo_url",
                "What is the link to the github project on EASS github?",
                "Repo URL",
                "GitHub Repo",
            ],
        )

        if not repo_url:
            logger.warning("Skipping row %d because repo_url is missing", index)
            continue

        slug = slugify(student_name or email, repo_url)
        logger.info("[%d/%d] %s (%s)", index, len(rows), slug, repo_url)
        artifacts_dir = (work_base / slug / "artifacts").resolve()
        reports_dir = (work_base / slug / "reports").resolve()
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        notes: List[str] = []
        analysis_log: List[Dict[str, Any]] = []
        score_value: Optional[float] = None
        clone_failure = False
        ruff_stats: Dict[str, Any] = {}
        project_files: Dict[str, Any] = {}
        gemini_validation: Dict[str, Any] = {}
        codex_validation: Dict[str, Any] = {}
        gemini_command_ran = False
        codex_command_ran = False
        score_file_used = args.score_file

        with tempfile.TemporaryDirectory(prefix=f"{slug}-") as temp_root:
            repo_dir = Path(temp_root) / "repo"
            manifest = derive_repo_ref(repo_url)
            clone_cmd = f"gh repo clone {manifest} {repo_dir}"
            timeout = (
                None
                if args.timeout_seconds and args.timeout_seconds <= 0
                else args.timeout_seconds
            )
            clone_result = run_command(
                clone_cmd, cwd=Path(temp_root), dry_run=args.dry_run, timeout=timeout
            )

            analysis_log.append(
                {
                    "step": "clone",
                    "command": clone_cmd,
                    "returncode": clone_result.returncode,
                    "stdout": (clone_result.stdout or "").strip(),
                    "stderr": (clone_result.stderr or "").strip(),
                }
            )

            if clone_result.returncode != 0:
                clone_failure = True
                notes.append("Clone failed")
                append_to_log(
                    artifacts_dir / "errors.log",
                    f"[clone] {clone_cmd}\n{clone_result.stderr}",
                )
                logger.error("Unable to clone %s", repo_url)
            else:
                # Capture tree structure
                capture_tree(repo_dir, artifacts_dir)

                # Quick local scan for project files (may miss complex layouts)
                # Gemini's file_inventory will be the authoritative source
                project_files = check_project_files(repo_dir)

                # Save project files check (for reference only)
                project_files_path = artifacts_dir / "project_files.json"
                with project_files_path.open("w", encoding="utf-8") as f:
                    json.dump(project_files, f, indent=2)

                if args.keep_clones:
                    dst_repo = work_base / slug / "repo"
                    if dst_repo.exists():
                        shutil.rmtree(dst_repo)
                    shutil.copytree(repo_dir, dst_repo)

                for idx, (command_label, command_template) in enumerate(
                    analysis_commands, start=1
                ):
                    context = {
                        "repo_dir": repo_dir,
                        "artifacts_dir": artifacts_dir,
                        "student_slug": slug,
                        "repo_url": repo_url,
                        "student_name": student_name,
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
                    command = command_template.format(
                        **{k: str(v) for k, v in context.items()}
                    )
                    start = datetime.now(UTC)
                    result = run_command(
                        command, cwd=repo_dir, dry_run=args.dry_run, timeout=timeout
                    )
                    duration = (datetime.now(UTC) - start).total_seconds()

                    entry = {
                        "step": f"analysis-{idx}",
                        "command": command,
                        "returncode": result.returncode,
                        "stdout": (result.stdout or "").strip(),
                        "stderr": (result.stderr or "").strip(),
                        "duration_seconds": duration,
                    }
                    analysis_log.append(entry)

                    log_message = f"[analysis-{idx}] {command}\nExit: {result.returncode}\nStdout: {entry['stdout']}\nStderr: {entry['stderr']}"
                    append_to_log(artifacts_dir / "errors.log", log_message)

                    # Parse ruff output for detailed stats
                    if args.format_check and command_label == "format":
                        ruff_stats = parse_ruff_output(result.stdout or "")
                        ruff_json_path = artifacts_dir / "ruff_stats.json"
                        with ruff_json_path.open("w", encoding="utf-8") as f:
                            json.dump(ruff_stats, f, indent=2)

                        if result.returncode != 0:
                            notes.append(
                                f"Format: {ruff_stats['files_ok']}/{ruff_stats['total_files']} OK ({ruff_stats['format_ratio']}%)"
                            )
                            logger.warning(
                                "Ruff: %d/%d files need formatting",
                                ruff_stats["files_need_formatting"],
                                ruff_stats["total_files"],
                            )
                    elif command_label in {"gemini", "codex"}:
                        validation_target = (
                            args.score_file
                            if command_label == "gemini"
                            else "codex.json"
                        )
                        validation = validate_json_file(
                            artifacts_dir / validation_target
                        )
                        
                        # Also check stderr for CLI errors (rate limits, connection issues)
                        stderr_lower = (result.stderr or "").lower()
                        if not validation.get("quota_exceeded") and not validation.get("rate_limited"):
                            # Check stderr for error patterns
                            stderr_quota, stderr_rate = detect_quota_error({}, stderr_lower)
                            if stderr_quota:
                                validation["quota_exceeded"] = True
                                validation["error"] = validation.get("error", "") + " (stderr: quota)"
                            if stderr_rate:
                                validation["rate_limited"] = True
                                validation["error"] = validation.get("error", "") + " (stderr: rate limit)"
                        
                        # Also mark as error if command failed and output is missing/empty
                        if result.returncode != 0 and not validation.get("valid"):
                            if not validation.get("rate_limited") and not validation.get("quota_exceeded"):
                                validation["rate_limited"] = True  # Treat CLI failure as rate-limit-like
                                validation["error"] = f"CLI exit code {result.returncode}"
                        
                        analysis_log.append(
                            {
                                "step": f"validate-json-{command_label}",
                                **validation,
                            }
                        )
                        # Log quota/rate limit issues
                        if validation.get("quota_exceeded"):
                            logger.warning("%s quota exceeded for %s", command_label.title(), slug)
                        elif validation.get("rate_limited"):
                            logger.warning("%s rate limited/error for %s", command_label.title(), slug)
                        elif validation.get("exists") and not validation.get("valid"):
                            notes.append("AI output invalid JSON")
                            append_to_log(
                                artifacts_dir / "errors.log",
                                f"[validate-json-{command_label}] {validation}",
                            )
                        if command_label == "gemini":
                            gemini_validation = validation
                            gemini_command_ran = True
                        if command_label == "codex":
                            codex_validation = validation
                            codex_command_ran = True
                    elif result.returncode != 0:
                        notes.append(f"{command.split()[0]} error")
                        logger.warning(
                            "Command %s returned %d", command, result.returncode
                        )

                # Conditional fallback: if Gemini failed and fallback enabled, run Codex
                if (
                    command_label == "gemini"
                    and not clone_failure
                    and args.fallback_codex
                    and args.codex_command
                ):
                    # Check if Gemini succeeded or hit quota/rate limits
                    gemini_ok = (
                        result.returncode == 0 and gemini_validation.get("valid", False)
                    )
                    quota_issue = (
                        gemini_validation.get("quota_exceeded", False) or 
                        gemini_validation.get("rate_limited", False)
                    )
                    
                    # Log quota/rate limit issues
                    if quota_issue:
                        if gemini_validation.get("quota_exceeded"):
                            logger.warning("Gemini quota exceeded, switching to fallback")
                            notes.append("Gemini quota exceeded")
                        elif gemini_validation.get("rate_limited"):
                            logger.warning("Gemini rate limited, switching to fallback")
                            notes.append("Gemini rate limited")
                    
                    if not gemini_ok or quota_issue:
                        fallback_cmd = args.codex_command.format(
                            **{k: str(v) for k, v in context.items()}
                        )
                        start = datetime.now(UTC)
                        fallback_result = run_command(
                            fallback_cmd,
                            cwd=repo_dir,
                            dry_run=args.dry_run,
                            timeout=timeout,
                        )
                        duration_fallback = (datetime.now(UTC) - start).total_seconds()
                        entry_fb = {
                            "step": f"analysis-{idx}-fallback-codex",
                            "command": fallback_cmd,
                            "returncode": fallback_result.returncode,
                            "stdout": (fallback_result.stdout or "").strip(),
                            "stderr": (fallback_result.stderr or "").strip(),
                            "duration_seconds": duration_fallback,
                        }
                        analysis_log.append(entry_fb)
                        append_to_log(
                            artifacts_dir / "errors.log",
                            f"[analysis-{idx}-fallback-codex] {fallback_cmd}\nExit: {fallback_result.returncode}\nStdout: {entry_fb['stdout']}\nStderr: {entry_fb['stderr']}",
                        )

                        codex_validation = validate_json_file(
                            artifacts_dir / "codex.json"
                        )
                        analysis_log.append(
                            {"step": "validate-json-codex", **codex_validation}
                        )
                        codex_command_ran = True

                        # Check for quota issues in Codex response too
                        if codex_validation.get("quota_exceeded"):
                            notes.append("Codex quota exceeded")
                            logger.warning("Codex also hit quota limits")
                        elif codex_validation.get("rate_limited"):
                            notes.append("Codex rate limited")
                            logger.warning("Codex also hit rate limits")
                        elif (
                            codex_validation.get("exists")
                            and not codex_validation.get("valid")
                        ):
                            notes.append("AI output invalid JSON (fallback)")
                            append_to_log(
                                artifacts_dir / "errors.log",
                                f"[validate-json-codex] {codex_validation}",
                            )

        # Parse comprehensive Gemini evaluation
        gemini_eval: Dict[str, Any] = {}
        if not clone_failure:
            # Choose which AI output to read: prefer Gemini, else valid Codex fallback
            score_filename = args.score_file
            if args.fallback_codex and codex_command_ran:
                gemini_ok = gemini_validation.get("valid", False)
                gemini_quota_issue = (
                    gemini_validation.get("quota_exceeded", False) or
                    gemini_validation.get("rate_limited", False)
                )
                codex_ok = codex_validation.get("valid", False)
                codex_quota_issue = (
                    codex_validation.get("quota_exceeded", False) or
                    codex_validation.get("rate_limited", False)
                )
                
                # Use Codex if Gemini failed/quota OR if Codex is valid and Gemini isn't
                if (not gemini_ok or gemini_quota_issue) and codex_ok and not codex_quota_issue:
                    score_filename = "codex.json"
                    if gemini_quota_issue:
                        notes.append("Used Codex fallback (Gemini quota)")
                    else:
                        notes.append("Used Codex fallback")
                elif codex_quota_issue:
                    notes.append("Warning: Both providers hit quota limits")
            gemini_eval = parse_gemini_evaluation(
                Path(artifacts_dir), logger, filename=score_filename
            )
            score_value = gemini_eval.get("final_score")
            if score_value is None:
                score_value = parse_score(
                    Path(artifacts_dir), score_filename, args.score_key, logger
                )

            # Use Gemini's file_inventory as source of truth (it analyzes the full tree)
            gemini_file_inv = gemini_eval.get("file_inventory", {})
            if gemini_file_inv:
                missing_from_gemini = []
                if not gemini_file_inv.get("has_readme"):
                    missing_from_gemini.append("README")
                if not gemini_file_inv.get("has_dockerfile"):
                    missing_from_gemini.append("Dockerfile")
                if not gemini_file_inv.get("has_gitignore"):
                    missing_from_gemini.append(".gitignore")
                if not gemini_file_inv.get("has_tests_dir"):
                    missing_from_gemini.append("tests")
                if missing_from_gemini:
                    notes.append(f"Missing: {', '.join(missing_from_gemini)}")
                    logger.info(
                        "Gemini detected missing: %s", ", ".join(missing_from_gemini)
                    )

        if score_value is None:
            score_value = 0.0
            notes.append("Score not found")

        final_score = max(0.0, min(score_value, 10.0))
        final_percentage = round(final_score * 10.0, 2)

        if clone_failure:
            status = "clone_failed"
        elif score_value == 0.0:
            status = "needs_attention"
        elif final_score >= 6.0:
            status = "pass"
        else:
            status = "needs_attention"

        entry = {
            "student_name": student_name or email or "unknown",
            "email": email,
            "repo_url": repo_url,
            "student_slug": slug,
            "score": final_score,
            "final_percentage": final_percentage,
            "status": status,
            "notes": summarize_notes(notes),
            "artifacts_dir": str(artifacts_dir),
            "analysis": analysis_log,
            "ruff_stats": ruff_stats,
            "project_files": project_files,
            "gemini_eval": gemini_eval,  # Full Gemini evaluation data
        }
        results.append(entry)

    apply_normalization(results)

    evaluation_path = results_dir / "evaluation.json"
    with evaluation_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    # Create detailed CSV with all score dimensions
    ranked_path = results_dir / "ranked_list.csv"
    with ranked_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        # Comprehensive headers
        headers = [
            "rank",
            "student_name",
            "email",
            "repo_url",
            "final_score",
            "normalized_score",
            "functional",
            "architecture",
            "code_quality",
            "api_design",
            "error_handling",
            "security",
            "testing",
            "documentation",
            "docker",
            "dependencies",
            "format_ratio",
            "project_files",
            "status",
            "grade",
            "notes",
        ]
        writer.writerow(headers)
        sorted_entries = sorted(
            results, key=lambda item: (-item["normalized_score"], item["student_name"])
        )
        for rank, entry in enumerate(sorted_entries, start=1):
            ruff = entry.get("ruff_stats", {})
            pf = entry.get("project_files", {})
            ge = entry.get("gemini_eval", {})
            scores = ge.get("scores", {})
            pf_grade = ge.get("pass_fail", {})

            writer.writerow(
                [
                    rank,
                    entry["student_name"],
                    entry["email"],
                    entry["repo_url"],
                    f"{entry['score']:.2f}",
                    f"{entry['normalized_score']:.2f}",
                    scores.get("functional_correctness", ""),
                    scores.get("architecture_design", ""),
                    scores.get("code_quality", ""),
                    scores.get("api_design", ""),
                    scores.get("error_handling", ""),
                    scores.get("security_practices", ""),
                    scores.get("test_quality", ""),
                    scores.get("documentation", ""),
                    scores.get("docker_containerization", ""),
                    scores.get("dependency_management", ""),
                    f"{ruff.get('format_ratio', 0):.1f}%",
                    f"{pf.get('score', 0)}/5" if pf else "N/A",
                    entry["status"],
                    pf_grade.get("grade", ""),
                    entry["notes"],
                ]
            )

    write_index(results, results_dir)
    generate_report(results, results_dir)

    logger.info("Pipeline complete. See results/REPORT.md for detailed analysis.")


if __name__ == "__main__":
    main()
