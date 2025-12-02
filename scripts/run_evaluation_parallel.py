#!/usr/bin/env python3
"""Parallel orchestrator for Gemini/Codex evaluations using uv-friendly commands."""

import argparse
import concurrent.futures
import csv
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import urlparse


def slugify(student_name: str, repo_url: str) -> str:
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
            # First try parsing as JSON
            try:
                payload = json.loads(response_str)
            except json.JSONDecodeError:
                # Try to extract JSON from response text (e.g., ```json {...} ```)
                json_match = re.search(
                    r'\{[^{}]*"final_score"\s*:\s*(\d+(?:\.\d+)?)[^{}]*\}', response_str
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
                    # Last resort: look for any score pattern like "score: 7" or "7/10"
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


def apply_normalization(entries: List[Dict[str, object]]) -> None:
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


def write_index(entries: List[Dict[str, object]], results_dir: Path) -> None:
    index_path = results_dir / "index.md"
    now = datetime.now(UTC).isoformat()
    lines = [
        "# Ranking overview (parallel run)",
        "",
        f"Generated on {now} UTC",
        "",
        "| Rank | Student | Email | Raw Score | Normalized | Status | Notes | Feedback |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for rank, entry in enumerate(
        sorted(
            entries,
            key=lambda item: (-item["normalized_score"], item["student_name"] or ""),
        ),
        start=1,
    ):
        slug = entry["student_slug"]
        link = f"[feedback](../work/{slug}/reports/{slug}.feedback.md)"
        notes = entry.get("notes") or ""
        lines.append(
            "| {} | {} | {} | {:.2f} | {:.2f} | {} | {} | {} |".format(
                rank,
                entry["student_name"] or "unknown",
                entry["email"] or "n/a",
                entry["score"],
                entry["normalized_score"],
                entry["status"],
                notes,
                link,
            )
        )
    index_path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_row(
    row: Dict[str, str],
    index: int,
    total: int,
    args,
    commands: List[str],
    logger: logging.Logger,
):
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
        return None

    slug = slugify(student_name or email, repo_url)
    logger.info("[%d/%d] %s (%s)", index, total, slug, repo_url)
    work_base = Path(args.work_dir)
    artifacts_dir = (work_base / slug / "artifacts").resolve()
    reports_dir = (work_base / slug / "reports").resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    notes: List[str] = []
    analysis_log: List[Dict[str, object]] = []
    score_value: Optional[float] = None
    clone_failure = False
    tool_failure = False

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
            if args.keep_clones:
                dst_repo = work_base / slug / "repo"
                if dst_repo.exists():
                    shutil.rmtree(dst_repo)
                shutil.copytree(repo_dir, dst_repo)
            for idx, command_template in enumerate(commands, start=1):
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
                log_message = "\n".join(
                    [
                        f"[analysis-{idx}] {command}",
                        f"Exit code: {result.returncode}",
                        f"Stdout: {entry['stdout']}",
                        f"Stderr: {entry['stderr']}",
                    ]
                )
                append_to_log(artifacts_dir / "errors.log", log_message)
                if result.returncode != 0:
                    tool_failure = True
                    if args.format_check and command_template == args.format_command:
                        notes.append("Formatting check failed")
                    else:
                        notes.append(
                            f"{command.split()[0]} returned {result.returncode}"
                        )
                    logger.warning("Command %s returned %d", command, result.returncode)

    if not clone_failure:
        score_value = parse_score(
            Path(artifacts_dir), args.score_file, args.score_key, logger
        )
    if score_value is None:
        score_value = 0.0
        notes.append("Score not found")
    final_score = max(0.0, min(score_value, 10.0))
    final_percentage = round(final_score * 10.0, 2)
    status = (
        "pass"
        if final_score >= 6.0 and not (clone_failure or tool_failure)
        else "needs_attention"
    )
    if clone_failure:
        status = "clone_failed"
    elif tool_failure:
        status = "tool_failed"

    return {
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Gemini/Codex evaluations in parallel."
    )
    parser.add_argument(
        "--submission-csv",
        default="submission.csv",
        help="Path to the CSV that lists student repos.",
    )
    parser.add_argument(
        "--work-dir",
        default="work",
        help="Base path where per-student artifacts are collected.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory that will receive aggregated ranking outputs.",
    )
    parser.add_argument(
        "--logs-dir", default="logs", help="Directory for pipeline logs."
    )
    parser.add_argument(
        "--score-file",
        default="gemini.json",
        help="Relative file under artifacts that contains the numeric score.",
    )
    parser.add_argument(
        "--score-key",
        default="final_score",
        help="Dot-separated key within the score file to turn into the numeric score.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N submissions (for testing).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log the commands without executing clones or analysis.",
    )
    parser.add_argument(
        "--keep-clones",
        action="store_true",
        help="Copy the repository into work/<slug>/repo after analysis.",
    )
    parser.add_argument(
        "--extra-command",
        action="append",
        default=[],
        help="Additional command templates to run after the default Gemini/Codex steps.",
    )
    parser.add_argument(
        "--format-check",
        action="store_true",
        help="Run a formatting check before other commands (default: python -m ruff format --check .).",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Skip cleaning the work directory before starting (default: always clean).",
    )
    parser.add_argument(
        "--format-command",
        default="python -m ruff format --check .",
        help="Template for formatting validation command.",
    )
    parser.add_argument(
        "--gemini-command",
        default="gemini evaluate {repo_dir} --prompt prompt.md --output {artifacts_dir}/gemini.json",
        help="Template for the Gemini CLI invocation.",
    )
    parser.add_argument(
        "--codex-command",
        default=None,
        help="Optional Codex CLI template to run after the Gemini command.",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of concurrent workers."
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=300.0,
        help="Timeout in seconds per command (clone/analysis). Use 0 or negative to disable.",
    )

    args = parser.parse_args()
    submission_path = Path(args.submission_csv)
    if not submission_path.exists():
        raise SystemExit(f"{submission_path} does not exist")

    work_base = Path(args.work_dir)
    results_dir = Path(args.results_dir)
    logs_dir = Path(args.logs_dir)

    # Always clean work directory unless --no-clean is specified
    if not args.no_clean and work_base.exists():
        shutil.rmtree(work_base)
        logging.info("Cleaned work directory: %s", work_base)

    work_base.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / "pipeline_parallel.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    logger = logging.getLogger("ranker_parallel")
    logger.setLevel(logging.INFO)
    logger.handlers[:] = []
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    rows = normalize_rows(submission_path)
    if not rows:
        raise SystemExit("No rows found in the submission CSV")
    if args.limit:
        rows = rows[: args.limit]

    commands: List[str] = []
    if args.format_check:
        commands.append(args.format_command)
    if args.gemini_command:
        commands.append(args.gemini_command)
    if args.codex_command:
        commands.append(args.codex_command)
    commands.extend(args.extra_command)
    if not commands:
        raise SystemExit("At least one analysis command must be specified")

    results: List[Dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        total = len(rows)
        for idx, row in enumerate(rows, start=1):
            futures.append(
                executor.submit(evaluate_row, row, idx, total, args, commands, logger)
            )
        for future in concurrent.futures.as_completed(futures):
            entry = future.result()
            if entry:
                results.append(entry)

    apply_normalization(results)

    evaluation_path = results_dir / "evaluation.json"
    with evaluation_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    ranked_path = results_dir / "ranked_list.csv"
    with ranked_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "student_name",
                "email",
                "repo_url",
                "raw_score",
                "normalized_score",
                "final_percentage",
                "normalized_percentage",
                "status",
                "notes",
            ]
        )
        sorted_entries = sorted(
            results, key=lambda item: (-item["normalized_score"], item["student_name"])
        )
        for rank, entry in enumerate(sorted_entries, start=1):
            writer.writerow(
                [
                    rank,
                    entry["student_name"],
                    entry["email"],
                    entry["repo_url"],
                    f"{entry['score']:.2f}",
                    f"{entry['normalized_score']:.2f}",
                    f"{entry['final_percentage']:.2f}",
                    f"{entry['normalized_percentage']:.2f}",
                    entry["status"],
                    entry["notes"],
                ]
            )

    write_index(results, results_dir)


if __name__ == "__main__":
    main()
