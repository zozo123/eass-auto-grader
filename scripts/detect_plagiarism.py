#!/usr/bin/env python3
"""Code plagiarism detection for student repositories.

This script focuses on finding actual code copying/cheating by:
1. Comparing code at the line/block level (not just semantics)
2. Detecting identical or near-identical functions
3. Finding suspicious variable/function naming patterns
4. Computing code fingerprints using winnowing algorithm
5. Detecting copy-paste with minor modifications

Uses local embeddings + Qdrant for code-level similarity.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Tuple, Dict, Set, Optional, Any
from uuid import uuid4

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache"}

# Common boilerplate that shouldn't count as plagiarism
BOILERPLATE_PATTERNS = [
    r"from fastapi import",
    r"from pydantic import",
    r"import uvicorn",
    r"if __name__ == [\"']__main__[\"']",
    r"app = FastAPI\(",
    r"@app\.(get|post|put|patch|delete)",
    r"HTTPException",
    r"status_code=",
]

# Minimum thresholds for flagging
MIN_FUNCTION_LENGTH = 5  # lines
MIN_SIMILARITY_THRESHOLD = 0.85  # for exact matches
SUSPICIOUS_THRESHOLD = 0.70  # for suspicious similarity


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class CodeBlock:
    """A block of code (function, class, or segment)."""
    name: str
    code: str
    normalized: str  # whitespace/comment stripped
    fingerprint: str  # hash for quick comparison
    line_count: int
    file_path: str
    block_type: str  # "function", "class", "segment"
    
    
@dataclass
class CodeMatch:
    """A match between two code blocks."""
    repo1: str
    repo2: str
    file1: str
    file2: str
    block1_name: str
    block2_name: str
    similarity: float
    match_type: str  # "exact", "near_exact", "suspicious", "structural"
    code1_preview: str
    code2_preview: str
    evidence: List[str]


@dataclass
class RepoCodeProfile:
    """Code profile for plagiarism detection."""
    name: str
    path: Path
    code_blocks: List[CodeBlock]
    all_code_normalized: str
    function_names: Set[str]
    variable_names: Set[str]
    string_literals: Set[str]
    code_fingerprints: Set[str]
    total_lines: int
    

# ---------------------------------------------------------------------------
# Code Normalization & Fingerprinting
# ---------------------------------------------------------------------------
def normalize_code(code: str) -> str:
    """Normalize code by removing comments, docstrings, and normalizing whitespace."""
    # Remove comments
    code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
    # Remove docstrings (triple quotes)
    code = re.sub(r'"""[\s\S]*?"""', '', code)
    code = re.sub(r"'''[\s\S]*?'''", '', code)
    # Normalize whitespace
    lines = [line.strip() for line in code.splitlines() if line.strip()]
    return '\n'.join(lines)


def normalize_identifiers(code: str) -> str:
    """Replace all identifiers with generic names for structural comparison."""
    # This helps detect code that was copied and had variables renamed
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    
    # Collect all names
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
    
    # Replace with generic names (preserving which are the same)
    normalized = code
    name_map = {}
    counter = 0
    for name in sorted(names, key=len, reverse=True):  # longest first to avoid partial replacements
        if name not in name_map and not name.startswith('_') and name not in {'self', 'cls', 'True', 'False', 'None'}:
            name_map[name] = f"VAR{counter}"
            counter += 1
    
    for orig, replacement in name_map.items():
        # Use word boundaries to avoid partial replacements
        normalized = re.sub(rf'\b{re.escape(orig)}\b', replacement, normalized)
    
    return normalized


def compute_fingerprint(code: str) -> str:
    """Compute a hash fingerprint for code block."""
    normalized = normalize_code(code)
    return hashlib.md5(normalized.encode()).hexdigest()


def compute_ngram_fingerprints(code: str, n: int = 5) -> Set[str]:
    """Compute n-gram fingerprints using winnowing algorithm."""
    normalized = normalize_code(code)
    tokens = normalized.split()
    
    if len(tokens) < n:
        return {hashlib.md5(' '.join(tokens).encode()).hexdigest()}
    
    fingerprints = set()
    for i in range(len(tokens) - n + 1):
        ngram = ' '.join(tokens[i:i+n])
        fp = hashlib.md5(ngram.encode()).hexdigest()[:8]
        fingerprints.add(fp)
    
    return fingerprints


# ---------------------------------------------------------------------------
# AST Analysis for Code Extraction
# ---------------------------------------------------------------------------
def extract_code_blocks(source: str, file_path: str) -> List[CodeBlock]:
    """Extract functions and classes as code blocks."""
    blocks = []
    lines = source.splitlines()
    
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # If can't parse, treat whole file as one block
        normalized = normalize_code(source)
        if len(normalized.splitlines()) >= MIN_FUNCTION_LENGTH:
            blocks.append(CodeBlock(
                name="<file>",
                code=source[:2000],
                normalized=normalized[:2000],
                fingerprint=compute_fingerprint(source),
                line_count=len(lines),
                file_path=file_path,
                block_type="segment"
            ))
        return blocks
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                start = node.lineno - 1
                end = node.end_lineno
                code = '\n'.join(lines[start:end])
                
                if end - start >= MIN_FUNCTION_LENGTH:
                    normalized = normalize_code(code)
                    blocks.append(CodeBlock(
                        name=node.name,
                        code=code,
                        normalized=normalized,
                        fingerprint=compute_fingerprint(code),
                        line_count=end - start,
                        file_path=file_path,
                        block_type="function"
                    ))
        
        elif isinstance(node, ast.ClassDef):
            if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                start = node.lineno - 1
                end = node.end_lineno
                code = '\n'.join(lines[start:end])
                
                if end - start >= MIN_FUNCTION_LENGTH:
                    normalized = normalize_code(code)
                    blocks.append(CodeBlock(
                        name=node.name,
                        code=code,
                        normalized=normalized,
                        fingerprint=compute_fingerprint(code),
                        line_count=end - start,
                        file_path=file_path,
                        block_type="class"
                    ))
    
    return blocks


def extract_names_and_literals(source: str) -> Tuple[Set[str], Set[str], Set[str]]:
    """Extract function names, variable names, and string literals."""
    functions = set()
    variables = set()
    strings = set()
    
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return functions, variables, strings
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            variables.add(node.id)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if len(node.value) > 10:  # Only meaningful strings
                strings.add(node.value[:100])
    
    return functions, variables, strings


# ---------------------------------------------------------------------------
# Similarity Computation
# ---------------------------------------------------------------------------
def code_similarity(code1: str, code2: str) -> float:
    """Compute similarity between two code blocks using SequenceMatcher."""
    return SequenceMatcher(None, code1, code2).ratio()


def structural_similarity(code1: str, code2: str) -> float:
    """Compare code structure after normalizing identifiers."""
    norm1 = normalize_identifiers(code1)
    norm2 = normalize_identifiers(code2)
    return SequenceMatcher(None, norm1, norm2).ratio()


def fingerprint_overlap(fp1: Set[str], fp2: Set[str]) -> float:
    """Compute Jaccard similarity of fingerprint sets."""
    if not fp1 or not fp2:
        return 0.0
    intersection = len(fp1 & fp2)
    union = len(fp1 | fp2)
    return intersection / union if union > 0 else 0.0


def is_boilerplate(code: str) -> bool:
    """Check if code is likely boilerplate."""
    for pattern in BOILERPLATE_PATTERNS:
        if re.search(pattern, code):
            return True
    return False


# ---------------------------------------------------------------------------
# Repository Analysis
# ---------------------------------------------------------------------------
def analyze_repo_for_plagiarism(repo_path: Path, repo_name: str) -> RepoCodeProfile:
    """Analyze repository for plagiarism detection."""
    all_blocks: List[CodeBlock] = []
    all_code = []
    all_functions = set()
    all_variables = set()
    all_strings = set()
    all_fingerprints = set()
    total_lines = 0
    
    # Find all Python files
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        
        for file_name in files:
            if not file_name.endswith('.py'):
                continue
            
            full_path = Path(root) / file_name
            rel_path = str(full_path.relative_to(repo_path))
            
            # Skip test files for plagiarism (tests are expected to be similar)
            if 'test' in file_name.lower() or 'test' in rel_path.lower():
                continue
            
            try:
                source = full_path.read_text(encoding='utf-8', errors='ignore')
            except OSError:
                continue
            
            total_lines += len(source.splitlines())
            all_code.append(source)
            
            # Extract code blocks
            blocks = extract_code_blocks(source, rel_path)
            all_blocks.extend(blocks)
            
            # Extract names and literals
            funcs, vars_, strings = extract_names_and_literals(source)
            all_functions.update(funcs)
            all_variables.update(vars_)
            all_strings.update(strings)
            
            # Compute fingerprints
            fps = compute_ngram_fingerprints(source)
            all_fingerprints.update(fps)
    
    # Combine all code for overall comparison
    combined_code = '\n'.join(all_code)
    normalized_all = normalize_code(combined_code)
    
    return RepoCodeProfile(
        name=repo_name,
        path=repo_path,
        code_blocks=all_blocks,
        all_code_normalized=normalized_all[:50000],  # Limit size
        function_names=all_functions,
        variable_names=all_variables,
        string_literals=all_strings,
        code_fingerprints=all_fingerprints,
        total_lines=total_lines,
    )


# ---------------------------------------------------------------------------
# Plagiarism Detection
# ---------------------------------------------------------------------------
def find_code_matches(profile1: RepoCodeProfile, profile2: RepoCodeProfile) -> List[CodeMatch]:
    """Find suspicious code matches between two repositories."""
    matches = []
    
    # 1. Check for exact fingerprint matches (copy-paste detection)
    exact_blocks1 = {b.fingerprint: b for b in profile1.code_blocks}
    exact_blocks2 = {b.fingerprint: b for b in profile2.code_blocks}
    
    common_fingerprints = set(exact_blocks1.keys()) & set(exact_blocks2.keys())
    for fp in common_fingerprints:
        b1 = exact_blocks1[fp]
        b2 = exact_blocks2[fp]
        
        # Skip if it's boilerplate
        if is_boilerplate(b1.code):
            continue
        
        matches.append(CodeMatch(
            repo1=profile1.name,
            repo2=profile2.name,
            file1=b1.file_path,
            file2=b2.file_path,
            block1_name=b1.name,
            block2_name=b2.name,
            similarity=1.0,
            match_type="exact",
            code1_preview=b1.code[:500],
            code2_preview=b2.code[:500],
            evidence=["Identical code fingerprint (exact copy)"]
        ))
    
    # 2. Check for near-exact matches (code with minor changes)
    for b1 in profile1.code_blocks:
        if is_boilerplate(b1.code):
            continue
            
        for b2 in profile2.code_blocks:
            if b1.fingerprint == b2.fingerprint:
                continue  # Already handled
            if is_boilerplate(b2.code):
                continue
            
            # Quick filter: similar line counts
            if abs(b1.line_count - b2.line_count) > max(5, b1.line_count * 0.3):
                continue
            
            # Compare normalized code
            sim = code_similarity(b1.normalized, b2.normalized)
            
            if sim >= MIN_SIMILARITY_THRESHOLD:
                evidence = [f"Code similarity: {sim:.1%}"]
                
                # Check structural similarity (detects variable renaming)
                struct_sim = structural_similarity(b1.code, b2.code)
                if struct_sim > sim:
                    evidence.append(f"Structural similarity after normalization: {struct_sim:.1%}")
                    evidence.append("Possible variable renaming detected")
                
                matches.append(CodeMatch(
                    repo1=profile1.name,
                    repo2=profile2.name,
                    file1=b1.file_path,
                    file2=b2.file_path,
                    block1_name=b1.name,
                    block2_name=b2.name,
                    similarity=max(sim, struct_sim),
                    match_type="near_exact",
                    code1_preview=b1.code[:500],
                    code2_preview=b2.code[:500],
                    evidence=evidence
                ))
            
            elif sim >= SUSPICIOUS_THRESHOLD:
                # Check if structural similarity is higher (variable renaming)
                struct_sim = structural_similarity(b1.code, b2.code)
                
                if struct_sim >= MIN_SIMILARITY_THRESHOLD:
                    matches.append(CodeMatch(
                        repo1=profile1.name,
                        repo2=profile2.name,
                        file1=b1.file_path,
                        file2=b2.file_path,
                        block1_name=b1.name,
                        block2_name=b2.name,
                        similarity=struct_sim,
                        match_type="structural",
                        code1_preview=b1.code[:500],
                        code2_preview=b2.code[:500],
                        evidence=[
                            f"Direct code similarity: {sim:.1%}",
                            f"Structural similarity: {struct_sim:.1%}",
                            "HIGH SUSPICION: Code structure identical but identifiers changed"
                        ]
                    ))
    
    # 3. Check for suspicious naming patterns
    common_functions = profile1.function_names & profile2.function_names
    # Exclude common names
    common_names = {'main', 'get', 'post', 'put', 'delete', 'create', 'read', 'update', 
                   'list', 'init', '__init__', 'setup', 'teardown', 'run', 'start', 'stop'}
    suspicious_functions = common_functions - common_names
    
    if len(suspicious_functions) >= 5:
        matches.append(CodeMatch(
            repo1=profile1.name,
            repo2=profile2.name,
            file1="<multiple>",
            file2="<multiple>",
            block1_name="<naming>",
            block2_name="<naming>",
            similarity=len(suspicious_functions) / max(len(profile1.function_names), len(profile2.function_names)),
            match_type="naming_pattern",
            code1_preview=", ".join(list(suspicious_functions)[:20]),
            code2_preview=", ".join(list(suspicious_functions)[:20]),
            evidence=[
                f"Shared {len(suspicious_functions)} non-standard function names",
                f"Functions: {', '.join(list(suspicious_functions)[:10])}"
            ]
        ))
    
    # 4. Check fingerprint overlap (code segment copying)
    fp_overlap = fingerprint_overlap(profile1.code_fingerprints, profile2.code_fingerprints)
    if fp_overlap > 0.3:  # More than 30% of code segments match
        matches.append(CodeMatch(
            repo1=profile1.name,
            repo2=profile2.name,
            file1="<overall>",
            file2="<overall>",
            block1_name="<codebase>",
            block2_name="<codebase>",
            similarity=fp_overlap,
            match_type="fingerprint_overlap",
            code1_preview=f"Total lines: {profile1.total_lines}",
            code2_preview=f"Total lines: {profile2.total_lines}",
            evidence=[
                f"N-gram fingerprint overlap: {fp_overlap:.1%}",
                f"Common code segments detected across codebase"
            ]
        ))
    
    return matches


def compute_plagiarism_score(matches: List[CodeMatch]) -> Tuple[float, str]:
    """Compute overall plagiarism score and verdict."""
    if not matches:
        return 0.0, "clean"
    
    # Weight different match types
    weights = {
        "exact": 1.0,
        "near_exact": 0.9,
        "structural": 0.85,
        "naming_pattern": 0.3,
        "fingerprint_overlap": 0.5,
    }
    
    score = 0.0
    for match in matches:
        weight = weights.get(match.match_type, 0.5)
        score += match.similarity * weight
    
    # Normalize (cap at 1.0)
    normalized_score = min(score / 3.0, 1.0)  # Expect up to 3 significant matches
    
    if normalized_score >= 0.7:
        verdict = "HIGH_PLAGIARISM"
    elif normalized_score >= 0.4:
        verdict = "SUSPICIOUS"
    elif normalized_score >= 0.2:
        verdict = "NEEDS_REVIEW"
    else:
        verdict = "likely_clean"
    
    return normalized_score, verdict


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def write_plagiarism_report(
    output_path: Path,
    all_matches: List[CodeMatch],
    pair_scores: Dict[Tuple[str, str], Tuple[float, str]],
) -> None:
    """Write detailed plagiarism report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Sort by severity
    severity_order = {"HIGH_PLAGIARISM": 0, "SUSPICIOUS": 1, "NEEDS_REVIEW": 2, "likely_clean": 3}
    sorted_pairs = sorted(
        pair_scores.items(),
        key=lambda x: (severity_order.get(x[1][1], 4), -x[1][0])
    )
    
    report = {
        "summary": {
            "total_pairs_analyzed": len(pair_scores),
            "high_plagiarism": sum(1 for _, (_, v) in pair_scores.items() if v == "HIGH_PLAGIARISM"),
            "suspicious": sum(1 for _, (_, v) in pair_scores.items() if v == "SUSPICIOUS"),
            "needs_review": sum(1 for _, (_, v) in pair_scores.items() if v == "NEEDS_REVIEW"),
        },
        "flagged_pairs": [],
        "detailed_matches": [],
    }
    
    for (repo1, repo2), (score, verdict) in sorted_pairs:
        if verdict in ("HIGH_PLAGIARISM", "SUSPICIOUS", "NEEDS_REVIEW"):
            pair_matches = [m for m in all_matches if 
                          (m.repo1 == repo1 and m.repo2 == repo2) or
                          (m.repo1 == repo2 and m.repo2 == repo1)]
            
            report["flagged_pairs"].append({
                "repo1": repo1,
                "repo2": repo2,
                "score": round(score, 3),
                "verdict": verdict,
                "num_matches": len(pair_matches),
            })
            
            for match in pair_matches[:5]:  # Top 5 matches per pair
                report["detailed_matches"].append({
                    "repo1": match.repo1,
                    "repo2": match.repo2,
                    "file1": match.file1,
                    "file2": match.file2,
                    "block1": match.block1_name,
                    "block2": match.block2_name,
                    "similarity": round(match.similarity, 3),
                    "type": match.match_type,
                    "evidence": match.evidence,
                    "code1_preview": match.code1_preview[:300],
                    "code2_preview": match.code2_preview[:300],
                })
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def write_plagiarism_csv(output_path: Path, pair_scores: Dict[Tuple[str, str], Tuple[float, str]]) -> None:
    """Write CSV summary."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    rows = []
    for (repo1, repo2), (score, verdict) in pair_scores.items():
        rows.append({
            "repo1": repo1,
            "repo2": repo2,
            "plagiarism_score": round(score, 4),
            "verdict": verdict,
        })
    
    rows.sort(key=lambda x: -x["plagiarism_score"])
    
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["repo1", "repo2", "plagiarism_score", "verdict"])
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Detect code plagiarism between repositories")
    parser.add_argument("--root", default="work", help="Directory containing submission folders")
    parser.add_argument("--repo-subdir", default="repo", help="Subdirectory holding the cloned repo")
    parser.add_argument("--output", default="results/plagiarism_report.json", help="JSON report output")
    parser.add_argument("--csv-output", default="results/plagiarism_scores.csv", help="CSV summary output")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress")

    args = parser.parse_args()
    root_dir = Path(args.root)
    
    if not root_dir.exists():
        raise SystemExit(f"Root directory not found: {root_dir}")

    # Find repositories
    submissions = sorted([p for p in root_dir.iterdir() if p.is_dir()])
    repo_paths: List[Tuple[Path, str]] = []
    for submission in submissions:
        repo_dir = submission / args.repo_subdir
        if repo_dir.is_dir():
            repo_paths.append((repo_dir, submission.name))
        elif submission.is_dir():
            repo_paths.append((submission, submission.name))

    if not repo_paths:
        raise SystemExit("No repositories found")

    print(f"Found {len(repo_paths)} repositories")
    
    # Analyze all repos
    print("Analyzing repositories for code patterns...")
    profiles: List[RepoCodeProfile] = []
    for repo_path, repo_name in tqdm(repo_paths, desc="Analyzing"):
        profile = analyze_repo_for_plagiarism(repo_path, repo_name)
        profiles.append(profile)
        if args.verbose:
            print(f"  {repo_name}: {len(profile.code_blocks)} code blocks, {profile.total_lines} lines")

    # Compare all pairs
    print("Comparing repositories for plagiarism...")
    all_matches: List[CodeMatch] = []
    pair_scores: Dict[Tuple[str, str], Tuple[float, str]] = {}
    
    total_pairs = len(profiles) * (len(profiles) - 1) // 2
    with tqdm(total=total_pairs, desc="Comparing") as pbar:
        for i, profile1 in enumerate(profiles):
            for profile2 in profiles[i+1:]:
                matches = find_code_matches(profile1, profile2)
                all_matches.extend(matches)
                
                score, verdict = compute_plagiarism_score(matches)
                pair_scores[(profile1.name, profile2.name)] = (score, verdict)
                
                if args.verbose and verdict != "likely_clean":
                    print(f"  {profile1.name} <-> {profile2.name}: {verdict} ({score:.2f})")
                
                pbar.update(1)

    # Write reports
    write_plagiarism_report(Path(args.output), all_matches, pair_scores)
    print(f"Wrote detailed report to {args.output}")
    
    write_plagiarism_csv(Path(args.csv_output), pair_scores)
    print(f"Wrote CSV summary to {args.csv_output}")
    
    # Print summary
    print("\n" + "="*70)
    print("PLAGIARISM DETECTION SUMMARY")
    print("="*70)
    
    high = [(k, v) for k, v in pair_scores.items() if v[1] == "HIGH_PLAGIARISM"]
    suspicious = [(k, v) for k, v in pair_scores.items() if v[1] == "SUSPICIOUS"]
    review = [(k, v) for k, v in pair_scores.items() if v[1] == "NEEDS_REVIEW"]
    
    print(f"\n🚨 HIGH PLAGIARISM: {len(high)} pairs")
    for (r1, r2), (score, _) in sorted(high, key=lambda x: -x[1][0]):
        print(f"   • {r1} <-> {r2} (score: {score:.2f})")
        # Show evidence
        pair_matches = [m for m in all_matches if 
                      (m.repo1 == r1 and m.repo2 == r2) or (m.repo1 == r2 and m.repo2 == r1)]
        for match in pair_matches[:2]:
            print(f"     - {match.match_type}: {match.block1_name} <-> {match.block2_name} ({match.similarity:.0%})")
    
    print(f"\n⚠️  SUSPICIOUS: {len(suspicious)} pairs")
    for (r1, r2), (score, _) in sorted(suspicious, key=lambda x: -x[1][0]):
        print(f"   • {r1} <-> {r2} (score: {score:.2f})")
    
    print(f"\n📋 NEEDS REVIEW: {len(review)} pairs")
    for (r1, r2), (score, _) in sorted(review, key=lambda x: -x[1][0])[:10]:
        print(f"   • {r1} <-> {r2} (score: {score:.2f})")
    
    if not high and not suspicious:
        print("\n✅ No significant plagiarism detected!")


if __name__ == "__main__":
    main()
