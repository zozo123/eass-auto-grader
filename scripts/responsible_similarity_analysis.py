#!/usr/bin/env python3
"""
Responsible Similarity Analysis for Student Projects

IMPORTANT: This tool identifies similarities that WARRANT HUMAN REVIEW.
It does NOT make accusations of plagiarism.

Similarity in student code can arise from many legitimate sources:
1. Following the same tutorial (e.g., official FastAPI/SQLModel docs)
2. Assignment requirements forcing similar structure
3. Framework conventions (FastAPI patterns are highly standardized)
4. Using same StackOverflow solutions for common problems
5. Course-provided templates or starter code
6. Small codebase size (less room for variation)

This tool tries to distinguish between:
- EXPECTED: Similarities that are normal for the assignment
- INVESTIGATE: Unusual similarities that warrant human review
- HIGHLY UNUSUAL: Very specific matches that are worth discussing

NEVER use this tool's output alone to accuse students of cheating.
"""

import ast
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import argparse


# Common patterns that are EXPECTED to be similar in FastAPI CRUD projects
EXPECTED_PATTERNS = {
    # Standard SQLModel/SQLAlchemy CRUD (from official tutorial)
    "session.add(",
    "session.commit()",
    "session.refresh(",
    "session.delete(",
    "session.get(",
    "session.exec(",
    "select(",
    
    # Standard FastAPI patterns
    "@app.get(",
    "@app.post(",
    "@app.put(",
    "@app.delete(",
    "@app.patch(",
    "HTTPException(",
    "status_code=404",
    "status_code=201",
    "response_model=",
    "Depends(",
    
    # Standard Pydantic patterns
    "Field(",
    "model_dump(",
    "model_validate(",
    "BaseModel",
    "SQLModel",
    
    # Standard imports
    "from fastapi import",
    "from pydantic import",
    "from sqlmodel import",
    "from typing import",
}

# Standard CRUD function signatures - variations are expected
EXPECTED_CRUD_SIGNATURES = {
    r"def (create|add)_\w+\(",
    r"def (get|read|retrieve)_\w+\(",
    r"def (update|modify|edit)_\w+\(",
    r"def (delete|remove)_\w+\(",
    r"def list_\w+\(",
}


@dataclass
class SimilarityEvidence:
    """Evidence of similarity with context."""
    type: str  # expected, investigate, highly_unusual
    description: str
    repo1_context: str
    repo2_context: str
    confidence: str  # low, medium, high that this indicates actual copying
    legitimate_reasons: List[str]


@dataclass
class ResponsibleAnalysis:
    """Analysis result with context and nuance."""
    repo1: str
    repo2: str
    expected_similarities: List[SimilarityEvidence] = field(default_factory=list)
    investigate_similarities: List[SimilarityEvidence] = field(default_factory=list)
    highly_unusual_similarities: List[SimilarityEvidence] = field(default_factory=list)
    
    # Counter-evidence (reasons why they might NOT be copying)
    different_aspects: List[str] = field(default_factory=list)
    
    # Final assessment
    recommendation: str = ""
    requires_human_review: bool = False


def get_python_files(repo_path: Path) -> List[Path]:
    """Get all Python files excluding tests and common generated files."""
    files = []
    for py_file in repo_path.rglob("*.py"):
        # Skip obvious non-source files
        if any(part.startswith('.') for part in py_file.parts):
            continue
        if '__pycache__' in str(py_file):
            continue
        files.append(py_file)
    return files


def extract_code_features(repo_path: Path) -> Dict:
    """Extract code features for comparison."""
    features = {
        "comments": [],
        "docstrings": [],
        "unique_strings": [],  # String literals that aren't common
        "variable_names": defaultdict(int),
        "function_names": [],
        "class_names": [],
        "error_messages": [],
        "magic_numbers": [],
        "custom_patterns": [],  # Non-standard code patterns
        "file_structure": [],
        "imports": [],
    }
    
    files = get_python_files(repo_path)
    
    for py_file in files:
        try:
            content = py_file.read_text()
            rel_path = py_file.relative_to(repo_path)
            features["file_structure"].append(str(rel_path))
            
            # Extract comments (not docstrings)
            for match in re.finditer(r'#\s*(.+)$', content, re.MULTILINE):
                comment = match.group(1).strip()
                if len(comment) > 10:  # Skip short comments
                    features["comments"].append((str(rel_path), comment))
            
            # Parse AST for more features
            try:
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    # Docstrings
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                        docstring = ast.get_docstring(node)
                        if docstring and len(docstring) > 20:
                            features["docstrings"].append((str(rel_path), docstring))
                    
                    # Function names
                    if isinstance(node, ast.FunctionDef):
                        features["function_names"].append(node.name)
                    
                    # Class names
                    if isinstance(node, ast.ClassDef):
                        features["class_names"].append(node.name)
                    
                    # String literals (potential error messages, etc.)
                    if isinstance(node, ast.Constant) and isinstance(node.value, str):
                        s = node.value
                        if len(s) > 15 and not s.startswith(('/', 'http', 'sqlite')):
                            features["unique_strings"].append(s)
                    
                    # Import statements
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            features["imports"].append(alias.name)
                    if isinstance(node, ast.ImportFrom):
                        if node.module:
                            features["imports"].append(node.module)
                    
                    # Magic numbers
                    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                        if node.value not in (0, 1, 2, -1, 100, 200, 201, 204, 400, 401, 403, 404, 500):
                            features["magic_numbers"].append(node.value)
                            
            except SyntaxError:
                pass
                
        except Exception:
            pass
    
    return features


def is_expected_similarity(text: str) -> Tuple[bool, str]:
    """Check if a similarity is expected due to framework/tutorial patterns."""
    for pattern in EXPECTED_PATTERNS:
        if pattern.lower() in text.lower():
            return True, f"Standard pattern: {pattern}"
    
    for regex in EXPECTED_CRUD_SIGNATURES:
        if re.search(regex, text, re.IGNORECASE):
            return True, "Standard CRUD function signature"
    
    return False, ""


def analyze_pair_responsibly(repo1_path: Path, repo2_path: Path, 
                             repo1_name: str, repo2_name: str) -> ResponsibleAnalysis:
    """Analyze a pair of repos with nuance and responsibility."""
    
    analysis = ResponsibleAnalysis(repo1=repo1_name, repo2=repo2_name)
    
    features1 = extract_code_features(repo1_path)
    features2 = extract_code_features(repo2_path)
    
    # 1. Check comments for identical matches
    comments1 = set(c for _, c in features1["comments"])
    comments2 = set(c for _, c in features2["comments"])
    shared_comments = comments1 & comments2
    
    for comment in shared_comments:
        is_expected, reason = is_expected_similarity(comment)
        if is_expected:
            analysis.expected_similarities.append(SimilarityEvidence(
                type="expected",
                description=f"Same comment: '{comment[:50]}...'",
                repo1_context="comment",
                repo2_context="comment",
                confidence="low",
                legitimate_reasons=[reason, "Common explanatory comment"]
            ))
        elif len(comment) > 30:  # Longer comments are more significant
            analysis.investigate_similarities.append(SimilarityEvidence(
                type="investigate",
                description=f"Identical comment: '{comment}'",
                repo1_context="comment",
                repo2_context="comment",
                confidence="medium",
                legitimate_reasons=["Could be from same tutorial", "Common phrasing"]
            ))
    
    # 2. Check docstrings
    docstrings1 = set(d for _, d in features1["docstrings"])
    docstrings2 = set(d for _, d in features2["docstrings"])
    shared_docstrings = docstrings1 & docstrings2
    
    for docstring in shared_docstrings:
        if len(docstring) > 50:
            analysis.investigate_similarities.append(SimilarityEvidence(
                type="investigate",
                description=f"Identical docstring: '{docstring[:80]}...'",
                repo1_context="docstring",
                repo2_context="docstring",
                confidence="medium",
                legitimate_reasons=["Common documentation pattern", "Course template"]
            ))
    
    # 3. Check error messages (these are often unique choices)
    strings1 = set(features1["unique_strings"])
    strings2 = set(features2["unique_strings"])
    shared_strings = strings1 & strings2
    
    for s in shared_strings:
        is_expected, reason = is_expected_similarity(s)
        if not is_expected and "not found" not in s.lower():
            if len(s) > 30:
                analysis.highly_unusual_similarities.append(SimilarityEvidence(
                    type="highly_unusual",
                    description=f"Identical unique string: '{s}'",
                    repo1_context="string literal",
                    repo2_context="string literal",
                    confidence="high",
                    legitimate_reasons=["Copied from same source", "Could indicate collaboration"]
                ))
    
    # 4. Look for differences (counter-evidence)
    # Different function names
    funcs1 = set(features1["function_names"])
    funcs2 = set(features2["function_names"])
    different_funcs = (funcs1 - funcs2) | (funcs2 - funcs1)
    if different_funcs:
        analysis.different_aspects.append(
            f"Different function names: {', '.join(list(different_funcs)[:5])}"
        )
    
    # Different file structures
    files1 = set(features1["file_structure"])
    files2 = set(features2["file_structure"])
    different_files = (files1 - files2) | (files2 - files1)
    if different_files:
        analysis.different_aspects.append(
            f"Different file organization: {', '.join(list(different_files)[:5])}"
        )
    
    # Different imports (beyond standard)
    std_imports = {'fastapi', 'pydantic', 'sqlmodel', 'typing', 'uvicorn', 'pytest'}
    imports1 = set(features1["imports"]) - std_imports
    imports2 = set(features2["imports"]) - std_imports
    different_imports = (imports1 - imports2) | (imports2 - imports1)
    if different_imports:
        analysis.different_aspects.append(
            f"Different non-standard imports: {', '.join(list(different_imports)[:5])}"
        )
    
    # 5. Generate recommendation
    num_unusual = len(analysis.highly_unusual_similarities)
    num_investigate = len(analysis.investigate_similarities)
    num_different = len(analysis.different_aspects)
    
    if num_unusual >= 2:
        analysis.recommendation = (
            "HUMAN REVIEW RECOMMENDED: Multiple highly unusual similarities found. "
            "This could indicate collaboration, shared source, or copying. "
            "However, there could be legitimate explanations - please investigate manually."
        )
        analysis.requires_human_review = True
    elif num_unusual == 1 and num_investigate >= 3:
        analysis.recommendation = (
            "WORTH REVIEWING: Some unusual similarities combined with multiple investigate-level matches. "
            "Could be coincidence or could warrant a conversation with students."
        )
        analysis.requires_human_review = True
    elif num_investigate >= 5 and num_different < 3:
        analysis.recommendation = (
            "MINOR CONCERN: Several investigate-level similarities but could easily be explained by "
            "following same tutorials or using similar resources."
        )
        analysis.requires_human_review = False
    else:
        analysis.recommendation = (
            "LIKELY NORMAL: Similarities appear to be expected for this type of assignment. "
            "Both students likely followed similar tutorials/resources independently."
        )
        analysis.requires_human_review = False
    
    return analysis


def main():
    parser = argparse.ArgumentParser(description="Responsible similarity analysis")
    parser.add_argument("--root", default="work", help="Root directory")
    parser.add_argument("--repo-subdir", default="repo", help="Subdirectory containing repo")
    parser.add_argument("--pairs", help="Specific pairs to analyze (comma-separated)")
    parser.add_argument("--output", default="results/responsible_analysis.json")
    args = parser.parse_args()
    
    root = Path(args.root)
    repos = sorted([d for d in root.iterdir() if d.is_dir()])
    
    print(f"Found {len(repos)} repositories")
    print()
    print("=" * 70)
    print("RESPONSIBLE SIMILARITY ANALYSIS")
    print("=" * 70)
    print()
    print("⚠️  IMPORTANT DISCLAIMER:")
    print("This tool identifies similarities that may warrant human review.")
    print("It does NOT make accusations of academic dishonesty.")
    print()
    print("High similarity can result from:")
    print("  • Following the same tutorial (FastAPI docs are highly standardized)")
    print("  • Course requirements forcing similar structure")
    print("  • Using the same framework patterns (expected)")
    print("  • Common StackOverflow solutions")
    print("  • Course-provided templates")
    print("  • Small codebases having less room for variation")
    print()
    print("=" * 70)
    
    # Analyze previously flagged pairs
    flagged_pairs = [
        ("ben_lavi_movie-service-fastapi", "eli_offri_main"),
        ("haim_lev_tov_BookWorm", "tal_alayoff_books-catalogue-api"),
        ("david_yakhin_ex-1", "guy_cohen_CinemaPlanet"),
    ]
    
    results = []
    
    for repo1_name, repo2_name in flagged_pairs:
        repo1_dir = root / repo1_name / args.repo_subdir
        repo2_dir = root / repo2_name / args.repo_subdir
        
        if not repo1_dir.exists() or not repo2_dir.exists():
            continue
        
        print(f"\n📊 Analyzing: {repo1_name} ↔ {repo2_name}")
        print("-" * 60)
        
        analysis = analyze_pair_responsibly(repo1_dir, repo2_dir, repo1_name, repo2_name)
        results.append(analysis)
        
        print(f"\n  Expected similarities: {len(analysis.expected_similarities)}")
        print(f"  Investigate-level:     {len(analysis.investigate_similarities)}")
        print(f"  Highly unusual:        {len(analysis.highly_unusual_similarities)}")
        print(f"  Different aspects:     {len(analysis.different_aspects)}")
        
        if analysis.investigate_similarities:
            print("\n  🔍 INVESTIGATE (may have legitimate explanations):")
            for ev in analysis.investigate_similarities[:3]:
                print(f"    • {ev.description[:70]}")
                print(f"      Possible reasons: {', '.join(ev.legitimate_reasons[:2])}")
        
        if analysis.highly_unusual_similarities:
            print("\n  ⚠️  HIGHLY UNUSUAL (worth human review):")
            for ev in analysis.highly_unusual_similarities[:3]:
                print(f"    • {ev.description[:70]}")
        
        if analysis.different_aspects:
            print("\n  ✅ DIFFERENT (counter-evidence):")
            for diff in analysis.different_aspects[:3]:
                print(f"    • {diff}")
        
        print(f"\n  📋 RECOMMENDATION: {analysis.recommendation[:100]}...")
        print(f"  🔎 Requires human review: {'YES' if analysis.requires_human_review else 'NO'}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to serializable format
    results_json = []
    for r in results:
        results_json.append({
            "repo1": r.repo1,
            "repo2": r.repo2,
            "expected_count": len(r.expected_similarities),
            "investigate_count": len(r.investigate_similarities),
            "highly_unusual_count": len(r.highly_unusual_similarities),
            "different_aspects_count": len(r.different_aspects),
            "different_aspects": r.different_aspects,
            "recommendation": r.recommendation,
            "requires_human_review": r.requires_human_review,
            "investigate_items": [
                {"description": e.description, "reasons": e.legitimate_reasons}
                for e in r.investigate_similarities
            ],
            "highly_unusual_items": [
                {"description": e.description}
                for e in r.highly_unusual_similarities
            ],
        })
    
    with open(output_path, "w") as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n\n✅ Detailed results saved to {output_path}")
    print()
    print("=" * 70)
    print("FINAL NOTE")
    print("=" * 70)
    print("""
Before taking any action:
1. Review the code manually - automated tools make mistakes
2. Consider the assignment requirements and what patterns are expected
3. Check if students used the same tutorial/template (legitimate)
4. Talk to students privately before making any accusations
5. Remember: similarity is not proof of copying
""")


if __name__ == "__main__":
    main()
