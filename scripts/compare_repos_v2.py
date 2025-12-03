#!/usr/bin/env python3
"""Compare student repositories using local embeddings + Qdrant (v2).

Improvements over v1:
- AST-based feature extraction for Python files
- Separate embeddings for structure, code patterns, and dependencies
- Better similarity metrics (Jaccard, structural overlap)
- Clustering and visualization support
- Framework/pattern detection (FastAPI, Pydantic, etc.)
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
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
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache", "dist", "build", ".tox", "egg-info"}

FRAMEWORK_INDICATORS = {
    "fastapi": ["FastAPI", "APIRouter", "@app.get", "@app.post", "@router.get", "@router.post"],
    "flask": ["Flask", "@app.route", "flask"],
    "django": ["django", "models.Model", "views.py"],
    "pydantic": ["BaseModel", "Field", "validator", "model_validator"],
    "sqlalchemy": ["SQLAlchemy", "Column", "relationship", "declarative_base"],
    "pytest": ["pytest", "@pytest.fixture", "test_"],
    "docker": ["Dockerfile", "docker-compose"],
}

CRUD_PATTERNS = {
    "create": ["post", "create", "add", "insert"],
    "read": ["get", "read", "fetch", "list", "retrieve"],
    "update": ["put", "patch", "update", "modify"],
    "delete": ["delete", "remove", "destroy"],
}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class CodeFeatures:
    """AST-extracted features from Python code."""
    classes: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    base_classes: List[str] = field(default_factory=list)
    docstrings: List[str] = field(default_factory=list)
    type_hints: List[str] = field(default_factory=list)
    
    def merge(self, other: "CodeFeatures") -> "CodeFeatures":
        return CodeFeatures(
            classes=self.classes + other.classes,
            functions=self.functions + other.functions,
            decorators=self.decorators + other.decorators,
            imports=self.imports + other.imports,
            base_classes=self.base_classes + other.base_classes,
            docstrings=self.docstrings + other.docstrings,
            type_hints=self.type_hints + other.type_hints,
        )


@dataclass
class RepoAnalysis:
    """Complete analysis of a repository."""
    name: str
    path: Path
    files: List[str]
    file_count: int
    python_files: int
    test_files: int
    has_docker: bool
    has_readme: bool
    has_tests: bool
    lang_counts: Dict[str, int]
    dir_structure: Dict[str, int]
    frameworks: List[str]
    crud_operations: Dict[str, bool]
    dependencies: List[str]
    code_features: CodeFeatures
    readme_excerpt: str
    api_endpoints: List[str]
    pydantic_models: List[str]
    
    def to_payload(self) -> Dict[str, Any]:
        return {
            "repo": self.name,
            "path": str(self.path),
            "file_count": self.file_count,
            "python_files": self.python_files,
            "test_files": self.test_files,
            "has_docker": self.has_docker,
            "has_readme": self.has_readme,
            "has_tests": self.has_tests,
            "top_languages": dict(Counter(self.lang_counts).most_common(5)),
            "top_dirs": dict(Counter(self.dir_structure).most_common(5)),
            "frameworks": self.frameworks,
            "crud_operations": self.crud_operations,
            "dependencies": self.dependencies[:30],
            "classes": self.code_features.classes[:30],
            "functions": self.code_features.functions[:50],
            "imports": list(set(self.code_features.imports))[:30],
            "readme_excerpt": self.readme_excerpt[:500],
            "api_endpoints": self.api_endpoints[:20],
            "pydantic_models": self.pydantic_models[:20],
        }


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------
def read_text(path: Path, limit: int = 4000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:limit]
    except OSError:
        return ""


def is_python_file(path: Path) -> bool:
    return path.suffix.lower() == ".py"


def is_test_file(path: Path) -> bool:
    name = path.name.lower()
    return name.startswith("test_") or name.endswith("_test.py") or "tests" in str(path).lower()


# ---------------------------------------------------------------------------
# AST Analysis
# ---------------------------------------------------------------------------
def extract_code_features(source: str) -> CodeFeatures:
    """Extract structural features from Python source using AST."""
    features = CodeFeatures()
    
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return features
    
    for node in ast.walk(tree):
        # Classes
        if isinstance(node, ast.ClassDef):
            features.classes.append(node.name)
            for base in node.bases:
                if isinstance(base, ast.Name):
                    features.base_classes.append(base.id)
                elif isinstance(base, ast.Attribute):
                    features.base_classes.append(base.attr)
            
            # Check for docstring
            if (node.body and isinstance(node.body[0], ast.Expr) and 
                isinstance(node.body[0].value, ast.Constant) and 
                isinstance(node.body[0].value.value, str)):
                features.docstrings.append(node.body[0].value.value[:200])
        
        # Functions
        elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            features.functions.append(node.name)
            
            # Decorators
            for dec in node.decorator_list:
                if isinstance(dec, ast.Name):
                    features.decorators.append(dec.id)
                elif isinstance(dec, ast.Attribute):
                    features.decorators.append(f"{dec.attr}")
                elif isinstance(dec, ast.Call):
                    if isinstance(dec.func, ast.Attribute):
                        features.decorators.append(f"{dec.func.attr}")
                    elif isinstance(dec.func, ast.Name):
                        features.decorators.append(dec.func.id)
            
            # Return type hints
            if node.returns:
                if isinstance(node.returns, ast.Name):
                    features.type_hints.append(node.returns.id)
                elif isinstance(node.returns, ast.Subscript):
                    if isinstance(node.returns.value, ast.Name):
                        features.type_hints.append(node.returns.value.id)
        
        # Imports
        elif isinstance(node, ast.Import):
            for alias in node.names:
                features.imports.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                features.imports.append(node.module.split(".")[0])
    
    return features


def detect_api_endpoints(source: str) -> List[str]:
    """Detect FastAPI/Flask route definitions."""
    endpoints = []
    patterns = [
        r'@(?:app|router)\.(get|post|put|patch|delete)\s*\(\s*["\']([^"\']+)["\']',
        r'@(?:app|router)\.route\s*\(\s*["\']([^"\']+)["\']',
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, source, re.IGNORECASE):
            if len(match.groups()) == 2:
                endpoints.append(f"{match.group(1).upper()} {match.group(2)}")
            else:
                endpoints.append(f"ROUTE {match.group(1)}")
    return endpoints


def detect_pydantic_models(source: str, features: CodeFeatures) -> List[str]:
    """Detect Pydantic model definitions."""
    models = []
    pydantic_bases = {"BaseModel", "BaseSettings", "BaseConfig"}
    for i, cls in enumerate(features.classes):
        if i < len(features.base_classes):
            if features.base_classes[i] in pydantic_bases or "Base" in features.base_classes[i]:
                models.append(cls)
    return models


# ---------------------------------------------------------------------------
# Repository Analysis
# ---------------------------------------------------------------------------
def collect_files(repo_path: Path) -> Tuple[List[str], Dict[str, int], Dict[str, int]]:
    """Return relative file paths, language counter, and directory structure."""
    files: List[str] = []
    lang_counts: Dict[str, int] = defaultdict(int)
    dir_counts: Dict[str, int] = defaultdict(int)

    for root, dirs, file_names in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        rel_dir = Path(root).relative_to(repo_path)
        if rel_dir != Path("."):
            dir_counts[str(rel_dir.parts[0])] += 1
        for file_name in file_names:
            if file_name.startswith("."):
                continue
            full_path = Path(root) / file_name
            rel_path = full_path.relative_to(repo_path)
            files.append(str(rel_path))
            ext = full_path.suffix.lower() or "noext"
            lang_counts[ext] += 1

    return sorted(files), dict(lang_counts), dict(dir_counts)


def detect_frameworks(repo_path: Path, files: List[str], all_code: str) -> List[str]:
    """Detect which frameworks/tools are used."""
    detected = []
    file_content = all_code.lower()
    file_set = set(f.lower() for f in files)
    
    for framework, indicators in FRAMEWORK_INDICATORS.items():
        for indicator in indicators:
            if indicator.lower() in file_content or indicator.lower() in file_set:
                if framework not in detected:
                    detected.append(framework)
                break
    
    # Check for Docker files explicitly
    if (repo_path / "Dockerfile").exists() or (repo_path / "docker-compose.yml").exists():
        if "docker" not in detected:
            detected.append("docker")
    
    return detected


def detect_crud_operations(endpoints: List[str], functions: List[str]) -> Dict[str, bool]:
    """Detect which CRUD operations are implemented."""
    all_names = " ".join(endpoints + functions).lower()
    return {
        op: any(pattern in all_names for pattern in patterns)
        for op, patterns in CRUD_PATTERNS.items()
    }


def parse_dependencies(text: str) -> List[str]:
    """Extract dependency names from requirements/pyproject."""
    names = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" in line or ">=" in line or "~=" in line or "<=" in line:
            names.append(re.split(r"[=<>~!]", line, maxsplit=1)[0].strip())
        elif "[" in line and "]" in line:
            names.append(line.split("[", 1)[0].strip())
        elif " = " in line and '"' in line:
            names.append(line.split("=", 1)[0].strip().strip('"').strip("'"))
        elif re.match(r"^[A-Za-z0-9_.-]+$", line):
            names.append(line)
    
    # De-duplicate preserving order
    seen = set()
    return [n for n in names if n and not (n in seen or seen.add(n))]


def analyze_repo(repo_path: Path, repo_name: str, max_code_files: int = 30) -> RepoAnalysis:
    """Perform complete analysis of a repository."""
    files, lang_counts, dir_counts = collect_files(repo_path)
    
    # Basic stats
    python_files = [f for f in files if f.endswith(".py")]
    test_files = [f for f in files if is_test_file(Path(f))]
    has_docker = (repo_path / "Dockerfile").exists() or (repo_path / "docker-compose.yml").exists()
    has_readme = (repo_path / "README.md").exists()
    has_tests = len(test_files) > 0
    
    # Read README
    readme_excerpt = read_text(repo_path / "README.md", limit=2000)
    
    # Parse dependencies
    deps = []
    for dep_file in ["requirements.txt", "pyproject.toml", "package.json"]:
        if (repo_path / dep_file).exists():
            deps = parse_dependencies(read_text(repo_path / dep_file, limit=2000))
            break
    
    # Analyze Python code
    all_features = CodeFeatures()
    all_endpoints: List[str] = []
    all_code_text = ""
    
    # Prioritize non-test files
    sorted_py_files = sorted(
        python_files,
        key=lambda f: (is_test_file(Path(f)), f)
    )[:max_code_files]
    
    for rel_path in sorted_py_files:
        full_path = repo_path / rel_path
        try:
            source = full_path.read_text(encoding="utf-8", errors="ignore")
            all_code_text += source + "\n"
            features = extract_code_features(source)
            all_features = all_features.merge(features)
            all_endpoints.extend(detect_api_endpoints(source))
        except OSError:
            continue
    
    # Detect frameworks and patterns
    frameworks = detect_frameworks(repo_path, files, all_code_text)
    pydantic_models = detect_pydantic_models(all_code_text, all_features)
    crud_ops = detect_crud_operations(all_endpoints, all_features.functions)
    
    return RepoAnalysis(
        name=repo_name,
        path=repo_path,
        files=files,
        file_count=len(files),
        python_files=len(python_files),
        test_files=len(test_files),
        has_docker=has_docker,
        has_readme=has_readme,
        has_tests=has_tests,
        lang_counts=lang_counts,
        dir_structure=dir_counts,
        frameworks=frameworks,
        crud_operations=crud_ops,
        dependencies=deps,
        code_features=all_features,
        readme_excerpt=readme_excerpt,
        api_endpoints=all_endpoints,
        pydantic_models=pydantic_models,
    )


# ---------------------------------------------------------------------------
# Embedding & Similarity
# ---------------------------------------------------------------------------
def build_embedding_text(analysis: RepoAnalysis) -> str:
    """Build rich text representation for embedding."""
    parts = [
        f"Project: {analysis.name}",
        f"Frameworks: {', '.join(analysis.frameworks)}",
        f"CRUD Operations: {', '.join(k for k, v in analysis.crud_operations.items() if v)}",
        f"Dependencies: {', '.join(analysis.dependencies[:20])}",
        f"Classes: {', '.join(analysis.code_features.classes[:20])}",
        f"Pydantic Models: {', '.join(analysis.pydantic_models[:10])}",
        f"Functions: {', '.join(analysis.code_features.functions[:30])}",
        f"Decorators: {', '.join(set(analysis.code_features.decorators))}",
        f"API Endpoints: {', '.join(analysis.api_endpoints[:15])}",
        f"Base Classes: {', '.join(set(analysis.code_features.base_classes))}",
        f"Imports: {', '.join(set(analysis.code_features.imports))}",
    ]
    
    if analysis.readme_excerpt:
        parts.append(f"Description: {analysis.readme_excerpt[:800]}")
    
    # File structure summary
    parts.append(f"File structure: {', '.join(analysis.files[:50])}")
    
    return "\n".join(parts)


def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Calculate Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def compute_detailed_similarity(a1: RepoAnalysis, a2: RepoAnalysis) -> Dict[str, float]:
    """Compute multiple similarity metrics between two repos."""
    return {
        "dep_similarity": jaccard_similarity(set(a1.dependencies), set(a2.dependencies)),
        "framework_similarity": jaccard_similarity(set(a1.frameworks), set(a2.frameworks)),
        "class_similarity": jaccard_similarity(set(a1.code_features.classes), set(a2.code_features.classes)),
        "function_similarity": jaccard_similarity(
            set(a1.code_features.functions[:50]), 
            set(a2.code_features.functions[:50])
        ),
        "import_similarity": jaccard_similarity(
            set(a1.code_features.imports), 
            set(a2.code_features.imports)
        ),
        "file_overlap": jaccard_similarity(set(a1.files[:100]), set(a2.files[:100])),
        "crud_match": sum(
            1 for k in CRUD_PATTERNS 
            if a1.crud_operations.get(k) == a2.crud_operations.get(k)
        ) / len(CRUD_PATTERNS),
    }


# ---------------------------------------------------------------------------
# Qdrant Operations
# ---------------------------------------------------------------------------
def ensure_collection(client: QdrantClient, collection: str, vector_size: int, reset: bool) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if reset and collection in existing:
        client.delete_collection(collection)
        existing.remove(collection)

    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
        )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def write_results_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "repo", "match", "vector_score", "dep_similarity", "framework_similarity",
        "class_similarity", "function_similarity", "import_similarity", "file_overlap",
        "crud_match", "combined_score", "shared_frameworks", "shared_deps",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_similarity_matrix(path: Path, analyses: List[RepoAnalysis], scores: Dict[Tuple[str, str], float]) -> None:
    """Write a similarity matrix as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [a.name for a in analyses]
    
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + names)
        for a1 in analyses:
            row = [a1.name]
            for a2 in analyses:
                if a1.name == a2.name:
                    row.append("1.0")
                else:
                    key = tuple(sorted([a1.name, a2.name]))
                    row.append(f"{scores.get(key, 0):.3f}")
            writer.writerow(row)


def write_cluster_report(path: Path, analyses: List[RepoAnalysis]) -> None:
    """Write a summary report grouping repos by frameworks/patterns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    by_framework: Dict[str, List[str]] = defaultdict(list)
    by_crud: Dict[str, List[str]] = defaultdict(list)
    
    for a in analyses:
        for fw in a.frameworks:
            by_framework[fw].append(a.name)
        crud_profile = "-".join(k for k, v in sorted(a.crud_operations.items()) if v)
        by_crud[crud_profile or "none"].append(a.name)
    
    report = {
        "by_framework": dict(by_framework),
        "by_crud_profile": dict(by_crud),
        "summary": {
            "total_repos": len(analyses),
            "with_docker": sum(1 for a in analyses if a.has_docker),
            "with_tests": sum(1 for a in analyses if a.has_tests),
            "with_readme": sum(1 for a in analyses if a.has_readme),
            "framework_usage": {fw: len(repos) for fw, repos in by_framework.items()},
        }
    }
    
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare repositories by embedding structure/metadata into Qdrant (v2)."
    )
    parser.add_argument("--root", default="work", help="Directory containing submission folders")
    parser.add_argument("--repo-subdir", default="repo", help="Subdirectory holding the cloned repo")
    parser.add_argument("--collection", default="repo_similarity_v2", help="Qdrant collection name")
    parser.add_argument("--qdrant-url", default=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--model", default=os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    parser.add_argument("--top-k", type=int, default=5, help="Top similar repos per repo")
    parser.add_argument("--reset", action="store_true", help="Drop collection before writing")
    parser.add_argument("--output", default="results/repo_similarity_v2.csv")
    parser.add_argument("--matrix-output", default="results/similarity_matrix.csv")
    parser.add_argument("--cluster-output", default="results/cluster_report.json")
    parser.add_argument("--max-code-files", type=int, default=30, help="Max Python files to analyze per repo")

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
        raise SystemExit("No repositories found under the given root.")

    print(f"Found {len(repo_paths)} repositories")

    # Analyze all repos
    print("Analyzing repositories...")
    analyses: List[RepoAnalysis] = []
    for repo_path, repo_name in tqdm(repo_paths, desc="Analyzing"):
        analysis = analyze_repo(repo_path, repo_name, args.max_code_files)
        analyses.append(analysis)

    # Initialize embedder and Qdrant
    embedder = TextEmbedding(model_name=args.model)
    client = QdrantClient(url=args.qdrant_url)

    size = getattr(embedder, "embedding_size", None)
    if not size:
        size = len(next(embedder.embed(["probe"])))

    try:
        ensure_collection(client, args.collection, int(size), args.reset)
    except Exception as exc:
        raise SystemExit(
            f"Could not reach Qdrant at {args.qdrant_url}. "
            "Start the container with `make qdrant-up` first."
        ) from exc

    # Build embeddings and index
    print("Embedding and indexing...")
    points: List[qmodels.PointStruct] = []
    analysis_map: Dict[str, RepoAnalysis] = {}
    
    for analysis in tqdm(analyses, desc="Embedding"):
        embedding_text = build_embedding_text(analysis)
        vector = next(embedder.embed([embedding_text]))
        point_id = str(uuid4())
        
        analysis_map[point_id] = analysis
        
        points.append(qmodels.PointStruct(
            id=point_id,
            vector=vector.tolist(),
            payload=analysis.to_payload(),
        ))

    client.upsert(collection_name=args.collection, points=points, wait=True)

    # Query similarities
    print("Computing similarities...")
    rows = []
    matrix_scores: Dict[Tuple[str, str], float] = {}
    
    for point in tqdm(points, desc="Searching"):
        source_analysis = analysis_map[point.id]
        
        hits = client.query_points(
            collection_name=args.collection,
            query=point.vector,
            limit=args.top_k + 1,
            with_payload=True,
        ).points

        for hit in hits:
            if str(hit.id) == str(point.id):
                continue
            
            target_analysis = analysis_map.get(str(hit.id))
            if not target_analysis:
                continue
            
            # Compute detailed metrics
            detailed = compute_detailed_similarity(source_analysis, target_analysis)
            
            # Combined score (weighted average)
            combined = (
                hit.score * 0.3 +  # Vector similarity
                detailed["dep_similarity"] * 0.15 +
                detailed["framework_similarity"] * 0.15 +
                detailed["class_similarity"] * 0.1 +
                detailed["function_similarity"] * 0.1 +
                detailed["import_similarity"] * 0.1 +
                detailed["crud_match"] * 0.1
            )
            
            # Store for matrix
            key = tuple(sorted([source_analysis.name, target_analysis.name]))
            if key not in matrix_scores:
                matrix_scores[key] = combined
            
            # Shared items
            shared_frameworks = list(set(source_analysis.frameworks) & set(target_analysis.frameworks))
            shared_deps = list(set(source_analysis.dependencies[:20]) & set(target_analysis.dependencies[:20]))
            
            rows.append({
                "repo": source_analysis.name,
                "match": target_analysis.name,
                "vector_score": round(hit.score, 4),
                "dep_similarity": round(detailed["dep_similarity"], 4),
                "framework_similarity": round(detailed["framework_similarity"], 4),
                "class_similarity": round(detailed["class_similarity"], 4),
                "function_similarity": round(detailed["function_similarity"], 4),
                "import_similarity": round(detailed["import_similarity"], 4),
                "file_overlap": round(detailed["file_overlap"], 4),
                "crud_match": round(detailed["crud_match"], 4),
                "combined_score": round(combined, 4),
                "shared_frameworks": ", ".join(shared_frameworks),
                "shared_deps": ", ".join(shared_deps[:10]),
            })

    # Write outputs
    write_results_csv(Path(args.output), rows)
    print(f"Wrote similarity report to {args.output}")
    
    write_similarity_matrix(Path(args.matrix_output), analyses, matrix_scores)
    print(f"Wrote similarity matrix to {args.matrix_output}")
    
    write_cluster_report(Path(args.cluster_output), analyses)
    print(f"Wrote cluster report to {args.cluster_output}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total repositories analyzed: {len(analyses)}")
    print(f"With Docker: {sum(1 for a in analyses if a.has_docker)}")
    print(f"With Tests: {sum(1 for a in analyses if a.has_tests)}")
    print(f"With README: {sum(1 for a in analyses if a.has_readme)}")
    
    # Top similar pairs
    sorted_pairs = sorted(matrix_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 Most Similar Pairs:")
    for (n1, n2), score in sorted_pairs:
        print(f"  {score:.3f}: {n1} <-> {n2}")


if __name__ == "__main__":
    main()
