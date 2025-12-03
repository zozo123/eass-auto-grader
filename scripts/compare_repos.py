#!/usr/bin/env python3
"""Compare student repositories using local embeddings + Qdrant.

This script scans the graded `work/` directory, builds lightweight text
representations of each repository (file structure + README/deps), embeds them
locally, writes the vectors to a local Qdrant instance, and then queries
nearest neighbours to highlight similar projects.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import Counter
from pathlib import Path
from typing import List, Tuple
from uuid import uuid4

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from tqdm import tqdm

# Directories we skip to keep the corpus clean
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache"}


def read_text(path: Path, limit: int = 4000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:limit]
    except OSError:
        return ""


def is_code_file(path: Path) -> bool:
    return path.suffix.lower() in {
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".java",
        ".go",
        ".rs",
        ".rb",
        ".php",
        ".cs",
        ".swift",
        ".kt",
        ".scala",
    }


def sample_code_snippets(
    repo_path: Path,
    files: List[str],
    max_files: int,
    per_file_chars: int,
    total_chars: int,
) -> List[Tuple[str, str]]:
    """Return [(relative_path, snippet)] for up to max_files, bounded by total_chars."""
    code_paths = [f for f in files if is_code_file(Path(f))]
    non_tests = [f for f in code_paths if "test" not in Path(f).stem.lower()]
    tests = [f for f in code_paths if f not in non_tests]

    ordered = non_tests + tests
    snippets: List[Tuple[str, str]] = []
    budget = total_chars

    for rel in ordered[:max_files]:
        full = repo_path / rel
        try:
            text = full.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        snippet = text[:per_file_chars]
        if budget - len(snippet) < 0:
            break
        snippets.append((rel, snippet))
        budget -= len(snippet)
        if budget <= 0:
            break
    return snippets


def parse_dependencies(text: str) -> list[str]:
    """Rudimentary extraction of dependency names from requirements/pyproject."""
    names = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # requirements style
        if "==" in line or ">=" in line or "~=" in line or "<=" in line or ">" in line:
            names.append(re.split(r"[=<>~]", line, maxsplit=1)[0].strip())
        elif "[" in line and "]" in line:
            names.append(line.split("[", 1)[0].strip())
        elif " = " in line and "\"" in line:
            # pyproject style: foo = "^1.2.3"
            names.append(line.split("=", 1)[0].strip().strip('"').strip("'"))
        else:
            if re.match(r"^[A-Za-z0-9_.-]+$", line):
                names.append(line)
    # de-duplicate preserving order
    seen = set()
    uniq = []
    for n in names:
        if n and n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq


def collect_files(repo_path: Path) -> Tuple[List[str], Counter]:
    """Return relative file paths and a language counter."""
    files: List[str] = []
    lang_counts: Counter[str] = Counter()
    dir_counts: Counter[str] = Counter()

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

    return sorted(files), lang_counts, dir_counts


def build_embedding_text(
    repo_path: Path,
    repo_name: str,
    files: List[str],
    lang_counts: Counter,
    dir_counts: Counter,
    max_files: int,
    code_snippets: List[Tuple[str, str]],
) -> Tuple[str, dict]:
    """Compose a single embedding text plus payload we push to Qdrant."""
    file_sample = files[:max_files]
    python_files = len([f for f in files if f.endswith(".py")])
    test_files = len([f for f in files if "test" in Path(f).name.lower()])
    has_docker = (repo_path / "Dockerfile").exists() or (repo_path / "docker-compose.yml").exists()
    has_readme = (repo_path / "README.md").exists()

    readme_excerpt = read_text(repo_path / "README.md", limit=2000)
    deps_excerpt = ""
    deps_list: list[str] = []
    if (repo_path / "requirements.txt").exists():
        deps_excerpt = read_text(repo_path / "requirements.txt", limit=1500)
        deps_list = parse_dependencies(deps_excerpt)
    elif (repo_path / "pyproject.toml").exists():
        deps_excerpt = read_text(repo_path / "pyproject.toml", limit=1500)
        deps_list = parse_dependencies(deps_excerpt)
    elif (repo_path / "package.json").exists():
        deps_excerpt = read_text(repo_path / "package.json", limit=1500)
        deps_list = parse_dependencies(deps_excerpt)

    payload = {
        "repo": repo_name,
        "path": str(repo_path),
        "file_count": len(files),
        "python_files": python_files,
        "test_files": test_files,
        "has_docker": has_docker,
        "has_readme": has_readme,
        "top_languages": dict(lang_counts.most_common(5)),
        "top_dirs": dict(dir_counts.most_common(5)),
        "file_sample": file_sample[:50],
        "readme_excerpt": readme_excerpt[:500],
        "deps_excerpt": deps_excerpt[:500],
        "deps_list": deps_list[:50],
        "code_snippet_paths": [p for p, _ in code_snippets],
    }

    parts: List[str] = [
        f"Repository: {repo_name}",
        f"Files: total={len(files)}, python={python_files}, tests={test_files}, docker={has_docker}",
        f"Top languages: {payload['top_languages']}",
        f"Top dirs: {payload['top_dirs']}",
        "File tree sample:",
        "\n".join(file_sample),
    ]

    if readme_excerpt:
        parts.append("README excerpt:")
        parts.append(readme_excerpt)

    if deps_excerpt or deps_list:
        parts.append(f"Dependencies: {', '.join(deps_list[:50])}")
        parts.append("Dependency file excerpt:")
        parts.append(deps_excerpt)

    if code_snippets:
        parts.append("Code snippets:")
        for path, snippet in code_snippets:
            parts.append(f"--- {path} ---")
            parts.append(snippet)

    embedding_text = "\n".join(parts)
    return embedding_text, payload


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


def write_results_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["repo", "match", "score", "file_overlap", "repo_path", "match_path"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare repositories by embedding structure/metadata into Qdrant."
    )
    parser.add_argument("--root", default="work", help="Directory containing submission folders")
    parser.add_argument(
        "--repo-subdir",
        default="repo",
        help="Subdirectory that holds the cloned repo inside each submission folder",
    )
    parser.add_argument(
        "--collection",
        default="repo_similarity",
        help="Qdrant collection to use for vectors",
    )
    parser.add_argument(
        "--qdrant-url",
        default=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        help="Qdrant base URL",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
        help="FastEmbed model name",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Top similar repos to fetch per repo")
    parser.add_argument("--max-files", type=int, default=400, help="How many file paths to embed")
    parser.add_argument("--reset", action="store_true", help="Drop the collection before writing")
    parser.add_argument(
        "--output",
        default="results/repo_similarity.csv",
        help="Where to write the CSV summary of matches",
    )
    parser.add_argument(
        "--code-samples",
        type=int,
        default=20,
        help="Max number of code files to sample for content",
    )
    parser.add_argument(
        "--code-chars-per-file",
        type=int,
        default=800,
        help="Chars per sampled code file",
    )
    parser.add_argument(
        "--code-total-chars",
        type=int,
        default=12000,
        help="Total cap on code chars per repo",
    )

    args = parser.parse_args()
    root_dir = Path(args.root)
    if not root_dir.exists():
        raise SystemExit(f"Root directory not found: {root_dir}")

    submissions = sorted([p for p in root_dir.iterdir() if p.is_dir()])
    repo_paths: List[Path] = []
    for submission in submissions:
        repo_dir = submission / args.repo_subdir
        if repo_dir.is_dir():
            repo_paths.append(repo_dir)
        elif submission.is_dir():
            repo_paths.append(submission)

    if not repo_paths:
        raise SystemExit("No repositories found under the given root.")

    embedder = TextEmbedding(model_name=args.model)
    client = QdrantClient(url=args.qdrant_url)

    # Derive vector size safely (fastembed exposes embedding_size; fallback to a probe)
    size = getattr(embedder, "embedding_size", None)
    if not size:
        size = len(next(embedder.embed(["probe"])))

    try:
        ensure_collection(client, args.collection, int(size), args.reset)
    except Exception as exc:  # pragma: no cover - informative exit for runtime issues
        raise SystemExit(
            f"Could not reach Qdrant at {args.qdrant_url}. "
            "Start the container with `make qdrant-up` first."
        ) from exc

    points: List[qmodels.PointStruct] = []
    repo_payloads = {}
    file_sets = {}

    print(f"Indexing {len(repo_paths)} repos into Qdrant ({args.collection})...")
    for repo_path in tqdm(repo_paths, desc="Embedding repos"):
        files, lang_counts, dir_counts = collect_files(repo_path)
        repo_name = repo_path.parent.name if repo_path.name == args.repo_subdir else repo_path.name
        code_snippets = sample_code_snippets(
            repo_path,
            files,
            max_files=args.code_samples,
            per_file_chars=args.code_chars_per_file,
            total_chars=args.code_total_chars,
        )
        embedding_text, payload = build_embedding_text(
            repo_path,
            repo_name,
            files,
            lang_counts,
            dir_counts,
            args.max_files,
            code_snippets,
        )
        vector = next(embedder.embed([embedding_text]))
        point_id = str(uuid4())

        payload["id"] = point_id
        repo_payloads[point_id] = payload
        file_sets[point_id] = set(files[: args.max_files])

        points.append(
            qmodels.PointStruct(
                id=point_id,
                vector=vector.tolist(),
                payload=payload,
            )
        )

    client.upsert(collection_name=args.collection, points=points, wait=True)

    print("Querying nearest neighbours...")
    rows = []
    for point in tqdm(points, desc="Searching"):
        hits = client.query_points(
            collection_name=args.collection,
            query=point.vector,
            limit=args.top_k + 1,
            with_payload=True,
        ).points

        for hit in hits:
            if str(hit.id) == str(point.id):
                continue
            source_payload = repo_payloads[point.id]
            target_payload = repo_payloads.get(str(hit.id)) or hit.payload
            overlap = len(file_sets.get(point.id, set()) & file_sets.get(str(hit.id), set()))
            rows.append(
                {
                    "repo": source_payload["repo"],
                    "match": target_payload.get("repo"),
                    "score": round(hit.score, 4),
                    "file_overlap": overlap,
                    "repo_path": source_payload["path"],
                    "match_path": target_payload.get("path"),
                }
            )
            if len([r for r in rows if r["repo"] == source_payload["repo"]]) >= args.top_k:
                break

    output_path = Path(args.output)
    write_results_csv(output_path, rows)
    print(f"Wrote similarity report to {output_path}")


if __name__ == "__main__":
    main()
