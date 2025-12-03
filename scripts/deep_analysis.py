#!/usr/bin/env python3
"""Deep Code Analysis & Clustering System.

Performs comprehensive multi-dimensional analysis of student repositories:
1. Code Structure Analysis (AST-based)
2. Coding Style Analysis (formatting, naming conventions)
3. Architecture Analysis (file organization, patterns)
4. Semantic Analysis (embeddings of code meaning)
5. Dependency Analysis (libraries, versions)
6. Documentation Analysis (comments, docstrings)
7. Test Coverage Analysis
8. API Design Analysis (endpoints, schemas)

Then clusters repositories and visualizes similarities.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Tuple, Dict, Set, Optional, Any
from uuid import uuid4

import numpy as np
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"}


# ---------------------------------------------------------------------------
# Deep Analysis Data Structures
# ---------------------------------------------------------------------------
@dataclass
class StyleMetrics:
    """Coding style metrics."""
    avg_line_length: float = 0.0
    max_line_length: int = 0
    avg_function_length: float = 0.0
    avg_class_length: float = 0.0
    indentation_style: str = "spaces"  # spaces or tabs
    indent_size: int = 4
    blank_line_ratio: float = 0.0
    comment_ratio: float = 0.0
    docstring_ratio: float = 0.0
    uses_type_hints: bool = False
    type_hint_coverage: float = 0.0
    naming_style: Dict[str, str] = field(default_factory=dict)  # snake_case, camelCase, etc.
    

@dataclass
class ArchitectureMetrics:
    """Architecture and organization metrics."""
    total_files: int = 0
    total_lines: int = 0
    total_code_lines: int = 0
    directory_depth: int = 0
    has_app_folder: bool = False
    has_tests_folder: bool = False
    has_models_file: bool = False
    has_routes_file: bool = False
    has_schemas_file: bool = False
    has_crud_file: bool = False
    has_config_file: bool = False
    has_main_file: bool = False
    has_dockerfile: bool = False
    has_docker_compose: bool = False
    has_readme: bool = False
    has_requirements: bool = False
    has_pyproject: bool = False
    has_env_example: bool = False
    file_organization_pattern: str = "flat"  # flat, layered, domain-driven
    

@dataclass
class APIMetrics:
    """API design metrics."""
    total_endpoints: int = 0
    get_endpoints: int = 0
    post_endpoints: int = 0
    put_endpoints: int = 0
    patch_endpoints: int = 0
    delete_endpoints: int = 0
    endpoint_paths: List[str] = field(default_factory=list)
    uses_path_params: bool = False
    uses_query_params: bool = False
    uses_request_body: bool = False
    response_models: List[str] = field(default_factory=list)
    http_exceptions_used: List[int] = field(default_factory=list)
    has_pagination: bool = False
    has_filtering: bool = False
    has_sorting: bool = False
    

@dataclass 
class CodeComplexity:
    """Code complexity metrics."""
    total_functions: int = 0
    total_classes: int = 0
    total_methods: int = 0
    avg_cyclomatic_complexity: float = 0.0
    max_nesting_depth: int = 0
    total_imports: int = 0
    unique_imports: int = 0
    stdlib_imports: int = 0
    third_party_imports: int = 0
    local_imports: int = 0
    has_async_code: bool = False
    has_generators: bool = False
    has_decorators: bool = False
    has_context_managers: bool = False
    has_comprehensions: bool = False
    exception_handling_count: int = 0
    

@dataclass
class SemanticProfile:
    """Semantic code features for embedding."""
    function_signatures: List[str] = field(default_factory=list)
    class_definitions: List[str] = field(default_factory=list)
    docstrings: List[str] = field(default_factory=list)
    comments: List[str] = field(default_factory=list)
    string_literals: List[str] = field(default_factory=list)
    variable_names: List[str] = field(default_factory=list)
    api_patterns: List[str] = field(default_factory=list)
    error_messages: List[str] = field(default_factory=list)
    

@dataclass
class CodeFingerprints:
    """Various fingerprints for similarity detection."""
    function_hashes: Dict[str, str] = field(default_factory=dict)  # name -> hash
    class_hashes: Dict[str, str] = field(default_factory=dict)
    normalized_code_hash: str = ""
    structural_hash: str = ""  # AST structure only
    ngram_fingerprints: Set[str] = field(default_factory=set)
    token_sequence_hash: str = ""
    

@dataclass
class DeepAnalysis:
    """Complete deep analysis of a repository."""
    name: str
    path: str
    style: StyleMetrics
    architecture: ArchitectureMetrics
    api: APIMetrics
    complexity: CodeComplexity
    semantic: SemanticProfile
    fingerprints: CodeFingerprints
    dependencies: List[str]
    frameworks: List[str]
    raw_code_sample: str = ""  # For embedding
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "style": asdict(self.style),
            "architecture": asdict(self.architecture),
            "api": asdict(self.api),
            "complexity": asdict(self.complexity),
            "dependencies": self.dependencies,
            "frameworks": self.frameworks,
        }


# ---------------------------------------------------------------------------
# Code Analysis Functions
# ---------------------------------------------------------------------------
def analyze_style(sources: List[Tuple[str, str]]) -> StyleMetrics:
    """Analyze coding style across all source files."""
    metrics = StyleMetrics()
    
    all_lines = []
    function_lengths = []
    class_lengths = []
    type_hinted_funcs = 0
    total_funcs = 0
    docstring_count = 0
    comment_lines = 0
    blank_lines = 0
    naming_samples = {"functions": [], "variables": [], "classes": []}
    
    for filepath, source in sources:
        lines = source.splitlines()
        all_lines.extend(lines)
        
        for line in lines:
            if not line.strip():
                blank_lines += 1
            elif line.strip().startswith("#"):
                comment_lines += 1
        
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    total_funcs += 1
                    naming_samples["functions"].append(node.name)
                    if hasattr(node, 'end_lineno') and hasattr(node, 'lineno'):
                        function_lengths.append(node.end_lineno - node.lineno)
                    if node.returns is not None or any(arg.annotation for arg in node.args.args):
                        type_hinted_funcs += 1
                    # Check for docstring
                    if (node.body and isinstance(node.body[0], ast.Expr) and
                        isinstance(node.body[0].value, ast.Constant) and
                        isinstance(node.body[0].value.value, str)):
                        docstring_count += 1
                        
                elif isinstance(node, ast.ClassDef):
                    naming_samples["classes"].append(node.name)
                    if hasattr(node, 'end_lineno') and hasattr(node, 'lineno'):
                        class_lengths.append(node.end_lineno - node.lineno)
                        
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    naming_samples["variables"].append(node.id)
        except SyntaxError:
            pass
    
    if all_lines:
        line_lengths = [len(l) for l in all_lines if l.strip()]
        if line_lengths:
            metrics.avg_line_length = statistics.mean(line_lengths)
            metrics.max_line_length = max(line_lengths)
        metrics.blank_line_ratio = blank_lines / len(all_lines) if all_lines else 0
        metrics.comment_ratio = comment_lines / len(all_lines) if all_lines else 0
    
    if function_lengths:
        metrics.avg_function_length = statistics.mean(function_lengths)
    if class_lengths:
        metrics.avg_class_length = statistics.mean(class_lengths)
    
    if total_funcs:
        metrics.docstring_ratio = docstring_count / total_funcs
        metrics.type_hint_coverage = type_hinted_funcs / total_funcs
        metrics.uses_type_hints = type_hinted_funcs > 0
    
    # Detect naming conventions
    def detect_convention(names):
        if not names:
            return "unknown"
        snake = sum(1 for n in names if '_' in n and n.islower())
        camel = sum(1 for n in names if n[0].islower() and any(c.isupper() for c in n))
        pascal = sum(1 for n in names if n[0].isupper() and any(c.islower() for c in n))
        if snake > camel and snake > pascal:
            return "snake_case"
        elif camel > snake:
            return "camelCase"
        elif pascal > snake:
            return "PascalCase"
        return "mixed"
    
    metrics.naming_style = {
        "functions": detect_convention(naming_samples["functions"]),
        "variables": detect_convention(naming_samples["variables"]),
        "classes": detect_convention(naming_samples["classes"]),
    }
    
    # Detect indentation
    for line in all_lines:
        if line.startswith('\t'):
            metrics.indentation_style = "tabs"
            break
        elif line.startswith('    '):
            metrics.indent_size = 4
            break
        elif line.startswith('  '):
            metrics.indent_size = 2
            break
    
    return metrics


def analyze_architecture(repo_path: Path, files: List[str]) -> ArchitectureMetrics:
    """Analyze repository architecture and organization."""
    metrics = ArchitectureMetrics()
    metrics.total_files = len(files)
    
    file_set = set(f.lower() for f in files)
    dir_set = set(str(Path(f).parent).lower() for f in files)
    
    # Check for common patterns
    metrics.has_app_folder = any('app/' in f or 'src/' in f for f in files)
    metrics.has_tests_folder = any('test' in d for d in dir_set)
    metrics.has_models_file = any('models.py' in f or 'model.py' in f for f in file_set)
    metrics.has_routes_file = any('routes.py' in f or 'router.py' in f or 'routers/' in f for f in files)
    metrics.has_schemas_file = any('schemas.py' in f or 'schema.py' in f for f in file_set)
    metrics.has_crud_file = any('crud.py' in f for f in file_set)
    metrics.has_config_file = any('config.py' in f or 'settings.py' in f for f in file_set)
    metrics.has_main_file = any('main.py' in f for f in file_set)
    
    metrics.has_dockerfile = (repo_path / "Dockerfile").exists()
    metrics.has_docker_compose = (repo_path / "docker-compose.yml").exists() or (repo_path / "docker-compose.yaml").exists()
    metrics.has_readme = (repo_path / "README.md").exists()
    metrics.has_requirements = (repo_path / "requirements.txt").exists()
    metrics.has_pyproject = (repo_path / "pyproject.toml").exists()
    metrics.has_env_example = (repo_path / ".env.example").exists() or (repo_path / "env.example").exists()
    
    # Calculate directory depth
    if files:
        depths = [len(Path(f).parts) for f in files]
        metrics.directory_depth = max(depths)
    
    # Detect organization pattern
    if metrics.has_app_folder and (metrics.has_models_file or metrics.has_routes_file):
        if any('domain/' in f or 'modules/' in f for f in files):
            metrics.file_organization_pattern = "domain-driven"
        else:
            metrics.file_organization_pattern = "layered"
    else:
        metrics.file_organization_pattern = "flat"
    
    return metrics


def analyze_api(sources: List[Tuple[str, str]]) -> APIMetrics:
    """Analyze API design from source code."""
    metrics = APIMetrics()
    
    endpoint_pattern = re.compile(
        r'@(?:app|router)\.(get|post|put|patch|delete)\s*\(\s*["\']([^"\']+)["\']',
        re.IGNORECASE
    )
    path_param_pattern = re.compile(r'\{[^}]+\}')
    query_param_pattern = re.compile(r'Query\s*\(')
    body_pattern = re.compile(r'Body\s*\(')
    response_model_pattern = re.compile(r'response_model\s*=\s*(\w+)')
    http_exception_pattern = re.compile(r'HTTPException\s*\(\s*status_code\s*=\s*(\d+)')
    pagination_patterns = ['skip', 'limit', 'page', 'offset', 'per_page']
    filter_patterns = ['filter', 'search', 'query', 'where']
    sort_patterns = ['sort', 'order', 'order_by', 'sort_by']
    
    for filepath, source in sources:
        # Find endpoints
        for match in endpoint_pattern.finditer(source):
            method = match.group(1).lower()
            path = match.group(2)
            metrics.total_endpoints += 1
            metrics.endpoint_paths.append(f"{method.upper()} {path}")
            
            if method == 'get':
                metrics.get_endpoints += 1
            elif method == 'post':
                metrics.post_endpoints += 1
            elif method == 'put':
                metrics.put_endpoints += 1
            elif method == 'patch':
                metrics.patch_endpoints += 1
            elif method == 'delete':
                metrics.delete_endpoints += 1
            
            if path_param_pattern.search(path):
                metrics.uses_path_params = True
        
        # Check for query params, body, response models
        if query_param_pattern.search(source):
            metrics.uses_query_params = True
        if body_pattern.search(source):
            metrics.uses_request_body = True
        
        for match in response_model_pattern.finditer(source):
            metrics.response_models.append(match.group(1))
        
        for match in http_exception_pattern.finditer(source):
            metrics.http_exceptions_used.append(int(match.group(1)))
        
        source_lower = source.lower()
        if any(p in source_lower for p in pagination_patterns):
            metrics.has_pagination = True
        if any(p in source_lower for p in filter_patterns):
            metrics.has_filtering = True
        if any(p in source_lower for p in sort_patterns):
            metrics.has_sorting = True
    
    metrics.response_models = list(set(metrics.response_models))
    metrics.http_exceptions_used = list(set(metrics.http_exceptions_used))
    
    return metrics


def analyze_complexity(sources: List[Tuple[str, str]]) -> CodeComplexity:
    """Analyze code complexity metrics."""
    metrics = CodeComplexity()
    
    all_imports = []
    stdlib_modules = {
        'os', 'sys', 'json', 'typing', 'collections', 'itertools', 'functools',
        'pathlib', 'datetime', 're', 'math', 'random', 'hashlib', 'uuid',
        'dataclasses', 'abc', 'contextlib', 'logging', 'unittest', 'asyncio'
    }
    
    for filepath, source in sources:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                metrics.total_functions += 1
                if isinstance(node, ast.AsyncFunctionDef):
                    metrics.has_async_code = True
                if node.decorator_list:
                    metrics.has_decorators = True
                    
            elif isinstance(node, ast.ClassDef):
                metrics.total_classes += 1
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        metrics.total_methods += 1
                        
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    mod = alias.name.split('.')[0]
                    all_imports.append(mod)
                    
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    mod = node.module.split('.')[0]
                    all_imports.append(mod)
                    
            elif isinstance(node, ast.Yield) or isinstance(node, ast.YieldFrom):
                metrics.has_generators = True
                
            elif isinstance(node, ast.With) or isinstance(node, ast.AsyncWith):
                metrics.has_context_managers = True
                
            elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                metrics.has_comprehensions = True
                
            elif isinstance(node, (ast.Try, ast.ExceptHandler)):
                metrics.exception_handling_count += 1
    
    metrics.total_imports = len(all_imports)
    metrics.unique_imports = len(set(all_imports))
    metrics.stdlib_imports = sum(1 for i in all_imports if i in stdlib_modules)
    metrics.third_party_imports = metrics.unique_imports - metrics.stdlib_imports
    
    return metrics


def extract_semantic_profile(sources: List[Tuple[str, str]]) -> SemanticProfile:
    """Extract semantic features for embedding."""
    profile = SemanticProfile()
    
    for filepath, source in sources:
        # Extract comments
        for match in re.finditer(r'#\s*(.+)$', source, re.MULTILINE):
            comment = match.group(1).strip()
            if len(comment) > 5:
                profile.comments.append(comment)
        
        # Extract string literals (potential messages, labels)
        for match in re.finditer(r'["\']([^"\']{10,100})["\']', source):
            s = match.group(1)
            if not s.startswith(('/', 'http', '{', '<')):
                profile.string_literals.append(s)
        
        # Extract error messages
        for match in re.finditer(r'(?:detail|message|msg)\s*=\s*["\']([^"\']+)["\']', source, re.IGNORECASE):
            profile.error_messages.append(match.group(1))
        
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Function signature
                    args = [arg.arg for arg in node.args.args]
                    ret = ""
                    if node.returns:
                        ret = f" -> {ast.unparse(node.returns)}" if hasattr(ast, 'unparse') else ""
                    sig = f"def {node.name}({', '.join(args)}){ret}"
                    profile.function_signatures.append(sig)
                    
                    # Docstring
                    if (node.body and isinstance(node.body[0], ast.Expr) and
                        isinstance(node.body[0].value, ast.Constant)):
                        doc = node.body[0].value.value
                        if isinstance(doc, str) and len(doc) > 10:
                            profile.docstrings.append(doc[:200])
                
                elif isinstance(node, ast.ClassDef):
                    bases = [ast.unparse(b) if hasattr(ast, 'unparse') else str(b) for b in node.bases]
                    profile.class_definitions.append(f"class {node.name}({', '.join(bases)})")
                    
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    profile.variable_names.append(node.id)
        except SyntaxError:
            pass
    
    # Deduplicate
    profile.variable_names = list(set(profile.variable_names))[:100]
    profile.string_literals = list(set(profile.string_literals))[:50]
    profile.error_messages = list(set(profile.error_messages))[:30]
    
    return profile


def compute_fingerprints(sources: List[Tuple[str, str]]) -> CodeFingerprints:
    """Compute various code fingerprints."""
    fingerprints = CodeFingerprints()
    
    all_normalized = []
    all_tokens = []
    
    for filepath, source in sources:
        # Skip test files for fingerprinting
        if 'test' in filepath.lower():
            continue
        
        # Normalize code
        normalized = re.sub(r'#.*$', '', source, flags=re.MULTILINE)
        normalized = re.sub(r'"""[\s\S]*?"""', '', normalized)
        normalized = re.sub(r"'''[\s\S]*?'''", '', normalized)
        lines = [l.strip() for l in normalized.splitlines() if l.strip()]
        normalized = '\n'.join(lines)
        all_normalized.append(normalized)
        
        # Tokenize
        tokens = re.findall(r'\b\w+\b', normalized)
        all_tokens.extend(tokens)
        
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                        func_source = '\n'.join(source.splitlines()[node.lineno-1:node.end_lineno])
                        func_normalized = re.sub(r'#.*$', '', func_source, flags=re.MULTILINE)
                        func_normalized = re.sub(r'\s+', ' ', func_normalized).strip()
                        fingerprints.function_hashes[node.name] = hashlib.md5(func_normalized.encode()).hexdigest()
                        
                elif isinstance(node, ast.ClassDef):
                    if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                        class_source = '\n'.join(source.splitlines()[node.lineno-1:node.end_lineno])
                        class_normalized = re.sub(r'#.*$', '', class_source, flags=re.MULTILINE)
                        class_normalized = re.sub(r'\s+', ' ', class_normalized).strip()
                        fingerprints.class_hashes[node.name] = hashlib.md5(class_normalized.encode()).hexdigest()
        except SyntaxError:
            pass
    
    # Overall fingerprints
    combined = '\n'.join(all_normalized)
    fingerprints.normalized_code_hash = hashlib.md5(combined.encode()).hexdigest()
    fingerprints.token_sequence_hash = hashlib.md5(' '.join(all_tokens[:5000]).encode()).hexdigest()
    
    # N-gram fingerprints
    if len(all_tokens) >= 7:
        for i in range(len(all_tokens) - 7):
            ngram = ' '.join(all_tokens[i:i+7])
            fingerprints.ngram_fingerprints.add(hashlib.md5(ngram.encode()).hexdigest()[:8])
    
    return fingerprints


def parse_dependencies(repo_path: Path) -> Tuple[List[str], List[str]]:
    """Parse dependencies and detect frameworks."""
    deps = []
    frameworks = []
    
    framework_map = {
        'fastapi': 'fastapi',
        'flask': 'flask',
        'django': 'django',
        'pydantic': 'pydantic',
        'sqlalchemy': 'sqlalchemy',
        'sqlmodel': 'sqlmodel',
        'pytest': 'pytest',
        'httpx': 'httpx',
        'uvicorn': 'uvicorn',
        'alembic': 'alembic',
        'redis': 'redis',
        'celery': 'celery',
        'mongodb': 'mongodb',
        'motor': 'motor',
    }
    
    # Try requirements.txt
    req_file = repo_path / "requirements.txt"
    if req_file.exists():
        try:
            content = req_file.read_text()
            for line in content.splitlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    pkg = re.split(r'[=<>~!]', line)[0].strip().lower()
                    if pkg:
                        deps.append(pkg)
                        if pkg in framework_map:
                            frameworks.append(framework_map[pkg])
        except:
            pass
    
    # Try pyproject.toml
    pyproj_file = repo_path / "pyproject.toml"
    if pyproj_file.exists():
        try:
            content = pyproj_file.read_text()
            # Simple extraction
            for match in re.finditer(r'"([a-zA-Z0-9_-]+)', content):
                pkg = match.group(1).lower()
                if pkg in framework_map:
                    frameworks.append(framework_map[pkg])
                    if pkg not in deps:
                        deps.append(pkg)
        except:
            pass
    
    return list(set(deps)), list(set(frameworks))


# ---------------------------------------------------------------------------
# Main Analysis Function
# ---------------------------------------------------------------------------
def deep_analyze_repo(repo_path: Path, repo_name: str) -> DeepAnalysis:
    """Perform deep analysis of a repository."""
    
    # Collect all Python source files
    sources: List[Tuple[str, str]] = []
    files: List[str] = []
    
    for root, dirs, file_names in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in file_names:
            if fname.endswith('.py'):
                full_path = Path(root) / fname
                rel_path = str(full_path.relative_to(repo_path))
                files.append(rel_path)
                try:
                    source = full_path.read_text(encoding='utf-8', errors='ignore')
                    sources.append((rel_path, source))
                except:
                    pass
    
    # Run all analyses
    style = analyze_style(sources)
    architecture = analyze_architecture(repo_path, files)
    api = analyze_api(sources)
    complexity = analyze_complexity(sources)
    semantic = extract_semantic_profile(sources)
    fingerprints = compute_fingerprints(sources)
    deps, frameworks = parse_dependencies(repo_path)
    
    # Count lines
    total_lines = sum(len(s.splitlines()) for _, s in sources)
    code_lines = sum(len([l for l in s.splitlines() if l.strip() and not l.strip().startswith('#')]) for _, s in sources)
    architecture.total_lines = total_lines
    architecture.total_code_lines = code_lines
    
    # Build raw code sample for embedding (excluding tests)
    code_sample_parts = []
    for fpath, source in sources:
        if 'test' not in fpath.lower():
            code_sample_parts.append(source[:3000])
    raw_code_sample = '\n---\n'.join(code_sample_parts)[:15000]
    
    return DeepAnalysis(
        name=repo_name,
        path=str(repo_path),
        style=style,
        architecture=architecture,
        api=api,
        complexity=complexity,
        semantic=semantic,
        fingerprints=fingerprints,
        dependencies=deps,
        frameworks=frameworks,
        raw_code_sample=raw_code_sample,
    )


# ---------------------------------------------------------------------------
# Similarity Computation
# ---------------------------------------------------------------------------
def compute_deep_similarity(a1: DeepAnalysis, a2: DeepAnalysis) -> Dict[str, float]:
    """Compute multi-dimensional similarity between two analyses."""
    
    def jaccard(s1: set, s2: set) -> float:
        if not s1 and not s2:
            return 0.0
        intersection = len(s1 & s2)
        union = len(s1 | s2)
        return intersection / union if union > 0 else 0.0
    
    def list_jaccard(l1: list, l2: list) -> float:
        return jaccard(set(l1), set(l2))
    
    def numeric_similarity(v1: float, v2: float, max_diff: float) -> float:
        if max_diff == 0:
            return 1.0 if v1 == v2 else 0.0
        return max(0, 1 - abs(v1 - v2) / max_diff)
    
    similarities = {}
    
    # Style similarity
    style_sims = []
    style_sims.append(numeric_similarity(a1.style.avg_line_length, a2.style.avg_line_length, 50))
    style_sims.append(numeric_similarity(a1.style.avg_function_length, a2.style.avg_function_length, 30))
    style_sims.append(numeric_similarity(a1.style.comment_ratio, a2.style.comment_ratio, 0.3))
    style_sims.append(numeric_similarity(a1.style.type_hint_coverage, a2.style.type_hint_coverage, 1.0))
    style_sims.append(1.0 if a1.style.naming_style == a2.style.naming_style else 0.5)
    similarities['style'] = statistics.mean(style_sims) if style_sims else 0.0
    
    # Architecture similarity
    arch_features1 = {
        'app_folder': a1.architecture.has_app_folder,
        'tests_folder': a1.architecture.has_tests_folder,
        'models': a1.architecture.has_models_file,
        'routes': a1.architecture.has_routes_file,
        'schemas': a1.architecture.has_schemas_file,
        'crud': a1.architecture.has_crud_file,
        'config': a1.architecture.has_config_file,
        'docker': a1.architecture.has_dockerfile,
        'compose': a1.architecture.has_docker_compose,
    }
    arch_features2 = {
        'app_folder': a2.architecture.has_app_folder,
        'tests_folder': a2.architecture.has_tests_folder,
        'models': a2.architecture.has_models_file,
        'routes': a2.architecture.has_routes_file,
        'schemas': a2.architecture.has_schemas_file,
        'crud': a2.architecture.has_crud_file,
        'config': a2.architecture.has_config_file,
        'docker': a2.architecture.has_dockerfile,
        'compose': a2.architecture.has_docker_compose,
    }
    arch_match = sum(1 for k in arch_features1 if arch_features1[k] == arch_features2[k])
    similarities['architecture'] = arch_match / len(arch_features1)
    
    # API similarity
    api_sims = []
    api_sims.append(numeric_similarity(a1.api.total_endpoints, a2.api.total_endpoints, 20))
    api_sims.append(list_jaccard(a1.api.endpoint_paths, a2.api.endpoint_paths))
    api_sims.append(list_jaccard(a1.api.response_models, a2.api.response_models))
    api_sims.append(list_jaccard(a1.api.http_exceptions_used, a2.api.http_exceptions_used))
    similarities['api_design'] = statistics.mean(api_sims) if api_sims else 0.0
    
    # Complexity similarity  
    comp_sims = []
    comp_sims.append(numeric_similarity(a1.complexity.total_functions, a2.complexity.total_functions, 50))
    comp_sims.append(numeric_similarity(a1.complexity.total_classes, a2.complexity.total_classes, 20))
    comp_sims.append(1.0 if a1.complexity.has_async_code == a2.complexity.has_async_code else 0.0)
    similarities['complexity'] = statistics.mean(comp_sims) if comp_sims else 0.0
    
    # Dependency similarity
    similarities['dependencies'] = list_jaccard(a1.dependencies, a2.dependencies)
    similarities['frameworks'] = list_jaccard(a1.frameworks, a2.frameworks)
    
    # Semantic similarity
    semantic_sims = []
    semantic_sims.append(list_jaccard(a1.semantic.function_signatures, a2.semantic.function_signatures))
    semantic_sims.append(list_jaccard(a1.semantic.class_definitions, a2.semantic.class_definitions))
    semantic_sims.append(list_jaccard(a1.semantic.variable_names[:50], a2.semantic.variable_names[:50]))
    semantic_sims.append(list_jaccard(a1.semantic.error_messages, a2.semantic.error_messages))
    similarities['semantic'] = statistics.mean(semantic_sims) if semantic_sims else 0.0
    
    # Code fingerprint similarity (IMPORTANT for plagiarism)
    fp_sims = []
    # Function hash overlap
    common_funcs = set(a1.fingerprints.function_hashes.keys()) & set(a2.fingerprints.function_hashes.keys())
    if common_funcs:
        identical_funcs = sum(1 for f in common_funcs 
                            if a1.fingerprints.function_hashes[f] == a2.fingerprints.function_hashes[f])
        fp_sims.append(identical_funcs / len(common_funcs))
    
    # Class hash overlap
    common_classes = set(a1.fingerprints.class_hashes.keys()) & set(a2.fingerprints.class_hashes.keys())
    if common_classes:
        identical_classes = sum(1 for c in common_classes
                               if a1.fingerprints.class_hashes[c] == a2.fingerprints.class_hashes[c])
        fp_sims.append(identical_classes / len(common_classes))
    
    # N-gram overlap
    ngram_overlap = jaccard(a1.fingerprints.ngram_fingerprints, a2.fingerprints.ngram_fingerprints)
    fp_sims.append(ngram_overlap)
    
    similarities['fingerprint'] = statistics.mean(fp_sims) if fp_sims else 0.0
    
    # Combined score (weighted)
    weights = {
        'style': 0.05,
        'architecture': 0.10,
        'api_design': 0.15,
        'complexity': 0.05,
        'dependencies': 0.10,
        'frameworks': 0.10,
        'semantic': 0.20,
        'fingerprint': 0.25,
    }
    similarities['combined'] = sum(similarities[k] * weights[k] for k in weights)
    
    return similarities


# ---------------------------------------------------------------------------
# Embedding Functions
# ---------------------------------------------------------------------------
def build_embedding_text(analysis: DeepAnalysis) -> str:
    """Build comprehensive text for embedding."""
    parts = [
        f"Project: {analysis.name}",
        f"Frameworks: {', '.join(analysis.frameworks)}",
        f"Dependencies: {', '.join(analysis.dependencies[:30])}",
        f"Total files: {analysis.architecture.total_files}, Lines: {analysis.architecture.total_lines}",
        f"Functions: {analysis.complexity.total_functions}, Classes: {analysis.complexity.total_classes}",
        f"API Endpoints: {', '.join(analysis.api.endpoint_paths[:20])}",
        f"Response Models: {', '.join(analysis.api.response_models[:10])}",
        f"Function signatures: {'; '.join(analysis.semantic.function_signatures[:30])}",
        f"Class definitions: {'; '.join(analysis.semantic.class_definitions[:20])}",
        f"Error messages: {'; '.join(analysis.semantic.error_messages[:15])}",
        f"Variable names: {', '.join(analysis.semantic.variable_names[:50])}",
    ]
    
    if analysis.semantic.docstrings:
        parts.append(f"Documentation: {' '.join(analysis.semantic.docstrings[:10])}")
    
    # Add code sample
    if analysis.raw_code_sample:
        parts.append(f"Code sample:\n{analysis.raw_code_sample[:8000]}")
    
    return '\n'.join(parts)


# ---------------------------------------------------------------------------
# Output Functions
# ---------------------------------------------------------------------------
def save_analysis_results(
    output_dir: Path,
    analyses: List[DeepAnalysis],
    similarity_matrix: Dict[Tuple[str, str], Dict[str, float]],
    clusters: List[List[str]],
) -> None:
    """Save all analysis results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual analyses
    analyses_data = [a.to_dict() for a in analyses]
    with open(output_dir / "deep_analyses.json", "w") as f:
        json.dump(analyses_data, f, indent=2, default=str)
    
    # Save similarity matrix
    matrix_data = {f"{k[0]}|{k[1]}": v for k, v in similarity_matrix.items()}
    with open(output_dir / "deep_similarity_matrix.json", "w") as f:
        json.dump(matrix_data, f, indent=2)
    
    # Save clusters
    with open(output_dir / "clusters.json", "w") as f:
        json.dump({"clusters": clusters}, f, indent=2)
    
    # Save CSV summary
    import csv
    rows = []
    for (r1, r2), sims in sorted(similarity_matrix.items(), key=lambda x: -x[1]['combined']):
        rows.append({
            'repo1': r1,
            'repo2': r2,
            'combined': round(sims['combined'], 4),
            'fingerprint': round(sims['fingerprint'], 4),
            'semantic': round(sims['semantic'], 4),
            'api_design': round(sims['api_design'], 4),
            'architecture': round(sims['architecture'], 4),
            'dependencies': round(sims['dependencies'], 4),
            'style': round(sims['style'], 4),
        })
    
    with open(output_dir / "deep_similarity.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [])
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def cluster_repositories(
    analyses: List[DeepAnalysis],
    similarity_matrix: Dict[Tuple[str, str], Dict[str, float]],
    threshold: float = 0.5,
) -> List[List[str]]:
    """Cluster repositories based on similarity using agglomerative approach."""
    names = [a.name for a in analyses]
    n = len(names)
    
    # Build adjacency based on threshold
    adjacency = defaultdict(set)
    for (r1, r2), sims in similarity_matrix.items():
        if sims['combined'] >= threshold:
            adjacency[r1].add(r2)
            adjacency[r2].add(r1)
    
    # Find connected components (clusters)
    visited = set()
    clusters = []
    
    for name in names:
        if name in visited:
            continue
        
        # BFS to find cluster
        cluster = []
        queue = [name]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            cluster.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    queue.append(neighbor)
        
        if cluster:
            clusters.append(sorted(cluster))
    
    # Sort clusters by size (largest first)
    clusters.sort(key=len, reverse=True)
    
    return clusters


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Deep code analysis and clustering")
    parser.add_argument("--root", default="work", help="Root directory with submissions")
    parser.add_argument("--repo-subdir", default="repo", help="Subdirectory containing repo")
    parser.add_argument("--output-dir", default="results/deep_analysis", help="Output directory")
    parser.add_argument("--qdrant-url", default=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--collection", default="deep_code_analysis")
    parser.add_argument("--model", default=os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    parser.add_argument("--cluster-threshold", type=float, default=0.45, help="Similarity threshold for clustering")
    parser.add_argument("--reset", action="store_true", help="Reset Qdrant collection")
    
    args = parser.parse_args()
    root_dir = Path(args.root)
    output_dir = Path(args.output_dir)
    
    # Find repositories
    submissions = sorted([p for p in root_dir.iterdir() if p.is_dir()])
    repo_paths = []
    for sub in submissions:
        repo_dir = sub / args.repo_subdir
        if repo_dir.is_dir():
            repo_paths.append((repo_dir, sub.name))
    
    if not repo_paths:
        raise SystemExit("No repositories found")
    
    print(f"Found {len(repo_paths)} repositories")
    
    # Deep analyze all repos
    print("\n📊 Performing deep analysis...")
    analyses: List[DeepAnalysis] = []
    for repo_path, repo_name in tqdm(repo_paths, desc="Analyzing"):
        analysis = deep_analyze_repo(repo_path, repo_name)
        analyses.append(analysis)
    
    # Initialize embedder and Qdrant
    print("\n🔤 Initializing embedding model...")
    embedder = TextEmbedding(model_name=args.model)
    client = QdrantClient(url=args.qdrant_url)
    
    size = getattr(embedder, 'embedding_size', None)
    if not size:
        size = len(next(embedder.embed(["probe"])))
    
    # Create collection
    existing = {c.name for c in client.get_collections().collections}
    if args.reset and args.collection in existing:
        client.delete_collection(args.collection)
        existing.discard(args.collection)
    
    if args.collection not in existing:
        client.create_collection(
            collection_name=args.collection,
            vectors_config=qmodels.VectorParams(size=int(size), distance=qmodels.Distance.COSINE),
        )
    
    # Embed and index
    print("\n🧠 Computing embeddings...")
    vectors = {}
    points = []
    for analysis in tqdm(analyses, desc="Embedding"):
        text = build_embedding_text(analysis)
        vector = next(embedder.embed([text]))
        vectors[analysis.name] = vector
        
        point_id = str(uuid4())
        points.append(qmodels.PointStruct(
            id=point_id,
            vector=vector.tolist(),
            payload={"name": analysis.name, "path": analysis.path},
        ))
    
    client.upsert(collection_name=args.collection, points=points, wait=True)
    
    # Compute pairwise similarities
    print("\n🔍 Computing similarities...")
    similarity_matrix: Dict[Tuple[str, str], Dict[str, float]] = {}
    
    for i, a1 in enumerate(tqdm(analyses, desc="Comparing")):
        for a2 in analyses[i+1:]:
            # Deep similarity
            sims = compute_deep_similarity(a1, a2)
            
            # Add vector similarity
            v1 = vectors[a1.name]
            v2 = vectors[a2.name]
            cosine_sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            sims['vector'] = cosine_sim
            
            # Recalculate combined with vector
            sims['combined'] = (sims['combined'] * 0.7) + (cosine_sim * 0.3)
            
            similarity_matrix[(a1.name, a2.name)] = sims
    
    # Cluster repositories
    print("\n🔗 Clustering repositories...")
    clusters = cluster_repositories(analyses, similarity_matrix, args.cluster_threshold)
    
    # Save results
    print("\n💾 Saving results...")
    save_analysis_results(output_dir, analyses, similarity_matrix, clusters)
    
    # Print summary
    print("\n" + "="*70)
    print("DEEP ANALYSIS SUMMARY")
    print("="*70)
    
    print(f"\nRepositories analyzed: {len(analyses)}")
    print(f"Similarity pairs computed: {len(similarity_matrix)}")
    print(f"Clusters found: {len(clusters)}")
    
    print("\n📦 CLUSTERS:")
    for i, cluster in enumerate(clusters):
        if len(cluster) > 1:
            print(f"\n  Cluster {i+1} ({len(cluster)} repos):")
            for name in cluster:
                print(f"    • {name}")
    
    # Top similar pairs
    print("\n🔝 TOP 15 MOST SIMILAR PAIRS:")
    sorted_pairs = sorted(similarity_matrix.items(), key=lambda x: -x[1]['combined'])[:15]
    for (r1, r2), sims in sorted_pairs:
        print(f"\n  {sims['combined']:.3f}: {r1} ↔ {r2}")
        print(f"         fingerprint={sims['fingerprint']:.2f}, semantic={sims['semantic']:.2f}, "
              f"api={sims['api_design']:.2f}, vector={sims['vector']:.2f}")
    
    print(f"\n✅ Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
