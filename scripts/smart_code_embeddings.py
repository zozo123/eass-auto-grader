#!/usr/bin/env python3
"""Smart Multi-Level Code Embedding System.

This module provides deep, intelligent code embeddings that understand code at multiple levels:

1. **Token Level**: Variable names, function names, literals
2. **Statement Level**: Individual code statements and their semantics
3. **Block Level**: Functions, classes, code blocks
4. **File Level**: Entire file semantics
5. **Module Level**: Cross-file relationships and architecture
6. **Project Level**: Overall project structure and patterns

The embeddings capture:
- Syntactic structure (via AST normalization)
- Semantic meaning (via code-aware embeddings)
- Behavioral patterns (what the code does)
- Style patterns (how the code is written)
- Architectural patterns (how the code is organized)

Uses multiple embedding strategies:
- fastembed for general text embeddings
- Code-specific preprocessing for better semantic capture
- AST-based structural embeddings
- N-gram fingerprinting for copy detection
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Set, Tuple, Any, Optional
from uuid import uuid4

import numpy as np
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build", ".tox", "htmlcov"}
CODE_EXTENSIONS = {".py"}  # Focus on Python for this analysis

# Embedding collection names
COLLECTION_FUNCTION = "code_functions"
COLLECTION_CLASS = "code_classes"
COLLECTION_FILE = "code_files"
COLLECTION_PROJECT = "code_projects"
COLLECTION_COMBINED = "code_smart_combined"

# Winnowing parameters for fingerprinting
KGRAM_SIZE = 5
WINDOW_SIZE = 4


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------
@dataclass
class FunctionEmbedding:
    """Embedding data for a single function."""
    name: str
    repo: str
    file_path: str
    signature: str
    body_hash: str
    normalized_body: str
    docstring: str
    complexity: int  # Cyclomatic complexity estimate
    calls: List[str]  # Functions it calls
    decorators: List[str]
    is_async: bool
    line_count: int
    embedding_text: str = ""
    
    
@dataclass
class ClassEmbedding:
    """Embedding data for a class."""
    name: str
    repo: str
    file_path: str
    bases: List[str]
    methods: List[str]
    attributes: List[str]
    docstring: str
    decorators: List[str]
    line_count: int
    method_count: int
    embedding_text: str = ""
    

@dataclass
class FileEmbedding:
    """Embedding data for a file."""
    path: str
    repo: str
    imports: List[str]
    functions: List[str]
    classes: List[str]
    top_level_code: str
    docstring: str
    line_count: int
    code_line_count: int
    embedding_text: str = ""


@dataclass
class ProjectEmbedding:
    """Embedding data for entire project."""
    name: str
    path: str
    files: List[str]
    all_functions: List[str]
    all_classes: List[str]
    all_imports: Set[str]
    frameworks: List[str]
    api_endpoints: List[str]
    db_operations: List[str]
    architecture_pattern: str
    total_lines: int
    total_functions: int
    total_classes: int
    embedding_text: str = ""
    
    # Multi-level embeddings
    function_embeddings: List[np.ndarray] = field(default_factory=list)
    class_embeddings: List[np.ndarray] = field(default_factory=list)
    file_embeddings: List[np.ndarray] = field(default_factory=list)


# ---------------------------------------------------------------------------
# AST Normalization for Structural Comparison
# ---------------------------------------------------------------------------
class ASTNormalizer(ast.NodeTransformer):
    """Normalize AST to compare structure while ignoring superficial differences."""
    
    def __init__(self):
        self.var_counter = 0
        self.func_counter = 0
        self.class_counter = 0
        self.var_map = {}
        
    def _get_normalized_name(self, original: str, prefix: str = "var") -> str:
        """Map original names to normalized placeholders."""
        if original not in self.var_map:
            if prefix == "var":
                self.var_map[original] = f"VAR_{self.var_counter}"
                self.var_counter += 1
            elif prefix == "func":
                self.var_map[original] = f"FUNC_{self.func_counter}"
                self.func_counter += 1
            elif prefix == "class":
                self.var_map[original] = f"CLASS_{self.class_counter}"
                self.class_counter += 1
        return self.var_map[original]
    
    def visit_Name(self, node):
        """Normalize variable names."""
        # Keep built-in names and module names
        builtins = {'True', 'False', 'None', 'print', 'len', 'range', 'str', 'int', 'list', 'dict', 'set', 'tuple'}
        if node.id in builtins:
            return node
        node.id = self._get_normalized_name(node.id, "var")
        return node
    
    def visit_FunctionDef(self, node):
        """Normalize function names but preserve structure."""
        # Don't normalize special methods
        if not node.name.startswith('_'):
            node.name = self._get_normalized_name(node.name, "func")
        self.generic_visit(node)
        return node
    
    def visit_AsyncFunctionDef(self, node):
        """Same as FunctionDef for async."""
        return self.visit_FunctionDef(node)
    
    def visit_ClassDef(self, node):
        """Normalize class names."""
        if not node.name.startswith('_'):
            node.name = self._get_normalized_name(node.name, "class")
        self.generic_visit(node)
        return node
    
    def visit_Constant(self, node):
        """Normalize string and number literals."""
        if isinstance(node.value, str):
            # Keep short strings, normalize longer ones
            if len(node.value) > 20:
                node.value = "STRING_LITERAL"
        elif isinstance(node.value, (int, float)):
            # Normalize numbers except common ones
            if node.value not in (0, 1, -1, 2, 100, 200, 404, 500):
                node.value = 0
        return node


def normalize_ast(source: str) -> str:
    """Normalize Python source code's AST for structural comparison."""
    try:
        tree = ast.parse(source)
        normalizer = ASTNormalizer()
        normalized_tree = normalizer.visit(tree)
        return ast.dump(normalized_tree, indent=2)
    except SyntaxError:
        return ""


def get_ast_structure(source: str) -> str:
    """Extract just the AST structure without values."""
    try:
        tree = ast.parse(source)
        
        def structure_only(node, depth=0):
            lines = [" " * depth + node.__class__.__name__]
            for child in ast.iter_child_nodes(node):
                lines.extend(structure_only(child, depth + 1))
            return lines
        
        return "\n".join(structure_only(tree))
    except SyntaxError:
        return ""


# ---------------------------------------------------------------------------
# Code Analysis Functions
# ---------------------------------------------------------------------------
def extract_functions(source: str, filepath: str, repo: str) -> List[FunctionEmbedding]:
    """Extract detailed function information for embedding."""
    functions = []
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Get function signature
                args = []
                for arg in node.args.args:
                    arg_str = arg.arg
                    if arg.annotation:
                        try:
                            arg_str += f": {ast.unparse(arg.annotation)}"
                        except:
                            pass
                    args.append(arg_str)
                
                returns = ""
                if node.returns:
                    try:
                        returns = f" -> {ast.unparse(node.returns)}"
                    except:
                        pass
                
                signature = f"def {node.name}({', '.join(args)}){returns}"
                
                # Get function body
                try:
                    body_source = ast.get_source_segment(source, node)
                    if body_source:
                        body_hash = hashlib.md5(body_source.encode()).hexdigest()
                        normalized = normalize_ast(body_source)
                    else:
                        body_hash = ""
                        normalized = ""
                except:
                    body_hash = ""
                    normalized = ""
                
                # Get docstring
                docstring = ast.get_docstring(node) or ""
                
                # Get decorators
                decorators = []
                for dec in node.decorator_list:
                    try:
                        decorators.append(ast.unparse(dec))
                    except:
                        pass
                
                # Get function calls
                calls = []
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Name):
                            calls.append(child.func.id)
                        elif isinstance(child.func, ast.Attribute):
                            calls.append(child.func.attr)
                
                # Estimate complexity (simplified cyclomatic)
                complexity = 1
                for child in ast.walk(node):
                    if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler, 
                                         ast.With, ast.Assert, ast.comprehension)):
                        complexity += 1
                    elif isinstance(child, ast.BoolOp):
                        complexity += len(child.values) - 1
                
                line_count = (node.end_lineno - node.lineno + 1) if hasattr(node, 'end_lineno') else 1
                
                func = FunctionEmbedding(
                    name=node.name,
                    repo=repo,
                    file_path=filepath,
                    signature=signature,
                    body_hash=body_hash,
                    normalized_body=normalized[:2000],  # Truncate for embedding
                    docstring=docstring[:500],
                    complexity=complexity,
                    calls=calls[:20],
                    decorators=decorators,
                    is_async=isinstance(node, ast.AsyncFunctionDef),
                    line_count=line_count,
                )
                
                # Build embedding text
                func.embedding_text = build_function_embedding_text(func)
                functions.append(func)
                
    except SyntaxError:
        pass
    
    return functions


def build_function_embedding_text(func: FunctionEmbedding) -> str:
    """Build rich embedding text for a function."""
    parts = [
        f"Function: {func.signature}",
        f"Decorators: {', '.join(func.decorators)}" if func.decorators else "",
        f"Calls: {', '.join(func.calls)}" if func.calls else "",
        f"Complexity: {func.complexity}, Lines: {func.line_count}",
        f"Async: {func.is_async}",
    ]
    
    if func.docstring:
        parts.append(f"Documentation: {func.docstring}")
    
    # Add normalized structure
    if func.normalized_body:
        parts.append(f"Structure:\n{func.normalized_body[:1000]}")
    
    return "\n".join(p for p in parts if p)


def extract_classes(source: str, filepath: str, repo: str) -> List[ClassEmbedding]:
    """Extract detailed class information for embedding."""
    classes = []
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Get base classes
                bases = []
                for base in node.bases:
                    try:
                        bases.append(ast.unparse(base))
                    except:
                        pass
                
                # Get methods
                methods = []
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.append(item.name)
                
                # Get class attributes
                attributes = []
                for item in node.body:
                    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        attr_str = item.target.id
                        if item.annotation:
                            try:
                                attr_str += f": {ast.unparse(item.annotation)}"
                            except:
                                pass
                        attributes.append(attr_str)
                    elif isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name):
                                attributes.append(target.id)
                
                # Get decorators
                decorators = []
                for dec in node.decorator_list:
                    try:
                        decorators.append(ast.unparse(dec))
                    except:
                        pass
                
                docstring = ast.get_docstring(node) or ""
                line_count = (node.end_lineno - node.lineno + 1) if hasattr(node, 'end_lineno') else 1
                
                cls = ClassEmbedding(
                    name=node.name,
                    repo=repo,
                    file_path=filepath,
                    bases=bases,
                    methods=methods,
                    attributes=attributes,
                    docstring=docstring[:500],
                    decorators=decorators,
                    line_count=line_count,
                    method_count=len(methods),
                )
                
                # Build embedding text
                cls.embedding_text = build_class_embedding_text(cls)
                classes.append(cls)
                
    except SyntaxError:
        pass
    
    return classes


def build_class_embedding_text(cls: ClassEmbedding) -> str:
    """Build rich embedding text for a class."""
    parts = [
        f"Class: {cls.name}",
        f"Inherits: {', '.join(cls.bases)}" if cls.bases else "Inherits: object",
        f"Decorators: {', '.join(cls.decorators)}" if cls.decorators else "",
        f"Attributes: {', '.join(cls.attributes)}" if cls.attributes else "",
        f"Methods: {', '.join(cls.methods)}" if cls.methods else "",
        f"Lines: {cls.line_count}, Methods: {cls.method_count}",
    ]
    
    if cls.docstring:
        parts.append(f"Documentation: {cls.docstring}")
    
    return "\n".join(p for p in parts if p)


def extract_file_info(source: str, filepath: str, repo: str) -> FileEmbedding:
    """Extract file-level information for embedding."""
    imports = []
    functions = []
    classes = []
    top_level = []
    
    try:
        tree = ast.parse(source)
        
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.Expr)):
                try:
                    top_level.append(ast.unparse(node)[:100])
                except:
                    pass
        
        docstring = ast.get_docstring(tree) or ""
    except SyntaxError:
        docstring = ""
    
    lines = source.splitlines()
    code_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
    
    file_emb = FileEmbedding(
        path=filepath,
        repo=repo,
        imports=imports,
        functions=functions,
        classes=classes,
        top_level_code="\n".join(top_level[:10]),
        docstring=docstring[:300],
        line_count=len(lines),
        code_line_count=len(code_lines),
    )
    
    file_emb.embedding_text = build_file_embedding_text(file_emb)
    return file_emb


def build_file_embedding_text(file: FileEmbedding) -> str:
    """Build rich embedding text for a file."""
    parts = [
        f"File: {file.path}",
        f"Imports: {', '.join(file.imports[:30])}",
        f"Functions: {', '.join(file.functions)}",
        f"Classes: {', '.join(file.classes)}",
        f"Lines: {file.line_count}, Code lines: {file.code_line_count}",
    ]
    
    if file.docstring:
        parts.append(f"Module documentation: {file.docstring}")
    
    if file.top_level_code:
        parts.append(f"Top-level code:\n{file.top_level_code}")
    
    return "\n".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# API and Pattern Detection
# ---------------------------------------------------------------------------
def detect_api_patterns(source: str) -> Dict[str, Any]:
    """Detect API patterns in FastAPI/Flask code."""
    patterns = {
        "endpoints": [],
        "methods": Counter(),
        "path_params": [],
        "query_params": [],
        "response_models": [],
        "dependencies": [],
        "middleware": [],
    }
    
    # FastAPI patterns
    endpoint_pattern = re.compile(r'@(?:app|router)\.(get|post|put|patch|delete|options|head)\s*\(\s*["\']([^"\']+)["\']')
    for match in endpoint_pattern.finditer(source):
        method, path = match.groups()
        patterns["endpoints"].append(f"{method.upper()} {path}")
        patterns["methods"][method.upper()] += 1
        
        # Detect path parameters
        path_params = re.findall(r'\{([^}]+)\}', path)
        patterns["path_params"].extend(path_params)
    
    # Response models
    response_model_pattern = re.compile(r'response_model\s*=\s*(\w+)')
    for match in response_model_pattern.finditer(source):
        patterns["response_models"].append(match.group(1))
    
    # Depends
    depends_pattern = re.compile(r'Depends\s*\(\s*(\w+)')
    for match in depends_pattern.finditer(source):
        patterns["dependencies"].append(match.group(1))
    
    return patterns


def detect_db_patterns(source: str) -> Dict[str, Any]:
    """Detect database operation patterns."""
    patterns = {
        "operations": [],
        "models": [],
        "session_usage": False,
        "orm_style": "unknown",
    }
    
    # SQLModel/SQLAlchemy patterns
    if "session.add" in source or "session.commit" in source:
        patterns["session_usage"] = True
        patterns["orm_style"] = "sqlmodel/sqlalchemy"
    
    if "session.refresh" in source:
        patterns["operations"].append("refresh_after_commit")
    
    # CRUD operations
    crud_patterns = [
        (r'session\.add\s*\(', "create"),
        (r'session\.query\s*\(|select\s*\(', "read"),
        (r'session\.commit\s*\(', "update"),
        (r'session\.delete\s*\(', "delete"),
        (r'session\.execute\s*\(', "execute"),
    ]
    
    for pattern, op in crud_patterns:
        if re.search(pattern, source):
            patterns["operations"].append(op)
    
    # Model definitions
    model_pattern = re.compile(r'class\s+(\w+)\s*\([^)]*(?:SQLModel|Base|Model)[^)]*\)')
    for match in model_pattern.finditer(source):
        patterns["models"].append(match.group(1))
    
    return patterns


# ---------------------------------------------------------------------------
# Winnowing Fingerprinting
# ---------------------------------------------------------------------------
def hash_kgram(kgram: str) -> int:
    """Hash a k-gram to an integer."""
    return int(hashlib.md5(kgram.encode()).hexdigest()[:8], 16)


def get_winnowing_fingerprints(text: str, k: int = KGRAM_SIZE, w: int = WINDOW_SIZE) -> Set[int]:
    """Generate winnowing fingerprints for text."""
    # Normalize: remove whitespace and lowercase
    text = re.sub(r'\s+', '', text.lower())
    
    if len(text) < k:
        return set()
    
    # Generate k-grams
    kgrams = [text[i:i+k] for i in range(len(text) - k + 1)]
    hashes = [hash_kgram(kg) for kg in kgrams]
    
    # Winnowing: select minimum from each window
    fingerprints = set()
    for i in range(len(hashes) - w + 1):
        window = hashes[i:i+w]
        fingerprints.add(min(window))
    
    return fingerprints


def fingerprint_similarity(fp1: Set[int], fp2: Set[int]) -> float:
    """Calculate Jaccard similarity between fingerprint sets."""
    if not fp1 or not fp2:
        return 0.0
    intersection = len(fp1 & fp2)
    union = len(fp1 | fp2)
    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Multi-Level Analysis
# ---------------------------------------------------------------------------
def analyze_repository(repo_path: Path, repo_name: str) -> ProjectEmbedding:
    """Perform deep multi-level analysis of a repository."""
    all_functions = []
    all_classes = []
    all_files = []
    all_imports = set()
    all_api_patterns = defaultdict(list)
    all_db_patterns = defaultdict(list)
    total_lines = 0
    
    python_files = []
    for py_file in repo_path.rglob("*.py"):
        if any(skip in py_file.parts for skip in SKIP_DIRS):
            continue
        if "test" in py_file.name.lower():
            continue
        python_files.append(py_file)
    
    for py_file in python_files:
        try:
            source = py_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        
        rel_path = str(py_file.relative_to(repo_path))
        total_lines += len(source.splitlines())
        
        # Extract functions
        funcs = extract_functions(source, rel_path, repo_name)
        all_functions.extend(funcs)
        
        # Extract classes
        classes = extract_classes(source, rel_path, repo_name)
        all_classes.extend(classes)
        
        # Extract file info
        file_info = extract_file_info(source, rel_path, repo_name)
        all_files.append(file_info)
        all_imports.update(file_info.imports)
        
        # Detect patterns
        api_patterns = detect_api_patterns(source)
        for key, value in api_patterns.items():
            if isinstance(value, list):
                all_api_patterns[key].extend(value)
            elif isinstance(value, Counter):
                # Convert Counter to list of items
                all_api_patterns[key].extend([f"{k}:{v}" for k, v in value.items()])
        
        db_patterns = detect_db_patterns(source)
        for key, value in db_patterns.items():
            if isinstance(value, list):
                all_db_patterns[key].extend(value)
    
    # Detect frameworks
    frameworks = []
    if "fastapi" in all_imports or "FastAPI" in str(all_imports):
        frameworks.append("FastAPI")
    if "sqlmodel" in all_imports:
        frameworks.append("SQLModel")
    if "sqlalchemy" in all_imports:
        frameworks.append("SQLAlchemy")
    if "pydantic" in all_imports:
        frameworks.append("Pydantic")
    if "uvicorn" in all_imports:
        frameworks.append("Uvicorn")
    
    # Detect architecture pattern
    file_names = [f.path for f in all_files]
    if any("routes" in f or "routers" in f for f in file_names):
        arch_pattern = "router-based"
    elif any("views" in f for f in file_names):
        arch_pattern = "mvc"
    elif len(all_files) <= 3:
        arch_pattern = "single-file"
    else:
        arch_pattern = "modular"
    
    project = ProjectEmbedding(
        name=repo_name,
        path=str(repo_path),
        files=[f.path for f in all_files],
        all_functions=[f.name for f in all_functions],
        all_classes=[c.name for c in all_classes],
        all_imports=all_imports,
        frameworks=frameworks,
        api_endpoints=all_api_patterns.get("endpoints", []),
        db_operations=list(set(all_db_patterns.get("operations", []))),
        architecture_pattern=arch_pattern,
        total_lines=total_lines,
        total_functions=len(all_functions),
        total_classes=len(all_classes),
    )
    
    project.embedding_text = build_project_embedding_text(project, all_functions, all_classes, all_files)
    
    return project


def build_project_embedding_text(
    project: ProjectEmbedding,
    functions: List[FunctionEmbedding],
    classes: List[ClassEmbedding],
    files: List[FileEmbedding],
) -> str:
    """Build comprehensive embedding text for a project."""
    parts = [
        f"# Project: {project.name}",
        f"Architecture: {project.architecture_pattern}",
        f"Frameworks: {', '.join(project.frameworks)}",
        f"Files: {len(project.files)}, Functions: {project.total_functions}, Classes: {project.total_classes}",
        f"Total lines: {project.total_lines}",
        "",
        "## API Endpoints",
        "\n".join(project.api_endpoints[:30]),
        "",
        "## Database Operations",
        ", ".join(project.db_operations),
        "",
        "## Key Imports",
        ", ".join(sorted(project.all_imports)[:40]),
        "",
        "## Functions",
    ]
    
    # Add function signatures (most important for similarity)
    for func in functions[:50]:
        parts.append(f"- {func.signature}")
        if func.docstring:
            parts.append(f"  Doc: {func.docstring[:100]}")
    
    parts.append("\n## Classes")
    for cls in classes[:30]:
        parts.append(f"- class {cls.name}({', '.join(cls.bases)})")
        parts.append(f"  Methods: {', '.join(cls.methods)}")
        if cls.docstring:
            parts.append(f"  Doc: {cls.docstring[:100]}")
    
    parts.append("\n## File Structure")
    for f in files[:20]:
        parts.append(f"- {f.path}: {f.code_line_count} lines, funcs: {', '.join(f.functions[:5])}")
    
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Multi-Level Embedding Generation
# ---------------------------------------------------------------------------
def compute_multi_level_embeddings(
    project: ProjectEmbedding,
    functions: List[FunctionEmbedding],
    classes: List[ClassEmbedding],
    files: List[FileEmbedding],
    embedder: TextEmbedding,
) -> Dict[str, np.ndarray]:
    """Compute embeddings at multiple levels and aggregate."""
    
    # Function-level embeddings
    func_embeddings = []
    if functions:
        func_texts = [f.embedding_text for f in functions if f.embedding_text]
        if func_texts:
            func_embeddings = list(embedder.embed(func_texts))
    
    # Class-level embeddings
    class_embeddings = []
    if classes:
        class_texts = [c.embedding_text for c in classes if c.embedding_text]
        if class_texts:
            class_embeddings = list(embedder.embed(class_texts))
    
    # File-level embeddings
    file_embeddings = []
    if files:
        file_texts = [f.embedding_text for f in files if f.embedding_text]
        if file_texts:
            file_embeddings = list(embedder.embed(file_texts))
    
    # Project-level embedding
    project_embedding = list(embedder.embed([project.embedding_text]))[0]
    
    # Aggregate function embeddings (mean pooling)
    if func_embeddings:
        func_aggregate = np.mean(func_embeddings, axis=0)
    else:
        func_aggregate = np.zeros_like(project_embedding)
    
    # Aggregate class embeddings
    if class_embeddings:
        class_aggregate = np.mean(class_embeddings, axis=0)
    else:
        class_aggregate = np.zeros_like(project_embedding)
    
    # Aggregate file embeddings
    if file_embeddings:
        file_aggregate = np.mean(file_embeddings, axis=0)
    else:
        file_aggregate = np.zeros_like(project_embedding)
    
    return {
        "project": project_embedding,
        "functions": func_aggregate,
        "classes": class_aggregate,
        "files": file_aggregate,
        "function_count": len(func_embeddings),
        "class_count": len(class_embeddings),
        "file_count": len(file_embeddings),
    }


def compute_combined_embedding(embeddings: Dict[str, np.ndarray], weights: Dict[str, float] = None) -> np.ndarray:
    """Combine multi-level embeddings with weights."""
    if weights is None:
        weights = {
            "project": 0.25,
            "functions": 0.35,  # Functions are most important for similarity
            "classes": 0.25,
            "files": 0.15,
        }
    
    combined = np.zeros_like(embeddings["project"])
    for key, weight in weights.items():
        if key in embeddings and isinstance(embeddings[key], np.ndarray):
            combined += weight * embeddings[key]
    
    # Normalize
    norm = np.linalg.norm(combined)
    if norm > 0:
        combined = combined / norm
    
    return combined


# ---------------------------------------------------------------------------
# Similarity Computation
# ---------------------------------------------------------------------------
def compute_deep_similarity(
    emb1: Dict[str, np.ndarray],
    emb2: Dict[str, np.ndarray],
    fp1: Set[int],
    fp2: Set[int],
) -> Dict[str, float]:
    """Compute similarity at multiple levels."""
    
    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    project_sim = cosine_sim(emb1["project"], emb2["project"])
    function_sim = cosine_sim(emb1["functions"], emb2["functions"])
    class_sim = cosine_sim(emb1["classes"], emb2["classes"])
    file_sim = cosine_sim(emb1["files"], emb2["files"])
    fingerprint_sim = fingerprint_similarity(fp1, fp2)
    
    # Combined with weighted average
    combined = (
        0.15 * project_sim +
        0.35 * function_sim +
        0.20 * class_sim +
        0.10 * file_sim +
        0.20 * fingerprint_sim
    )
    
    return {
        "combined": combined,
        "project": project_sim,
        "functions": function_sim,
        "classes": class_sim,
        "files": file_sim,
        "fingerprint": fingerprint_sim,
    }


# ---------------------------------------------------------------------------
# Main Analysis Pipeline
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Smart Multi-Level Code Embedding Analysis")
    parser.add_argument("--root", default="work", help="Root directory containing repos")
    parser.add_argument("--repo-subdir", default="repo", help="Subdirectory within each folder containing the repo")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--model", default=os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    parser.add_argument("--qdrant-host", default=os.environ.get("QDRANT_HOST", "localhost"))
    parser.add_argument("--qdrant-port", type=int, default=int(os.environ.get("QDRANT_PORT", 6333)))
    parser.add_argument("--reset", action="store_true", help="Reset Qdrant collections")
    args = parser.parse_args()
    
    root = Path(args.root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    
    # Find repositories
    repos = []
    for item in sorted(root.iterdir()):
        if item.is_dir():
            repo_path = item / args.repo_subdir if args.repo_subdir else item
            if repo_path.is_dir():
                repos.append((item.name, repo_path))
    
    print(f"🔍 Found {len(repos)} repositories to analyze")
    
    # Initialize embedder
    print(f"\n🤖 Loading embedding model: {args.model}")
    embedder = TextEmbedding(model_name=args.model)
    
    # Get embedding dimension
    size = getattr(embedder, 'embedding_size', None)
    if size is None:
        size = len(next(embedder.embed(["probe"])))
    print(f"   Embedding dimension: {size}")
    
    # Initialize Qdrant
    print(f"\n📦 Connecting to Qdrant at {args.qdrant_host}:{args.qdrant_port}")
    qdrant = QdrantClient(host=args.qdrant_host, port=args.qdrant_port)
    
    if args.reset:
        for collection in [COLLECTION_COMBINED]:
            try:
                qdrant.delete_collection(collection)
                print(f"   Deleted collection: {collection}")
            except:
                pass
    
    # Create collection
    try:
        qdrant.create_collection(
            collection_name=COLLECTION_COMBINED,
            vectors_config=qmodels.VectorParams(size=size, distance=qmodels.Distance.COSINE),
        )
        print(f"   Created collection: {COLLECTION_COMBINED}")
    except:
        print(f"   Collection {COLLECTION_COMBINED} already exists")
    
    # Analyze all repositories
    print("\n📊 Analyzing repositories at multiple levels...")
    analyses = []
    all_embeddings = {}
    all_fingerprints = {}
    all_functions = {}
    all_classes = {}
    all_files = {}
    
    for repo_name, repo_path in tqdm(repos, desc="Analyzing"):
        # Deep analysis
        project = analyze_repository(repo_path, repo_name)
        analyses.append(project)
        
        # Extract components for embedding
        functions = []
        classes = []
        files = []
        
        for py_file in repo_path.rglob("*.py"):
            if any(skip in py_file.parts for skip in SKIP_DIRS):
                continue
            if "test" in py_file.name.lower():
                continue
            try:
                source = py_file.read_text(encoding="utf-8", errors="ignore")
                rel_path = str(py_file.relative_to(repo_path))
                functions.extend(extract_functions(source, rel_path, repo_name))
                classes.extend(extract_classes(source, rel_path, repo_name))
                files.append(extract_file_info(source, rel_path, repo_name))
            except Exception:
                continue
        
        all_functions[repo_name] = functions
        all_classes[repo_name] = classes
        all_files[repo_name] = files
        
        # Compute fingerprints
        all_code = "\n".join(f.embedding_text for f in functions)
        all_fingerprints[repo_name] = get_winnowing_fingerprints(all_code)
    
    # Compute embeddings
    print("\n🔤 Computing multi-level embeddings...")
    for project in tqdm(analyses, desc="Embedding"):
        embs = compute_multi_level_embeddings(
            project,
            all_functions[project.name],
            all_classes[project.name],
            all_files[project.name],
            embedder,
        )
        all_embeddings[project.name] = embs
    
    # Store in Qdrant
    print("\n📤 Storing embeddings in Qdrant...")
    points = []
    for i, project in enumerate(analyses):
        combined = compute_combined_embedding(all_embeddings[project.name])
        points.append(qmodels.PointStruct(
            id=i,
            vector=combined.tolist(),
            payload={
                "name": project.name,
                "frameworks": project.frameworks,
                "total_functions": project.total_functions,
                "total_classes": project.total_classes,
                "total_lines": project.total_lines,
                "architecture": project.architecture_pattern,
                "endpoints": project.api_endpoints[:20],
            }
        ))
    
    qdrant.upsert(collection_name=COLLECTION_COMBINED, points=points)
    print(f"   Stored {len(points)} project embeddings")
    
    # Compute pairwise similarities
    print("\n🔗 Computing pairwise similarities...")
    similarity_matrix = {}
    names = [p.name for p in analyses]
    
    for i, name1 in enumerate(tqdm(names, desc="Comparing")):
        for j, name2 in enumerate(names):
            if i >= j:
                continue
            
            sims = compute_deep_similarity(
                all_embeddings[name1],
                all_embeddings[name2],
                all_fingerprints[name1],
                all_fingerprints[name2],
            )
            similarity_matrix[(name1, name2)] = sims
    
    # Save results
    print("\n💾 Saving results...")
    
    # CSV with all similarity scores
    import csv
    rows = []
    for (r1, r2), sims in sorted(similarity_matrix.items(), key=lambda x: -x[1]['combined']):
        rows.append({
            'repo1': r1,
            'repo2': r2,
            'combined': round(sims['combined'], 4),
            'project_level': round(sims['project'], 4),
            'function_level': round(sims['functions'], 4),
            'class_level': round(sims['classes'], 4),
            'file_level': round(sims['files'], 4),
            'fingerprint': round(sims['fingerprint'], 4),
        })
    
    with open(output / "smart_similarity.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    
    # JSON with detailed analysis
    analysis_data = []
    for project in analyses:
        analysis_data.append({
            "name": project.name,
            "frameworks": project.frameworks,
            "architecture": project.architecture_pattern,
            "total_lines": project.total_lines,
            "total_functions": project.total_functions,
            "total_classes": project.total_classes,
            "api_endpoints": project.api_endpoints[:50],
            "db_operations": project.db_operations,
            "files": project.files,
            "function_count": all_embeddings[project.name]["function_count"],
            "class_count": all_embeddings[project.name]["class_count"],
        })
    
    with open(output / "smart_analysis.json", "w") as f:
        json.dump(analysis_data, f, indent=2, default=str)
    
    # Summary
    print("\n" + "=" * 60)
    print("📈 SMART CODE EMBEDDING ANALYSIS COMPLETE")
    print("=" * 60)
    
    # Top similar pairs
    print("\n🔝 Top 10 Most Similar Repository Pairs:")
    sorted_pairs = sorted(similarity_matrix.items(), key=lambda x: -x[1]['combined'])[:10]
    for (r1, r2), sims in sorted_pairs:
        print(f"  {r1} <-> {r2}")
        print(f"    Combined: {sims['combined']:.3f} | Functions: {sims['functions']:.3f} | "
              f"Classes: {sims['classes']:.3f} | Fingerprint: {sims['fingerprint']:.3f}")
    
    # Statistics
    all_combined = [s['combined'] for s in similarity_matrix.values()]
    all_functions_sim = [s['functions'] for s in similarity_matrix.values()]
    all_fingerprints_sim = [s['fingerprint'] for s in similarity_matrix.values()]
    
    print(f"\n📊 Similarity Statistics:")
    print(f"  Combined:    mean={statistics.mean(all_combined):.3f}, max={max(all_combined):.3f}, min={min(all_combined):.3f}")
    print(f"  Functions:   mean={statistics.mean(all_functions_sim):.3f}, max={max(all_functions_sim):.3f}")
    print(f"  Fingerprint: mean={statistics.mean(all_fingerprints_sim):.3f}, max={max(all_fingerprints_sim):.3f}")
    
    print(f"\n📁 Results saved to: {output}/")
    print(f"  - smart_similarity.csv")
    print(f"  - smart_analysis.json")


if __name__ == "__main__":
    main()
