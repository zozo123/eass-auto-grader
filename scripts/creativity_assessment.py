#!/usr/bin/env python3
"""
Originality & Creativity Assessment for Student Projects

This tool evaluates how much effort students put into going BEYOND
the minimum requirements. It's not about plagiarism - it's about
assessing creative investment and learning depth.

Assessment Dimensions:
1. Domain Creativity - Did they pick an interesting/unique topic?
2. Feature Richness - Did they add features beyond basic CRUD?
3. Code Quality - Clean architecture, error handling, documentation
4. Technical Depth - Advanced patterns, async, testing depth
5. Extra Mile - CLI tools, documentation, deployment configs
"""

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


@dataclass
class CreativityScore:
    """Creativity assessment for a single repo."""
    name: str
    domain: str
    
    # Scores out of 10
    domain_creativity: int = 0
    feature_richness: int = 0
    code_quality: int = 0
    technical_depth: int = 0
    extra_mile: int = 0
    
    # Evidence
    domain_notes: str = ""
    features_found: List[str] = field(default_factory=list)
    quality_notes: List[str] = field(default_factory=list)
    technical_notes: List[str] = field(default_factory=list)
    extra_notes: List[str] = field(default_factory=list)
    
    @property
    def total_score(self) -> int:
        return (self.domain_creativity + self.feature_richness + 
                self.code_quality + self.technical_depth + self.extra_mile)
    
    @property
    def grade(self) -> str:
        total = self.total_score
        if total >= 40: return "A - Exceptional"
        if total >= 32: return "B - Above Average"
        if total >= 24: return "C - Meets Expectations"
        if total >= 16: return "D - Below Average"
        return "F - Minimal Effort"


# Domain creativity scoring
COMMON_DOMAINS = {"movie", "book", "task", "todo"}  # Very common = lower creativity
MODERATE_DOMAINS = {"recipe", "workout", "expense", "game", "appointment"}
CREATIVE_DOMAINS = {"wardrobe", "budget", "device", "travel", "gym", "calorie", "football", "kitchen"}


def analyze_repo(repo_path: Path, name: str) -> CreativityScore:
    """Analyze a single repository for creativity."""
    
    score = CreativityScore(name=name, domain="")
    
    # Detect domain from name and files
    name_lower = name.lower()
    for domain in CREATIVE_DOMAINS:
        if domain in name_lower:
            score.domain = domain
            score.domain_creativity = 8
            score.domain_notes = f"Creative/unique domain: {domain}"
            break
    if not score.domain:
        for domain in MODERATE_DOMAINS:
            if domain in name_lower:
                score.domain = domain
                score.domain_creativity = 5
                score.domain_notes = f"Moderate domain choice: {domain}"
                break
    if not score.domain:
        for domain in COMMON_DOMAINS:
            if domain in name_lower:
                score.domain = domain
                score.domain_creativity = 3
                score.domain_notes = f"Very common domain: {domain} (many students chose this)"
                break
    if not score.domain:
        score.domain = "unknown"
        score.domain_creativity = 5
        score.domain_notes = "Domain not easily categorized"
    
    # Gather all Python code
    all_code = ""
    py_files = list(repo_path.rglob("*.py"))
    for f in py_files:
        try:
            all_code += f.read_text() + "\n"
        except:
            pass
    
    # Feature richness analysis
    features = []
    
    if re.search(r'skip.*limit|offset|page', all_code, re.IGNORECASE):
        features.append("Pagination")
    if re.search(r'search|filter(?!ing)|query(?!_)', all_code, re.IGNORECASE):
        features.append("Search/Filter")
    if re.search(r'@field_validator|@model_validator|validator', all_code):
        features.append("Custom Validation")
    if re.search(r'sort|order_by|ordering', all_code, re.IGNORECASE):
        features.append("Sorting")
    if re.search(r'OAuth|JWT|token|Bearer|authenticate', all_code, re.IGNORECASE):
        features.append("Authentication")
    if re.search(r'relationship|ForeignKey|joinedload', all_code, re.IGNORECASE):
        features.append("Database Relationships")
    if re.search(r'BackgroundTask|celery|async.*task', all_code, re.IGNORECASE):
        features.append("Background Tasks")
    if re.search(r'websocket|WebSocket', all_code, re.IGNORECASE):
        features.append("WebSockets")
    if re.search(r'cache|redis|Cache', all_code, re.IGNORECASE):
        features.append("Caching")
    if re.search(r'rate.?limit|throttle', all_code, re.IGNORECASE):
        features.append("Rate Limiting")
    
    score.features_found = features
    # Score: 2 points per unique feature, max 10
    score.feature_richness = min(10, len(features) * 2)
    if not features:
        score.feature_richness = 2  # Basic CRUD gets 2
    
    # Code quality analysis
    quality = []
    
    if re.search(r'"""[\s\S]+?"""', all_code):
        quality.append("Has docstrings")
    if re.search(r'logging\.getLogger|logger\s*=', all_code):
        quality.append("Proper logging")
    if re.search(r'class.*Settings.*BaseSettings', all_code):
        quality.append("Config management")
    if re.search(r'try:[\s\S]+?except', all_code):
        quality.append("Exception handling")
    if re.search(r'HTTPException\(status_code=\d+,\s*detail=', all_code):
        quality.append("Proper HTTP errors")
    if (repo_path / "README.md").exists():
        readme = (repo_path / "README.md").read_text()
        if len(readme) > 500:
            quality.append("Comprehensive README")
    if len([f for f in py_files if "test" in f.name.lower()]) >= 2:
        quality.append("Multiple test files")
    
    score.quality_notes = quality
    score.code_quality = min(10, len(quality) * 2)
    if not quality:
        score.code_quality = 1
    
    # Technical depth analysis
    technical = []
    
    if re.search(r'async\s+def|await\s+', all_code):
        technical.append("Async/await patterns")
    if re.search(r'Depends\(', all_code):
        technical.append("Dependency injection")
    if re.search(r'middleware|Middleware', all_code):
        technical.append("Custom middleware")
    if re.search(r'Repository|repository.*pattern', all_code, re.IGNORECASE):
        technical.append("Repository pattern")
    if re.search(r'Factory|factory.*pattern', all_code, re.IGNORECASE):
        technical.append("Factory pattern")
    if re.search(r'@pytest\.fixture|fixture', all_code):
        technical.append("Test fixtures")
    if re.search(r'parametrize|@pytest\.mark', all_code):
        technical.append("Parametrized tests")
    if re.search(r'migration|alembic', all_code, re.IGNORECASE):
        technical.append("DB migrations")
    
    score.technical_notes = technical
    score.technical_depth = min(10, len(technical) * 2)
    if not technical:
        score.technical_depth = 2
    
    # Extra mile analysis
    extra = []
    
    if re.search(r'typer|click|argparse', all_code):
        extra.append("CLI tool")
    if (repo_path / "docker-compose.yml").exists() or (repo_path / "docker-compose.yaml").exists():
        extra.append("Docker Compose")
    if (repo_path / "Dockerfile").exists():
        extra.append("Dockerfile")
    if (repo_path / ".github").exists():
        extra.append("GitHub Actions/CI")
    if any(repo_path.rglob("*.html")):
        extra.append("Frontend/HTML")
    if (repo_path / "Makefile").exists():
        extra.append("Makefile")
    if re.search(r'seed|populate|init.*data', all_code, re.IGNORECASE):
        extra.append("Data seeding")
    if (repo_path / "render.yaml").exists() or re.search(r'render|heroku|railway', all_code, re.IGNORECASE):
        extra.append("Deployment config")
    
    score.extra_notes = extra
    score.extra_mile = min(10, len(extra) * 2)
    if not extra:
        score.extra_mile = 1
    
    return score


def main():
    parser = argparse.ArgumentParser(description="Assess originality and creativity")
    parser.add_argument("--root", default="work")
    parser.add_argument("--repo-subdir", default="repo")
    parser.add_argument("--output", default="results/creativity_assessment.json")
    args = parser.parse_args()
    
    root = Path(args.root)
    repos = sorted([d for d in root.iterdir() if d.is_dir()])
    
    print("=" * 70)
    print("🎨 ORIGINALITY & CREATIVITY ASSESSMENT")
    print("=" * 70)
    print()
    print("This assessment evaluates creative effort BEYOND minimum requirements.")
    print("It's about learning depth, not about detecting cheating.")
    print()
    
    scores = []
    for repo_dir in repos:
        repo_path = repo_dir / args.repo_subdir
        if not repo_path.exists():
            continue
        
        name = repo_dir.name
        score = analyze_repo(repo_path, name)
        scores.append(score)
    
    # Sort by total score
    scores.sort(key=lambda x: x.total_score, reverse=True)
    
    # Display results
    print("=" * 70)
    print("📊 RANKINGS (by Total Creativity Score)")
    print("=" * 70)
    print()
    print(f"{'Rank':<5} {'Student':<45} {'Domain':<3} {'Feat':<5} {'Qual':<5} {'Tech':<5} {'Extra':<6} {'TOTAL':<6} {'Grade'}")
    print("-" * 100)
    
    for i, s in enumerate(scores, 1):
        short_name = s.name[:42] + "..." if len(s.name) > 45 else s.name
        print(f"{i:<5} {short_name:<45} {s.domain_creativity:<3} {s.feature_richness:<5} {s.code_quality:<5} {s.technical_depth:<5} {s.extra_mile:<6} {s.total_score:<6} {s.grade}")
    
    # Grade distribution
    print()
    print("=" * 70)
    print("📈 GRADE DISTRIBUTION")
    print("=" * 70)
    
    grade_counts = defaultdict(int)
    for s in scores:
        grade_counts[s.grade[0]] += 1  # Just the letter
    
    for grade in ['A', 'B', 'C', 'D', 'F']:
        count = grade_counts.get(grade, 0)
        bar = "█" * count
        print(f"  {grade}: {bar} ({count})")
    
    # Top performers
    print()
    print("=" * 70)
    print("🌟 TOP PERFORMERS (Above Average Creativity)")
    print("=" * 70)
    
    for s in scores[:5]:
        print(f"\n📦 {s.name}")
        print(f"   Total Score: {s.total_score}/50 - {s.grade}")
        print(f"   Domain: {s.domain_notes}")
        if s.features_found:
            print(f"   Features: {', '.join(s.features_found)}")
        if s.technical_notes:
            print(f"   Technical: {', '.join(s.technical_notes)}")
        if s.extra_notes:
            print(f"   Extra Mile: {', '.join(s.extra_notes)}")
    
    # Minimum effort
    print()
    print("=" * 70)
    print("⚠️  MINIMUM EFFORT (Could Do More)")
    print("=" * 70)
    print()
    print("These students met basic requirements but showed limited creative investment:")
    print()
    
    for s in scores[-5:]:
        print(f"  • {s.name}")
        print(f"    Score: {s.total_score}/50 | {s.domain_notes}")
        if not s.features_found:
            print(f"    → No features beyond basic CRUD")
        if not s.extra_notes:
            print(f"    → No extra deliverables (CLI, Docker Compose, etc.)")
        print()
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = []
    for s in scores:
        results.append({
            "name": s.name,
            "domain": s.domain,
            "scores": {
                "domain_creativity": s.domain_creativity,
                "feature_richness": s.feature_richness,
                "code_quality": s.code_quality,
                "technical_depth": s.technical_depth,
                "extra_mile": s.extra_mile,
                "total": s.total_score,
            },
            "grade": s.grade,
            "evidence": {
                "domain_notes": s.domain_notes,
                "features": s.features_found,
                "quality": s.quality_notes,
                "technical": s.technical_notes,
                "extra": s.extra_notes,
            }
        })
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Detailed results saved to {output_path}")
    
    # Summary statistics
    avg_score = sum(s.total_score for s in scores) / len(scores)
    print()
    print("=" * 70)
    print("📊 SUMMARY STATISTICS")
    print("=" * 70)
    print(f"  Average creativity score: {avg_score:.1f}/50")
    print(f"  Highest score: {scores[0].total_score}/50 ({scores[0].name.split('_')[0]})")
    print(f"  Lowest score: {scores[-1].total_score}/50 ({scores[-1].name.split('_')[0]})")
    print()
    
    # Common domains
    domain_counts = defaultdict(int)
    for s in scores:
        domain_counts[s.domain] += 1
    
    print("  Domain choices:")
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        pct = count / len(scores) * 100
        print(f"    {domain}: {count} students ({pct:.0f}%)")


if __name__ == "__main__":
    main()
