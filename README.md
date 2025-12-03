# Multi-Repo Code Analysis & Similarity Detection

A comprehensive tool for analyzing, comparing, and evaluating code similarities across multiple repositories. Originally designed for academic project evaluation but generic enough for any multi-repository comparison task.

## Features

- **Vector-based similarity analysis** using Qdrant and local embeddings (no API keys needed)
- **AST-based code analysis** for structural comparison
- **Multiple similarity metrics**: vector, Jaccard, n-gram/winnowing, structural
- **Deep multi-dimensional analysis** covering 17 aspects of code
- **Clustering and visualization** with heatmaps and dendrograms
- **Responsible similarity analysis** that accounts for expected patterns
- **Creativity/originality assessment** for fair evaluation

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Docker (for Qdrant vector database)

## Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd eass-auto-grader

# Install dependencies using uv
uv sync

# Or using pip
pip install -e .
```

### 2. Start Qdrant

```bash
docker compose -f docker-compose.qdrant.yml up -d
```

### 3. Prepare Your Data

Create a `submission.csv` file with the following format:

```csv
name,email,repo_url
John Doe,john@example.com,https://github.com/johndoe/project
Jane Smith,jane@example.com,https://github.com/janesmith/project
```

See `submission.csv.example` for reference.

### 4. Clone Repositories

```bash
make clone
```

### 5. Run Analysis

```bash
# Compare all repositories
make compare

# Generate deep analysis with 17 dimensions
make deep-analysis

# Visualize similarities with clustering
make visualize-deep

# Run responsible analysis (accounts for expected patterns)
make responsible-analysis

# Assess creativity/originality
make creativity
```

## Analysis Scripts

| Script | Description |
|--------|-------------|
| `run_evaluation.py` | Core evaluation runner |
| `run_evaluation_parallel.py` | Parallel evaluation for multiple repos |
| `compare_repos.py` | Basic similarity comparison |
| `compare_repos_v2.py` | Advanced multi-metric comparison |
| `deep_analysis.py` | 17-dimension comprehensive analysis |
| `visualize_similarity.py` | Basic visualization |
| `visualize_deep_analysis.py` | Clustered heatmaps and dendrograms |
| `detect_plagiarism.py` | Plagiarism detection (use responsibly!) |
| `responsible_similarity_analysis.py` | Fair analysis with context |
| `creativity_assessment.py` | Originality and effort scoring |
| `generate_plagiarism_html.py` | HTML plagiarism report |
| `generate_combined_report.py` | Combined analysis report |
| `show_pairs.py` | Display similarity pairs |

## Similarity Metrics

The tool uses multiple complementary metrics:

1. **Vector Similarity**: Semantic similarity using embeddings
2. **Jaccard Similarity**: Token-based overlap
3. **N-gram/Winnowing**: Code fingerprinting for copy detection
4. **Structural Similarity**: AST-based structure comparison

## Deep Analysis Dimensions

The deep analysis covers 17 aspects grouped into categories:

### Structural (4 dimensions)
- File organization patterns
- Class/function architecture
- Import structure
- Configuration patterns

### Semantic (3 dimensions)
- Variable naming conventions
- Comment style and content
- String literals and messages

### Behavioral (4 dimensions)
- API endpoint patterns
- Database operations
- Error handling approaches
- CRUD implementation style

### Style (3 dimensions)
- Code formatting
- Type hints usage
- Documentation patterns

### Testing (2 dimensions)
- Test structure and coverage
- Mock patterns

### Documentation (1 dimension)
- README and docs content

## Responsible Use

⚠️ **Important**: High similarity scores don't always indicate plagiarism!

Common legitimate reasons for high similarity:
- Students following the same tutorials
- Using standard library patterns
- Same assignment requirements
- Common framework boilerplate

Always:
1. Check if subjects are from the same course/assignment
2. Look for shared tutorials or reference materials
3. Consider framework-specific patterns
4. Review the actual code in context

## Configuration

### Environment Variables

```bash
# Optional: Customize Qdrant connection
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

### Makefile Targets

```bash
make help              # Show all available targets
make clone             # Clone all repositories
make build             # Build Docker images
make test              # Run tests
make compare           # Run comparison analysis
make deep-analysis     # 17-dimension analysis
make responsible       # Context-aware analysis
make creativity        # Originality assessment
make qdrant-up         # Start Qdrant container
make qdrant-down       # Stop Qdrant container
```

## Output Files

Results are saved in the `results/` directory:
- `similarity_*.json` - Raw similarity scores
- `deep_analysis_results.json` - 17-dimension analysis
- `*_heatmap.png` - Visualization plots
- `*.html` - Interactive reports

## Project Structure

```
.
├── scripts/                 # Analysis scripts
│   ├── run_evaluation.py
│   ├── compare_repos_v2.py
│   ├── deep_analysis.py
│   └── ...
├── examples/                # Example configurations
├── work/                    # Cloned repositories (gitignored)
├── results/                 # Analysis outputs (gitignored)
├── logs/                    # Log files (gitignored)
├── submission.csv           # Your data file (gitignored)
├── submission.csv.example   # Example format
├── docker-compose.qdrant.yml
├── Makefile
├── pyproject.toml
└── README.md
```

## Extending for Your Use Case

1. **Custom Assignment Context**: Modify `responsible_similarity_analysis.py` to add expected patterns for your specific framework/tutorials
2. **Custom Metrics**: Add new dimension categories in `deep_analysis.py`
3. **Custom Scoring**: Adjust weights in `creativity_assessment.py`
4. **Custom Reports**: Modify HTML generators for your branding

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
