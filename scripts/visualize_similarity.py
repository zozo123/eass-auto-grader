#!/usr/bin/env python3
"""Visualize repository similarity data.

Generates:
- Heatmap of similarity matrix
- Network graph of similar repositories
- Bar charts of framework usage
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


def load_similarity_matrix(path: Path) -> pd.DataFrame:
    """Load similarity matrix from CSV."""
    df = pd.read_csv(path, index_col=0)
    return df.astype(float)


def plot_heatmap(matrix: pd.DataFrame, output: Path, title: str = "Repository Similarity Matrix") -> None:
    """Plot similarity matrix as heatmap."""
    n = len(matrix)
    fig_size = max(10, n * 0.4)
    
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    
    # Custom colormap (white -> blue)
    colors = ["#ffffff", "#e6f2ff", "#99ccff", "#3399ff", "#0066cc", "#003366"]
    cmap = LinearSegmentedColormap.from_list("similarity", colors)
    
    # Plot heatmap
    im = ax.imshow(matrix.values, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Similarity Score", rotation=270, labelpad=20)
    
    # Set ticks
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    
    # Shorten names for readability
    short_names = [name[:20] + "..." if len(name) > 23 else name for name in matrix.columns]
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short_names, fontsize=8)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap to {output}")


def plot_top_pairs(csv_path: Path, output: Path, top_n: int = 20) -> None:
    """Plot top similar pairs as horizontal bar chart."""
    df = pd.read_csv(csv_path)
    
    # Get unique pairs with highest combined scores
    seen = set()
    pairs = []
    for _, row in df.sort_values("combined_score", ascending=False).iterrows():
        key = tuple(sorted([row["repo"], row["match"]]))
        if key not in seen:
            seen.add(key)
            pairs.append({
                "pair": f"{row['repo'][:15]}... <-> {row['match'][:15]}...",
                "score": row["combined_score"],
                "vector": row["vector_score"],
                "deps": row["dep_similarity"],
            })
        if len(pairs) >= top_n:
            break
    
    pairs_df = pd.DataFrame(pairs)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y = np.arange(len(pairs_df))
    height = 0.35
    
    ax.barh(y - height/2, pairs_df["score"], height, label="Combined Score", color="#3399ff")
    ax.barh(y + height/2, pairs_df["vector"], height, label="Vector Similarity", color="#99ccff", alpha=0.7)
    
    ax.set_yticks(y)
    ax.set_yticklabels(pairs_df["pair"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Similarity Score")
    ax.set_title("Top Most Similar Repository Pairs", fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 1)
    
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved top pairs chart to {output}")


def plot_cluster_summary(cluster_path: Path, output: Path) -> None:
    """Plot framework usage from cluster report."""
    with open(cluster_path) as f:
        data = json.load(f)
    
    usage = data["summary"]["framework_usage"]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Framework usage
    ax1 = axes[0]
    frameworks = list(usage.keys())
    counts = list(usage.values())
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(frameworks)))
    
    bars = ax1.bar(frameworks, counts, color=colors)
    ax1.set_ylabel("Number of Projects")
    ax1.set_title("Framework/Tool Usage", fontsize=12, fontweight='bold')
    ax1.set_xticklabels(frameworks, rotation=45, ha='right')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(count), ha='center', va='bottom', fontsize=10)
    
    # Summary stats
    ax2 = axes[1]
    summary = data["summary"]
    categories = ["Total Repos", "With Docker", "With Tests", "With README"]
    values = [summary["total_repos"], summary["with_docker"], 
              summary["with_tests"], summary["with_readme"]]
    
    colors2 = ["#3399ff", "#66b2ff", "#99ccff", "#cce6ff"]
    bars2 = ax2.bar(categories, values, color=colors2)
    ax2.set_ylabel("Count")
    ax2.set_title("Project Statistics", fontsize=12, fontweight='bold')
    
    for bar, val in zip(bars2, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(val), ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved cluster summary to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize repository similarity data")
    parser.add_argument("--matrix", default="results/similarity_matrix.csv", help="Similarity matrix CSV")
    parser.add_argument("--similarity", default="results/repo_similarity_v2.csv", help="Detailed similarity CSV")
    parser.add_argument("--cluster", default="results/cluster_report.json", help="Cluster report JSON")
    parser.add_argument("--output-dir", default="results/plots", help="Output directory for plots")
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    
    # Plot heatmap
    if Path(args.matrix).exists():
        matrix = load_similarity_matrix(Path(args.matrix))
        plot_heatmap(matrix, output_dir / "similarity_heatmap.png")
    else:
        print(f"Matrix file not found: {args.matrix}")
    
    # Plot top pairs
    if Path(args.similarity).exists():
        plot_top_pairs(Path(args.similarity), output_dir / "top_similar_pairs.png")
    else:
        print(f"Similarity file not found: {args.similarity}")
    
    # Plot cluster summary
    if Path(args.cluster).exists():
        plot_cluster_summary(Path(args.cluster), output_dir / "cluster_summary.png")
    else:
        print(f"Cluster file not found: {args.cluster}")
    
    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
