#!/usr/bin/env python3
"""Advanced visualization for deep code analysis with clustering heatmaps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

# Use a non-interactive backend
import matplotlib
matplotlib.use('Agg')


def load_data(output_dir: Path) -> Tuple[List[dict], Dict, List[List[str]]]:
    """Load analysis data from files."""
    with open(output_dir / "deep_analyses.json") as f:
        analyses = json.load(f)
    
    with open(output_dir / "deep_similarity_matrix.json") as f:
        raw_matrix = json.load(f)
    
    # Convert keys back to tuples
    similarity_matrix = {}
    for k, v in raw_matrix.items():
        r1, r2 = k.split("|")
        similarity_matrix[(r1, r2)] = v
    
    with open(output_dir / "clusters.json") as f:
        clusters = json.load(f)["clusters"]
    
    return analyses, similarity_matrix, clusters


def build_matrix(names: List[str], similarity_matrix: Dict, metric: str = 'combined') -> np.ndarray:
    """Build NxN similarity matrix."""
    n = len(names)
    matrix = np.eye(n)  # Diagonal is 1
    
    name_to_idx = {name: i for i, name in enumerate(names)}
    
    for (r1, r2), sims in similarity_matrix.items():
        i = name_to_idx.get(r1)
        j = name_to_idx.get(r2)
        if i is not None and j is not None:
            val = sims.get(metric, 0)
            matrix[i, j] = val
            matrix[j, i] = val
    
    return matrix


def create_clustered_heatmap(
    names: List[str],
    similarity_matrix: Dict,
    output_path: Path,
    metric: str = 'combined',
    title: str = 'Repository Similarity Matrix (Clustered)',
) -> None:
    """Create a clustered heatmap with dendrogram."""
    matrix = build_matrix(names, similarity_matrix, metric)
    n = len(names)
    
    # Convert similarity to distance for clustering
    distance_matrix = 1 - matrix
    np.fill_diagonal(distance_matrix, 0)
    
    # Ensure symmetry and valid values
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    distance_matrix = np.clip(distance_matrix, 0, 1)
    
    # Hierarchical clustering
    condensed = squareform(distance_matrix)
    linkage = hierarchy.linkage(condensed, method='average')
    
    # Get dendrogram order
    dendro = hierarchy.dendrogram(linkage, no_plot=True)
    order = dendro['leaves']
    
    # Reorder matrix and names
    ordered_matrix = matrix[np.ix_(order, order)]
    ordered_names = [names[i] for i in order]
    
    # Create figure with dendrogram
    fig = plt.figure(figsize=(16, 14))
    
    # Main heatmap
    ax_heatmap = fig.add_axes([0.15, 0.1, 0.7, 0.7])
    
    # Custom colormap: white -> light blue -> blue -> dark blue -> purple (for high similarity)
    colors = [
        '#FFFFFF',  # 0.0 - white
        '#E8F4FD',  # 0.2 - very light blue
        '#B3D9F7',  # 0.4 - light blue
        '#4DA6E8',  # 0.6 - medium blue
        '#1565C0',  # 0.8 - dark blue
        '#6A1B9A',  # 0.9 - purple (suspicious)
        '#D32F2F',  # 1.0 - red (high match)
    ]
    positions = [0.0, 0.3, 0.5, 0.65, 0.8, 0.9, 1.0]
    cmap = LinearSegmentedColormap.from_list('similarity', list(zip(positions, colors)))
    
    im = ax_heatmap.imshow(ordered_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.87, 0.1, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Similarity Score', fontsize=12)
    
    # Add threshold lines on colorbar
    for threshold, label in [(0.7, 'Suspicious'), (0.85, 'High Match')]:
        cbar.ax.axhline(y=threshold, color='black', linewidth=1, linestyle='--')
        cbar.ax.text(1.5, threshold, label, fontsize=8, va='center')
    
    # Labels
    short_names = [name[:18] + '...' if len(name) > 20 else name for name in ordered_names]
    ax_heatmap.set_xticks(np.arange(n))
    ax_heatmap.set_yticks(np.arange(n))
    ax_heatmap.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
    ax_heatmap.set_yticklabels(short_names, fontsize=8)
    
    # Add grid
    ax_heatmap.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax_heatmap.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax_heatmap.grid(which='minor', color='#CCCCCC', linestyle='-', linewidth=0.5)
    
    # Add values for high similarities
    for i in range(n):
        for j in range(n):
            if i != j and ordered_matrix[i, j] >= 0.6:
                color = 'white' if ordered_matrix[i, j] >= 0.7 else 'black'
                ax_heatmap.text(j, i, f'{ordered_matrix[i, j]:.2f}',
                              ha='center', va='center', fontsize=6, color=color)
    
    ax_heatmap.set_title(title, fontsize=14, fontweight='bold', pad=10)
    
    # Top dendrogram
    ax_dendro = fig.add_axes([0.15, 0.82, 0.7, 0.12])
    hierarchy.dendrogram(linkage, ax=ax_dendro, leaf_rotation=90, leaf_font_size=0,
                        color_threshold=0.5 * max(linkage[:, 2]))
    ax_dendro.set_xticks([])
    ax_dendro.set_ylabel('Distance', fontsize=9)
    ax_dendro.spines['top'].set_visible(False)
    ax_dendro.spines['right'].set_visible(False)
    ax_dendro.spines['bottom'].set_visible(False)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved clustered heatmap to {output_path}")


def create_multi_metric_comparison(
    names: List[str],
    similarity_matrix: Dict,
    output_path: Path,
) -> None:
    """Create comparison of different similarity metrics."""
    metrics = ['combined', 'fingerprint', 'semantic', 'api_design', 'architecture', 'vector']
    metric_labels = ['Combined', 'Code Fingerprint', 'Semantic', 'API Design', 'Architecture', 'Vector Embedding']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Color scheme
    colors = ['#FFFFFF', '#E3F2FD', '#90CAF9', '#42A5F5', '#1E88E5', '#1565C0', '#D32F2F']
    positions = [0.0, 0.3, 0.5, 0.65, 0.8, 0.9, 1.0]
    cmap = LinearSegmentedColormap.from_list('similarity', list(zip(positions, colors)))
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        matrix = build_matrix(names, similarity_matrix, metric)
        
        im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)
        ax.set_title(label, fontsize=11, fontweight='bold')
        
        # Simplified labels
        if len(names) <= 15:
            short_names = [n[:10] for n in names]
            ax.set_xticks(np.arange(len(names)))
            ax.set_yticks(np.arange(len(names)))
            ax.set_xticklabels(short_names, rotation=90, fontsize=6)
            ax.set_yticklabels(short_names, fontsize=6)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle('Multi-Dimensional Similarity Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved multi-metric comparison to {output_path}")


def create_cluster_visualization(
    clusters: List[List[str]],
    similarity_matrix: Dict,
    output_path: Path,
) -> None:
    """Visualize clusters with internal similarity."""
    # Only show clusters with more than 1 member
    significant_clusters = [c for c in clusters if len(c) > 1]
    
    if not significant_clusters:
        print("No significant clusters to visualize")
        return
    
    fig, ax = plt.subplots(figsize=(14, max(6, len(significant_clusters) * 0.8)))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(significant_clusters)))
    
    y_positions = []
    y = 0
    
    for i, cluster in enumerate(significant_clusters):
        # Calculate average internal similarity
        internal_sims = []
        for j, r1 in enumerate(cluster):
            for r2 in cluster[j+1:]:
                key = (r1, r2) if (r1, r2) in similarity_matrix else (r2, r1)
                if key in similarity_matrix:
                    internal_sims.append(similarity_matrix[key]['combined'])
        
        avg_sim = np.mean(internal_sims) if internal_sims else 0
        
        # Draw cluster box
        height = len(cluster) * 0.5 + 0.3
        rect = mpatches.FancyBboxPatch(
            (0.05, y), 0.6, height,
            boxstyle="round,pad=0.02",
            facecolor=colors[i],
            edgecolor='black',
            linewidth=2,
            alpha=0.7
        )
        ax.add_patch(rect)
        
        # Cluster label
        ax.text(0.35, y + height/2, f"Cluster {i+1}\n(avg sim: {avg_sim:.2f})",
               ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Member names
        for j, name in enumerate(cluster):
            ax.text(0.7, y + 0.2 + j * 0.5, f"• {name[:30]}", fontsize=9, va='center')
        
        y += height + 0.3
    
    ax.set_xlim(0, 1.5)
    ax.set_ylim(-0.2, y)
    ax.axis('off')
    ax.set_title('Repository Clusters (Similar Code Groups)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved cluster visualization to {output_path}")


def create_top_pairs_chart(
    similarity_matrix: Dict,
    output_path: Path,
    top_n: int = 20,
) -> None:
    """Create horizontal bar chart of top similar pairs."""
    # Sort pairs by combined similarity
    sorted_pairs = sorted(
        [(k, v) for k, v in similarity_matrix.items()],
        key=lambda x: -x[1]['combined']
    )[:top_n]
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    y_pos = np.arange(len(sorted_pairs))
    
    # Different metrics as stacked bars
    metrics = ['fingerprint', 'semantic', 'api_design', 'vector']
    colors = ['#D32F2F', '#1976D2', '#388E3C', '#7B1FA2']
    weights = [0.25, 0.20, 0.15, 0.30]  # Match the weights used in deep_analysis.py
    
    left = np.zeros(len(sorted_pairs))
    for metric, color, weight in zip(metrics, colors, weights):
        values = [p[1][metric] * weight for p in sorted_pairs]
        ax.barh(y_pos, values, left=left, color=color, label=metric.replace('_', ' ').title(), alpha=0.8)
        left += values
    
    # Add combined score as text
    for i, (pair, sims) in enumerate(sorted_pairs):
        ax.text(sims['combined'] + 0.01, i, f"{sims['combined']:.2f}", va='center', fontsize=9)
    
    # Labels
    pair_labels = [f"{p[0][:15]}... ↔ {p[1][:15]}..." for (p, _) in sorted_pairs]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pair_labels, fontsize=9)
    ax.invert_yaxis()
    
    ax.set_xlabel('Weighted Similarity Score', fontsize=11)
    ax.set_title('Top Most Similar Repository Pairs (Component Breakdown)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(0, 1.1)
    
    # Add threshold lines
    ax.axvline(x=0.7, color='orange', linestyle='--', linewidth=2, label='Suspicious threshold')
    ax.axvline(x=0.85, color='red', linestyle='--', linewidth=2, label='High match threshold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved top pairs chart to {output_path}")


def create_framework_analysis(
    analyses: List[dict],
    output_path: Path,
) -> None:
    """Create framework and architecture analysis charts."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Framework usage
    ax1 = axes[0, 0]
    frameworks = {}
    for a in analyses:
        for fw in a.get('frameworks', []):
            frameworks[fw] = frameworks.get(fw, 0) + 1
    
    if frameworks:
        sorted_fw = sorted(frameworks.items(), key=lambda x: -x[1])
        ax1.barh([x[0] for x in sorted_fw], [x[1] for x in sorted_fw], color='#1976D2')
        ax1.set_xlabel('Number of Projects')
        ax1.set_title('Framework Usage', fontweight='bold')
    
    # Architecture patterns
    ax2 = axes[0, 1]
    arch_features = {
        'Docker': sum(1 for a in analyses if a['architecture']['has_dockerfile']),
        'Docker Compose': sum(1 for a in analyses if a['architecture']['has_docker_compose']),
        'Tests': sum(1 for a in analyses if a['architecture']['has_tests_folder']),
        'Models': sum(1 for a in analyses if a['architecture']['has_models_file']),
        'Routes': sum(1 for a in analyses if a['architecture']['has_routes_file']),
        'CRUD': sum(1 for a in analyses if a['architecture']['has_crud_file']),
        'Config': sum(1 for a in analyses if a['architecture']['has_config_file']),
    }
    
    ax2.bar(arch_features.keys(), arch_features.values(), color='#388E3C')
    ax2.set_ylabel('Number of Projects')
    ax2.set_title('Architecture Components', fontweight='bold')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Code complexity distribution
    ax3 = axes[1, 0]
    func_counts = [a['complexity']['total_functions'] for a in analyses]
    class_counts = [a['complexity']['total_classes'] for a in analyses]
    
    x = np.arange(len(analyses))
    width = 0.35
    ax3.bar(x - width/2, func_counts, width, label='Functions', color='#7B1FA2')
    ax3.bar(x + width/2, class_counts, width, label='Classes', color='#F57C00')
    ax3.set_xlabel('Repository')
    ax3.set_ylabel('Count')
    ax3.set_title('Functions vs Classes per Repo', fontweight='bold')
    ax3.legend()
    ax3.set_xticks([])
    
    # Lines of code distribution
    ax4 = axes[1, 1]
    lines = [a['architecture']['total_code_lines'] for a in analyses]
    names = [a['name'][:12] for a in analyses]
    
    ax4.bar(range(len(lines)), sorted(lines, reverse=True), color='#00897B')
    ax4.set_xlabel('Repository (sorted)')
    ax4.set_ylabel('Lines of Code')
    ax4.set_title('Code Size Distribution', fontweight='bold')
    ax4.set_xticks([])
    
    plt.suptitle('Repository Analysis Overview', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved framework analysis to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize deep analysis results")
    parser.add_argument("--input-dir", default="results/deep_analysis", help="Input directory with analysis results")
    parser.add_argument("--output-dir", default="results/deep_analysis/plots", help="Output directory for plots")
    
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    analyses, similarity_matrix, clusters = load_data(input_dir)
    names = [a['name'] for a in analyses]
    
    print(f"Loaded {len(analyses)} analyses, {len(similarity_matrix)} similarity pairs, {len(clusters)} clusters")
    
    print("\nGenerating visualizations...")
    
    # Main clustered heatmap
    create_clustered_heatmap(
        names, similarity_matrix,
        output_dir / "clustered_heatmap.png",
        metric='combined',
        title='Repository Similarity Matrix (Hierarchical Clustering)'
    )
    
    # Multi-metric comparison
    create_multi_metric_comparison(
        names, similarity_matrix,
        output_dir / "multi_metric_comparison.png"
    )
    
    # Cluster visualization
    create_cluster_visualization(
        clusters, similarity_matrix,
        output_dir / "cluster_groups.png"
    )
    
    # Top pairs chart
    create_top_pairs_chart(
        similarity_matrix,
        output_dir / "top_similar_pairs.png"
    )
    
    # Framework analysis
    create_framework_analysis(
        analyses,
        output_dir / "framework_analysis.png"
    )
    
    print(f"\n✅ All visualizations saved to {output_dir}/")


if __name__ == "__main__":
    main()
