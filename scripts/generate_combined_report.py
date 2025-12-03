#!/usr/bin/env python3
"""
Generate a combined HTML report that shows:
1. Clustered heatmap with plagiarism risk highlighting
2. Detailed similarity breakdown
3. Code comparison for flagged pairs
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import html


def load_deep_analysis(results_dir: Path) -> Tuple[List[dict], List[dict], Dict]:
    """Load all deep analysis results."""
    analyses = []
    similarities = []
    clusters = {}
    
    analyses_file = results_dir / "deep_analyses.json"
    if analyses_file.exists():
        with open(analyses_file) as f:
            analyses = json.load(f)
    
    sim_file = results_dir / "deep_similarity.csv"
    if sim_file.exists():
        with open(sim_file) as f:
            lines = f.readlines()[1:]  # Skip header
            for line in lines:
                parts = line.strip().split(",")
                if len(parts) >= 8:
                    similarities.append({
                        "repo1": parts[0],
                        "repo2": parts[1],
                        "combined": float(parts[2]),
                        "fingerprint": float(parts[3]),
                        "semantic": float(parts[4]),
                        "api_design": float(parts[5]),
                        "architecture": float(parts[6]),
                        "dependencies": float(parts[7]),
                        "style": float(parts[8]) if len(parts) > 8 else 0.0,
                    })
    
    clusters_file = results_dir / "clusters.json"
    if clusters_file.exists():
        with open(clusters_file) as f:
            clusters = json.load(f)
    
    return analyses, similarities, clusters


def load_plagiarism_results(results_dir: Path) -> List[dict]:
    """Load plagiarism detection results."""
    plagiarism_file = results_dir / "plagiarism_analysis.json"
    if plagiarism_file.exists():
        with open(plagiarism_file) as f:
            data = json.load(f)
            return data.get("flagged_pairs", [])
    return []


def classify_risk(fingerprint: float, semantic: float, api: float, arch: float) -> Tuple[str, str]:
    """Classify plagiarism risk based on multiple metrics."""
    if fingerprint >= 0.15:
        return "HIGH", "#ff4444"
    elif fingerprint >= 0.08 or (semantic >= 0.5 and api >= 0.6):
        return "MEDIUM", "#ffaa00"
    elif fingerprint >= 0.03 or (semantic >= 0.4 and api >= 0.5 and arch >= 0.8):
        return "LOW", "#ffff00"
    else:
        return "OK", "#44ff44"


def generate_html_report(deep_dir: Path, plagiarism_dir: Path, output_file: Path):
    """Generate combined HTML report."""
    
    analyses, similarities, clusters = load_deep_analysis(deep_dir)
    plagiarism_pairs = load_plagiarism_results(plagiarism_dir)
    
    # Build plagiarism lookup
    plagiarism_lookup = {}
    for p in plagiarism_pairs:
        key = tuple(sorted([p["repo1"], p["repo2"]]))
        plagiarism_lookup[key] = p
    
    # Classify all pairs
    classified_pairs = []
    for sim in similarities:
        risk, color = classify_risk(
            sim["fingerprint"],
            sim["semantic"],
            sim["api_design"],
            sim["architecture"]
        )
        sim["risk"] = risk
        sim["color"] = color
        
        # Check if also flagged by plagiarism detector
        key = tuple(sorted([sim["repo1"], sim["repo2"]]))
        sim["plagiarism_data"] = plagiarism_lookup.get(key)
        
        classified_pairs.append(sim)
    
    # Sort by risk then by fingerprint
    risk_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "OK": 3}
    classified_pairs.sort(key=lambda x: (risk_order[x["risk"]], -x["fingerprint"]))
    
    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Deep Code Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
        }}
        h1, h2, h3 {{
            color: #fff;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: #16213e;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 36px;
            font-weight: bold;
        }}
        .stat-label {{
            color: #888;
            margin-top: 5px;
        }}
        .high {{ color: #ff4444; }}
        .medium {{ color: #ffaa00; }}
        .low {{ color: #ffff00; }}
        .ok {{ color: #44ff44; }}
        
        .section {{
            background: #16213e;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #333;
        }}
        th {{
            background: #0f3460;
            position: sticky;
            top: 0;
        }}
        tr:hover {{
            background: #1f4287;
        }}
        
        .risk-badge {{
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
        }}
        .risk-HIGH {{ background: #ff4444; color: white; }}
        .risk-MEDIUM {{ background: #ffaa00; color: black; }}
        .risk-LOW {{ background: #ffff00; color: black; }}
        .risk-OK {{ background: #44ff44; color: black; }}
        
        .metric-bar {{
            height: 20px;
            background: #333;
            border-radius: 4px;
            overflow: hidden;
        }}
        .metric-fill {{
            height: 100%;
            transition: width 0.3s;
        }}
        .fingerprint-fill {{ background: linear-gradient(90deg, #ff6b6b, #ee5a24); }}
        .semantic-fill {{ background: linear-gradient(90deg, #48dbfb, #0abde3); }}
        .api-fill {{ background: linear-gradient(90deg, #1dd1a1, #10ac84); }}
        .arch-fill {{ background: linear-gradient(90deg, #feca57, #ff9f43); }}
        
        .pair-details {{
            background: #0f3460;
            padding: 15px;
            border-radius: 8px;
            margin-top: 10px;
            display: none;
        }}
        .pair-details.show {{
            display: block;
        }}
        
        .code-diff {{
            background: #0d1117;
            padding: 15px;
            border-radius: 8px;
            font-family: monospace;
            overflow-x: auto;
            white-space: pre;
            font-size: 12px;
        }}
        
        .toggle-btn {{
            background: #0f3460;
            border: none;
            color: #eee;
            padding: 5px 10px;
            border-radius: 4px;
            cursor: pointer;
        }}
        .toggle-btn:hover {{
            background: #1f4287;
        }}
        
        .cluster-group {{
            margin: 10px 0;
            padding: 15px;
            background: #0f3460;
            border-radius: 8px;
        }}
        .cluster-repos {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 10px;
        }}
        .repo-chip {{
            background: #1f4287;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 14px;
        }}
        
        .images {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 20px;
        }}
        .images img {{
            width: 100%;
            border-radius: 8px;
            cursor: pointer;
        }}
        .images img:hover {{
            transform: scale(1.02);
            transition: transform 0.2s;
        }}
        
        .filter-controls {{
            margin-bottom: 20px;
            display: flex;
            gap: 10px;
        }}
        .filter-btn {{
            padding: 8px 16px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            background: #0f3460;
            color: #eee;
        }}
        .filter-btn.active {{
            background: #667eea;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Deep Code Analysis Report</h1>
        <p>Comprehensive analysis of {len(analyses)} student repositories</p>
    </div>
    
    <div class="summary">
        <div class="stat-card">
            <div class="stat-value">{len(analyses)}</div>
            <div class="stat-label">Repositories</div>
        </div>
        <div class="stat-card">
            <div class="stat-value high">{len([p for p in classified_pairs if p['risk'] == 'HIGH'])}</div>
            <div class="stat-label">High Risk Pairs</div>
        </div>
        <div class="stat-card">
            <div class="stat-value medium">{len([p for p in classified_pairs if p['risk'] == 'MEDIUM'])}</div>
            <div class="stat-label">Medium Risk Pairs</div>
        </div>
        <div class="stat-card">
            <div class="stat-value low">{len([p for p in classified_pairs if p['risk'] == 'LOW'])}</div>
            <div class="stat-label">Low Risk Pairs</div>
        </div>
    </div>
"""
    
    # Add visualizations section
    html_content += """
    <div class="section">
        <h2>📊 Visualizations</h2>
        <div class="images">
            <div>
                <h3>Clustered Heatmap</h3>
                <a href="deep_analysis/plots/clustered_heatmap.png" target="_blank">
                    <img src="deep_analysis/plots/clustered_heatmap.png" alt="Clustered Heatmap">
                </a>
            </div>
            <div>
                <h3>Multi-Metric Comparison</h3>
                <a href="deep_analysis/plots/multi_metric_comparison.png" target="_blank">
                    <img src="deep_analysis/plots/multi_metric_comparison.png" alt="Multi-Metric Comparison">
                </a>
            </div>
        </div>
        <div class="images" style="margin-top: 20px;">
            <div>
                <h3>Top Similar Pairs</h3>
                <a href="deep_analysis/plots/top_similar_pairs.png" target="_blank">
                    <img src="deep_analysis/plots/top_similar_pairs.png" alt="Top Similar Pairs">
                </a>
            </div>
            <div>
                <h3>Cluster Groups</h3>
                <a href="deep_analysis/plots/cluster_groups.png" target="_blank">
                    <img src="deep_analysis/plots/cluster_groups.png" alt="Cluster Groups">
                </a>
            </div>
        </div>
    </div>
"""
    
    # Add flagged pairs section
    html_content += """
    <div class="section">
        <h2>⚠️ Flagged Pairs</h2>
        <div class="filter-controls">
            <button class="filter-btn active" onclick="filterPairs('all')">All</button>
            <button class="filter-btn" onclick="filterPairs('HIGH')">High Risk</button>
            <button class="filter-btn" onclick="filterPairs('MEDIUM')">Medium Risk</button>
            <button class="filter-btn" onclick="filterPairs('LOW')">Low Risk</button>
        </div>
        <table id="pairs-table">
            <thead>
                <tr>
                    <th>Risk</th>
                    <th>Repository 1</th>
                    <th>Repository 2</th>
                    <th>Fingerprint</th>
                    <th>Semantic</th>
                    <th>API Design</th>
                    <th>Architecture</th>
                    <th>Combined</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
"""
    
    # Only show top 50 pairs (most suspicious)
    for i, pair in enumerate(classified_pairs[:50]):
        row_class = f"risk-row-{pair['risk']}"
        html_content += f"""
                <tr class="{row_class}" data-risk="{pair['risk']}">
                    <td><span class="risk-badge risk-{pair['risk']}">{pair['risk']}</span></td>
                    <td>{html.escape(pair['repo1'])}</td>
                    <td>{html.escape(pair['repo2'])}</td>
                    <td>
                        <div class="metric-bar">
                            <div class="metric-fill fingerprint-fill" style="width: {pair['fingerprint']*100:.0f}%"></div>
                        </div>
                        <small>{pair['fingerprint']:.3f}</small>
                    </td>
                    <td>
                        <div class="metric-bar">
                            <div class="metric-fill semantic-fill" style="width: {pair['semantic']*100:.0f}%"></div>
                        </div>
                        <small>{pair['semantic']:.3f}</small>
                    </td>
                    <td>
                        <div class="metric-bar">
                            <div class="metric-fill api-fill" style="width: {pair['api_design']*100:.0f}%"></div>
                        </div>
                        <small>{pair['api_design']:.3f}</small>
                    </td>
                    <td>
                        <div class="metric-bar">
                            <div class="metric-fill arch-fill" style="width: {pair['architecture']*100:.0f}%"></div>
                        </div>
                        <small>{pair['architecture']:.3f}</small>
                    </td>
                    <td><strong>{pair['combined']:.3f}</strong></td>
                    <td>
                        <button class="toggle-btn" onclick="toggleDetails({i})">Details</button>
                    </td>
                </tr>
                <tr id="details-{i}" style="display: none;">
                    <td colspan="9">
                        <div class="pair-details show">
                            <h4>Similarity Breakdown</h4>
                            <ul>
                                <li><strong>Fingerprint Match:</strong> {pair['fingerprint']:.1%} - Direct code overlap detected via n-gram fingerprinting</li>
                                <li><strong>Semantic Similarity:</strong> {pair['semantic']:.1%} - Similar variable/function names and code semantics</li>
                                <li><strong>API Design:</strong> {pair['api_design']:.1%} - Similar endpoint structure and patterns</li>
                                <li><strong>Architecture:</strong> {pair['architecture']:.1%} - Similar file structure and organization</li>
                                <li><strong>Dependencies:</strong> {pair['dependencies']:.1%} - Same libraries used</li>
                                <li><strong>Code Style:</strong> {pair['style']:.1%} - Similar formatting and naming conventions</li>
                            </ul>
"""
        
        # Add plagiarism data if available
        if pair.get("plagiarism_data"):
            pdata = pair["plagiarism_data"]
            html_content += f"""
                            <h4>Previous Plagiarism Analysis</h4>
                            <ul>
                                <li><strong>Plagiarism Score:</strong> {pdata.get('score', 0):.2f}</li>
                                <li><strong>Category:</strong> {pdata.get('category', 'Unknown')}</li>
                            </ul>
"""
            if pdata.get("evidence"):
                html_content += "                            <h4>Evidence</h4>\n"
                for ev in pdata["evidence"][:3]:
                    html_content += f"                            <div class='code-diff'>{html.escape(str(ev))}</div>\n"
        
        html_content += """
                        </div>
                    </td>
                </tr>
"""
    
    html_content += """
            </tbody>
        </table>
    </div>
"""
    
    # Add clusters section
    if clusters:
        html_content += """
    <div class="section">
        <h2>🔗 Repository Clusters</h2>
        <p>Repositories grouped by overall similarity (not necessarily indicating plagiarism)</p>
"""
        # Handle different cluster data formats
        cluster_list = clusters.get("clusters", []) if isinstance(clusters, dict) else []
        if isinstance(clusters, dict) and "clusters" in clusters:
            cluster_list = clusters["clusters"]
        elif isinstance(clusters, dict):
            cluster_list = list(clusters.values())
        
        for i, repos in enumerate(cluster_list):
            if isinstance(repos, list) and len(repos) > 1:
                html_content += f"""
        <div class="cluster-group">
            <h3>Cluster {i + 1} ({len(repos)} repos)</h3>
            <div class="cluster-repos">
"""
                for repo in repos:
                    repo_name = repo if isinstance(repo, str) else str(repo)
                    html_content += f'                <span class="repo-chip">{html.escape(repo_name)}</span>\n'
                html_content += """
            </div>
        </div>
"""
        html_content += "    </div>\n"
    
    # JavaScript for interactivity
    html_content += """
    <script>
        function toggleDetails(idx) {
            const row = document.getElementById('details-' + idx);
            if (row.style.display === 'none') {
                row.style.display = 'table-row';
            } else {
                row.style.display = 'none';
            }
        }
        
        function filterPairs(risk) {
            const rows = document.querySelectorAll('#pairs-table tbody tr');
            const buttons = document.querySelectorAll('.filter-btn');
            
            buttons.forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            
            rows.forEach(row => {
                if (risk === 'all' || row.dataset.risk === risk || !row.dataset.risk) {
                    row.style.display = '';
                } else {
                    row.style.display = 'none';
                }
            });
        }
    </script>
</body>
</html>
"""
    
    with open(output_file, "w") as f:
        f.write(html_content)
    
    print(f"✅ Combined report saved to {output_file}")


def main():
    results_dir = Path("results")
    deep_dir = results_dir / "deep_analysis"
    output_file = results_dir / "combined_analysis_report.html"
    
    if not deep_dir.exists():
        print("Error: Run deep_analysis.py first")
        return
    
    generate_html_report(deep_dir, results_dir, output_file)


if __name__ == "__main__":
    main()
