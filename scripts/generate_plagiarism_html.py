#!/usr/bin/env python3
"""Generate HTML plagiarism report for easy review."""

import json
import html
from pathlib import Path
from datetime import datetime

def generate_html_report(json_path: Path, output_path: Path) -> None:
    """Generate HTML report from JSON plagiarism data."""
    
    with open(json_path) as f:
        data = json.load(f)
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Plagiarism Detection Report</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{ color: #333; border-bottom: 3px solid #e74c3c; padding-bottom: 10px; }}
        h2 {{ color: #444; margin-top: 30px; }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .summary-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .summary-card.high {{ border-left: 4px solid #e74c3c; }}
        .summary-card.suspicious {{ border-left: 4px solid #f39c12; }}
        .summary-card.review {{ border-left: 4px solid #3498db; }}
        .summary-card.clean {{ border-left: 4px solid #27ae60; }}
        .summary-card h3 {{ margin: 0 0 10px 0; font-size: 2em; }}
        .summary-card p {{ margin: 0; color: #666; }}
        
        .pair {{
            background: white;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .pair-header {{
            padding: 15px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
        }}
        .pair-header.high {{ background: #e74c3c; color: white; }}
        .pair-header.suspicious {{ background: #f39c12; color: white; }}
        .pair-header.review {{ background: #3498db; color: white; }}
        .pair-header h3 {{ margin: 0; font-size: 1.1em; }}
        .pair-header .score {{ 
            background: rgba(255,255,255,0.2);
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
        }}
        
        .pair-content {{ padding: 20px; display: none; }}
        .pair.expanded .pair-content {{ display: block; }}
        
        .match {{
            border: 1px solid #ddd;
            border-radius: 4px;
            margin: 15px 0;
            overflow: hidden;
        }}
        .match-header {{
            background: #f8f9fa;
            padding: 10px 15px;
            border-bottom: 1px solid #ddd;
            display: flex;
            justify-content: space-between;
        }}
        .match-type {{
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 0.85em;
            font-weight: bold;
        }}
        .match-type.exact {{ background: #e74c3c; color: white; }}
        .match-type.near_exact {{ background: #e67e22; color: white; }}
        .match-type.structural {{ background: #9b59b6; color: white; }}
        .match-type.naming_pattern {{ background: #3498db; color: white; }}
        .match-type.fingerprint_overlap {{ background: #1abc9c; color: white; }}
        
        .code-comparison {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            padding: 15px;
        }}
        .code-block {{
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 0.85em;
            white-space: pre-wrap;
            word-break: break-all;
        }}
        .code-block h4 {{
            color: #888;
            margin: 0 0 10px 0;
            font-size: 0.9em;
        }}
        
        .evidence {{
            background: #fff3cd;
            border: 1px solid #ffc107;
            padding: 10px 15px;
            border-radius: 4px;
            margin-top: 10px;
        }}
        .evidence ul {{ margin: 5px 0; padding-left: 20px; }}
        
        .timestamp {{ color: #888; font-size: 0.9em; margin-top: 30px; }}
        
        @media (max-width: 800px) {{
            .code-comparison {{ grid-template-columns: 1fr; }}
        }}
    </style>
    <script>
        function togglePair(el) {{
            el.closest('.pair').classList.toggle('expanded');
        }}
    </script>
</head>
<body>
    <h1>🔍 Code Plagiarism Detection Report</h1>
    
    <div class="summary">
        <div class="summary-card high">
            <h3>{data['summary']['high_plagiarism']}</h3>
            <p>🚨 High Plagiarism</p>
        </div>
        <div class="summary-card suspicious">
            <h3>{data['summary']['suspicious']}</h3>
            <p>⚠️ Suspicious</p>
        </div>
        <div class="summary-card review">
            <h3>{data['summary']['needs_review']}</h3>
            <p>📋 Needs Review</p>
        </div>
        <div class="summary-card clean">
            <h3>{data['summary']['total_pairs_analyzed'] - data['summary']['high_plagiarism'] - data['summary']['suspicious'] - data['summary']['needs_review']}</h3>
            <p>✅ Clean</p>
        </div>
    </div>
    
    <h2>Flagged Pairs</h2>
"""
    
    # Group matches by pair
    matches_by_pair = {}
    for match in data['detailed_matches']:
        key = (match['repo1'], match['repo2'])
        if key not in matches_by_pair:
            matches_by_pair[key] = []
        matches_by_pair[key].append(match)
    
    for pair in data['flagged_pairs']:
        verdict_class = pair['verdict'].lower().replace('_', '-').replace('high-plagiarism', 'high').replace('needs-review', 'review')
        verdict_emoji = {'high': '🚨', 'suspicious': '⚠️', 'review': '📋'}.get(verdict_class, '')
        
        html_content += f"""
    <div class="pair">
        <div class="pair-header {verdict_class}" onclick="togglePair(this)">
            <h3>{verdict_emoji} {html.escape(pair['repo1'])} ↔ {html.escape(pair['repo2'])}</h3>
            <span class="score">Score: {pair['score']:.2f}</span>
        </div>
        <div class="pair-content">
"""
        
        pair_matches = matches_by_pair.get((pair['repo1'], pair['repo2']), [])
        for match in pair_matches:
            type_class = match['type']
            html_content += f"""
            <div class="match">
                <div class="match-header">
                    <span><strong>{html.escape(match['block1'])} ↔ {html.escape(match['block2'])}</strong> ({match['similarity']:.0%})</span>
                    <span class="match-type {type_class}">{match['type'].replace('_', ' ').title()}</span>
                </div>
                <div class="code-comparison">
                    <div class="code-block">
                        <h4>📁 {html.escape(match['file1'])}</h4>{html.escape(match['code1_preview'])}
                    </div>
                    <div class="code-block">
                        <h4>📁 {html.escape(match['file2'])}</h4>{html.escape(match['code2_preview'])}
                    </div>
                </div>
                <div class="evidence">
                    <strong>Evidence:</strong>
                    <ul>
                        {''.join(f'<li>{html.escape(e)}</li>' for e in match['evidence'])}
                    </ul>
                </div>
            </div>
"""
        
        html_content += """
        </div>
    </div>
"""
    
    html_content += f"""
    <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</body>
</html>
"""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content)
    print(f"Generated HTML report: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/plagiarism_report.json")
    parser.add_argument("--output", default="results/plagiarism_report.html")
    args = parser.parse_args()
    
    generate_html_report(Path(args.input), Path(args.output))
