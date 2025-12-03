#!/usr/bin/env python3
"""Plot similarity results produced by scripts/compare_repos.py.

Reads a CSV (results/repo_similarity.csv by default) and writes a PNG plot
summarizing the strongest matches.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def make_plot(df: pd.DataFrame, out_path: Path, top_n: int) -> None:
    df = df.copy()
    df["pair"] = df["repo"] + " → " + df["match"]

    # Top-N by score
    top = df.sort_values("score", ascending=False).head(top_n)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Top pairs (bar)
    axes[0, 0].barh(top["pair"], top["score"], color="#4C72B0")
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_xlabel("Similarity score")
    axes[0, 0].set_title(f"Top {top_n} repo pairs")

    # Scatter: overlap vs score
    axes[0, 1].scatter(df["file_overlap"], df["score"], alpha=0.6, color="#55A868")
    axes[0, 1].set_xlabel("File path overlap (count)")
    axes[0, 1].set_ylabel("Similarity score")
    axes[0, 1].set_title("Score vs. file overlap")
    axes[0, 1].grid(True, linestyle="--", alpha=0.3)

    # Heatmap for top repos by max score involvement
    repo_scores = (
        pd.concat(
            [
                df[["repo", "score"]].rename(columns={"repo": "name"}),
                df[["match", "score"]].rename(columns={"match": "name"}),
            ]
        )
        .groupby("name")["score"]
        .max()
        .sort_values(ascending=False)
    )
    heat_repos = repo_scores.head(max(10, top_n // 2)).index.tolist()
    heat_df = pd.DataFrame(0.0, index=heat_repos, columns=heat_repos)
    for _, row in df.iterrows():
        if row["repo"] in heat_repos and row["match"] in heat_repos:
            current = heat_df.loc[row["repo"], row["match"]]
            new_score = max(current, row["score"])
            heat_df.loc[row["repo"], row["match"]] = new_score
            heat_df.loc[row["match"], row["repo"]] = new_score

    im = axes[1, 0].imshow(
        heat_df.values,
        cmap="Blues",
        vmin=heat_df.values.min(),
        vmax=heat_df.values.max(),
    )
    axes[1, 0].set_xticks(range(len(heat_repos)))
    axes[1, 0].set_xticklabels(heat_repos, rotation=45, ha="right", fontsize=8)
    axes[1, 0].set_yticks(range(len(heat_repos)))
    axes[1, 0].set_yticklabels(heat_repos, fontsize=8)
    axes[1, 0].set_title("Similarity heatmap (top repos)")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Histogram of scores
    axes[1, 1].hist(df["score"], bins=20, color="#C44E52", alpha=0.8)
    axes[1, 1].set_xlabel("Similarity score")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("Score distribution")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot repo similarity results.")
    parser.add_argument(
        "--input",
        default="results/repo_similarity.csv",
        help="CSV produced by compare_repos.py",
    )
    parser.add_argument(
        "--output",
        default="results/repo_similarity.png",
        help="Where to write the plot PNG",
    )
    parser.add_argument("--top-n", type=int, default=20, help="Top pairs to plot")
    args = parser.parse_args()

    csv_path = Path(args.input)
    if not csv_path.exists():
        raise SystemExit(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {"repo", "match", "score", "file_overlap"}
    if not required_cols.issubset(df.columns):
        raise SystemExit(f"CSV missing columns: {required_cols - set(df.columns)}")

    make_plot(df, Path(args.output), args.top_n)
    print(f"Wrote plot to {args.output}")


if __name__ == "__main__":
    main()
