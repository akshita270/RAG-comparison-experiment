"""
visualizations/charts.py — All chart generation for the experiment.

Generates four visualisations:
  1. faithfulness_bar.png    — Average faithfulness score per architecture
  2. latency_bar.png         — Average query latency per architecture
  3. radar_chart.png         — All metrics per architecture (spider chart)
  4. faithfulness_heatmap.png — Per-question faithfulness across all architectures

Run as a standalone script after the main experiment completes:
    python visualizations/charts.py
"""
import sys
import logging
from pathlib import Path

from typing import Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Shared style settings
# ─────────────────────────────────────────────────────────────────────────────

ARCH_COLORS = {
    "vanilla_rag": "#4C72B0",   # Blue
    "hyde_rag":    "#DD8452",   # Orange
    "graph_rag":   "#55A868",   # Green
    "agentic_rag": "#C44E52",   # Red
}
ARCH_LABELS = {
    "vanilla_rag": "Vanilla RAG",
    "hyde_rag":    "HyDE RAG",
    "graph_rag":   "Graph RAG",
    "agentic_rag": "Agentic RAG",
}

plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.size":       11,
    "axes.titlesize":  13,
    "axes.titleweight": "bold",
    "figure.dpi":      150,
})


def _save(fig, filename: str):
    """Save a figure to the visualizations/output/ directory."""
    config.VIZ_DIR.mkdir(parents=True, exist_ok=True)
    path = config.VIZ_DIR / filename
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"Saved chart: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Chart 1: Faithfulness bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_faithfulness_bar(df: pd.DataFrame) -> Path:
    """
    Grouped bar chart showing mean faithfulness ± std per architecture.

    Faithfulness is the primary hallucination metric — higher is better.
    Error bars show standard deviation across 25 questions.
    """
    summary = df.groupby("architecture")["faithfulness"].agg(["mean", "std"]).reset_index()
    archs = summary["architecture"].tolist()
    means = summary["mean"].tolist()
    stds  = summary["std"].tolist()
    colors = [ARCH_COLORS.get(a, "#999999") for a in archs]
    labels = [ARCH_LABELS.get(a, a) for a in archs]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, means, yerr=stds, color=colors, capsize=5,
                  edgecolor="white", linewidth=0.8, width=0.6)

    # Add value labels on top of each bar
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title("Faithfulness Score by Architecture\n(higher = less hallucination)")
    ax.set_ylabel("Mean Faithfulness Score (0–1)")
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.5, color="grey", linestyle="--", alpha=0.4, label="Midpoint")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    return _save(fig, "faithfulness_bar.png")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 2: Latency bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_latency_bar(df: pd.DataFrame) -> Path:
    """
    Bar chart showing average query latency in milliseconds per architecture.

    Agentic RAG is expected to be slowest (multiple tool calls).
    Vanilla RAG is the fastest baseline.
    """
    summary = df.groupby("architecture")["latency_ms"].agg(["mean", "std"]).reset_index()
    archs = summary["architecture"].tolist()
    means = summary["mean"].tolist()
    stds  = summary["std"].tolist()
    colors = [ARCH_COLORS.get(a, "#999999") for a in archs]
    labels = [ARCH_LABELS.get(a, a) for a in archs]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, means, yerr=stds, color=colors, capsize=5,
                  edgecolor="white", linewidth=0.8, width=0.6)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                f"{mean:.0f} ms", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title("Average Query Latency by Architecture\n(lower = faster)")
    ax.set_ylabel("Mean Latency (ms)")
    ax.spines[["top", "right"]].set_visible(False)

    return _save(fig, "latency_bar.png")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 3: Radar / spider chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_radar_chart(df: pd.DataFrame) -> Path:
    """
    Spider/radar chart overlaying all architectures across all metrics.

    Shows trade-offs at a glance — an architecture may score high on
    faithfulness but low on latency-normalised performance.

    Note: latency is inverted (1 - normalised_latency) so that higher
    always means better on all axes.
    """
    metrics = ["faithfulness", "answer_relevancy", "context_precision",
               "judge_factuality", "judge_relevance", "judge_citation"]
    metric_labels = ["Faithfulness", "Answer\nRelevancy", "Context\nPrecision",
                     "Judge\nFactuality", "Judge\nRelevance", "Judge\nCitation"]

    # Normalise judge scores (1-5) to 0-1 to match RAGAS metrics
    df = df.copy()
    for col in ["judge_factuality", "judge_relevance", "judge_citation"]:
        df[col] = (df[col] - 1) / 4  # map [1,5] → [0,1]

    summary = df.groupby("architecture")[metrics].mean()
    archs   = summary.index.tolist()
    N       = len(metrics)
    angles  = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for arch in archs:
        values  = summary.loc[arch, metrics].tolist()
        values += values[:1]
        color   = ARCH_COLORS.get(arch, "#999999")
        label   = ARCH_LABELS.get(arch, arch)
        ax.plot(angles, values, "o-", linewidth=2, color=color, label=label)
        ax.fill(angles, values, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, size=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], size=8)
    ax.set_title("All Metrics per Architecture\n(higher = better)", pad=20, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.5)

    return _save(fig, "radar_chart.png")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 4: Faithfulness heatmap (questions × architectures)
# ─────────────────────────────────────────────────────────────────────────────

def plot_faithfulness_heatmap(df: pd.DataFrame) -> Path:
    """
    Heatmap: rows = questions, columns = architectures, cells = faithfulness score.

    Reveals per-question patterns — some questions may be consistently hard
    for all architectures, while others show large variance between architectures.
    Red cells = low faithfulness (hallucination). Green = high faithfulness.
    """
    pivot = df.pivot_table(
        index="question_id", columns="architecture", values="faithfulness", aggfunc="mean"
    )
    # Rename columns for display
    pivot.columns = [ARCH_LABELS.get(c, c) for c in pivot.columns]
    # Short question labels for y-axis
    pivot.index = [f"Q{i+1}" for i in range(len(pivot))]

    fig_height = max(6, len(pivot) * 0.35)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    sns.heatmap(
        pivot,
        ax=ax,
        cmap="RdYlGn",   # Red (low) → Yellow (mid) → Green (high)
        vmin=0, vmax=1,
        annot=len(pivot) <= 30,   # Only annotate if not too many rows
        fmt=".2f",
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "Faithfulness Score", "shrink": 0.8},
    )
    ax.set_title("Faithfulness Score: Questions × Architectures\n(green = faithful, red = hallucination)", pad=12)
    ax.set_xlabel("Architecture")
    ax.set_ylabel("Question")
    ax.tick_params(axis="x", rotation=20)

    return _save(fig, "faithfulness_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_charts(df: pd.DataFrame) -> Dict[str, Path]:
    """
    Generate and save all 4 charts. Returns a dict of {chart_name: path}.
    Call this from main.py after the results DataFrame is built.
    """
    logger.info("Generating visualisations …")

    # Add a question_id column for the heatmap if not present
    if "question_id" not in df.columns:
        q_map = {q: i for i, q in enumerate(df["question"].unique())}
        df = df.copy()
        df["question_id"] = df["question"].map(q_map)

    paths = {}
    paths["faithfulness_bar"]     = plot_faithfulness_bar(df)
    paths["latency_bar"]          = plot_latency_bar(df)
    paths["radar_chart"]          = plot_radar_chart(df)
    paths["faithfulness_heatmap"] = plot_faithfulness_heatmap(df)

    logger.info(f"All charts saved to {config.VIZ_DIR}/")
    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if not config.RESULTS_FILE.exists():
        print(f"Results file not found: {config.RESULTS_FILE}")
        print("Run main.py first to generate results.")
        sys.exit(1)
    df = pd.read_csv(config.RESULTS_FILE)
    generate_all_charts(df)
    print("Charts saved to visualizations/output/")
