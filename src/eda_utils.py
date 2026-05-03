"""
src/eda_utils.py
────────────────
Reusable EDA helper functions for the ToS Shield project.
All functions accept a pd.DataFrame with columns:
    text, label, label_name, label_abbr, is_unfair,
    text_length, word_count, sentence_count
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec



PALETTE_UNFAIR = "#E63946"   # vivid red  -> unfair clauses
PALETTE_FAIR   = "#2A9D8F"   # teal       -> fair / OK
PALETTE_NEUTRAL = "#457B9D"  # steel blue -> neutral metrics

CLASS_COLORS = [
    "#E63946", "#F4A261", "#E9C46A", "#264653", "#2A9D8F",
    "#A8DADC", "#457B9D", "#1D3557", "#8338EC", "#06D6A0",
]


def compute_class_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-class count, percentage, and fair/unfair flag."""
    stats = (
        df.groupby(["label", "label_name", "label_abbr"])
        .agg(count=("text", "count"))
        .reset_index()
        .sort_values("label")
    )
    stats["pct"] = stats["count"] / stats["count"].sum() * 100
    stats["is_unfair"] = stats["label"] < 9
    return stats


def compute_imbalance_ratio(df: pd.DataFrame) -> float:
    """Majority-class count / minority-class count."""
    vc = df["label"].value_counts()
    return vc.max() / vc.min()


def compute_text_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-class descriptive stats for text_length and word_count."""
    return (
        df.groupby("label_name")[["text_length", "word_count", "sentence_count"]]
        .agg(["mean", "median", "std", "min", "max"])
        .round(1)
    )


def set_style() -> None:
    """Apply a clean, publication-ready matplotlib style."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#F8F9FA",
            "axes.edgecolor": "#DEE2E6",
            "axes.grid": True,
            "grid.color": "#E9ECEF",
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 15,
            "figure.titleweight": "bold",
        }
    )


def plot_class_distribution(
    stats: pd.DataFrame,
    title: str = "Class Distribution – unfair_tos",
    figsize: Tuple[int, int] = (14, 5),
) -> plt.Figure:
    """
    Dual-panel: absolute counts (bar) + fair vs unfair binary split (pie).
    """
    set_style()
    fig, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=figsize,
                                          gridspec_kw={"width_ratios": [2.5, 1]})

    colors = [PALETTE_UNFAIR if u else PALETTE_FAIR
              for u in stats["is_unfair"]]
    bars = ax_bar.barh(
        stats["label_name"],
        stats["count"],
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        height=0.7,
    )
    ax_bar.set_xlabel("Number of Samples", labelpad=8)
    ax_bar.set_title(title, pad=12)
    ax_bar.invert_yaxis()

    for bar, pct in zip(bars, stats["pct"]):
        w = bar.get_width()
        ax_bar.text(
            w + max(stats["count"]) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{w:,}  ({pct:.1f}%)",
            va="center",
            fontsize=8.5,
            color="#333",
        )
    ax_bar.set_xlim(0, stats["count"].max() * 1.25)

    ax_bar.legend(
        handles=[
            mpatches.Patch(color=PALETTE_UNFAIR, label="Unfair (labels 0–8)"),
            mpatches.Patch(color=PALETTE_FAIR,   label="Fair / OK (label 9)"),
        ],
        loc="lower right",
        framealpha=0.9,
    )

    fair_total   = stats.loc[~stats["is_unfair"], "count"].sum()
    unfair_total = stats.loc[stats["is_unfair"],  "count"].sum()
    pie_vals  = [fair_total, unfair_total]
    pie_labels = [f"Fair\n{fair_total:,}", f"Unfair\n{unfair_total:,}"]
    pie_colors = [PALETTE_FAIR, PALETTE_UNFAIR]

    wedges, texts, autotexts = ax_pie.pie(
        pie_vals,
        labels=pie_labels,
        colors=pie_colors,
        autopct="%1.1f%%",
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        pctdistance=0.75,
    )
    for t in autotexts:
        t.set_fontsize(10)
        t.set_color("white")
        t.set_weight("bold")
    ax_pie.set_title("Binary Split", pad=12)

    fig.tight_layout(pad=2)
    return fig


def plot_text_length_distribution(
    df: pd.DataFrame,
    col: str = "word_count",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5),
) -> plt.Figure:
    """
    Overlapping KDE + rug for text-length metrics, split by fair/unfair.
    """
    from scipy.stats import gaussian_kde  

    set_style()
    if title is None:
        title = f"Distribution of `{col}` by Fairness Label"

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, group_col, group_label, color in [
        (axes[0], 0, "Fair (OK)",  PALETTE_FAIR),
        (axes[1], 1, "Unfair",     PALETTE_UNFAIR),
    ]:
        subset = df[df["is_unfair"] == group_col][col].dropna()
        subset = subset[subset < subset.quantile(0.99)]  

        if len(subset) > 5:
            kde = gaussian_kde(subset, bw_method="scott")
            x = np.linspace(subset.min(), subset.max(), 300)
            ax.fill_between(x, kde(x), alpha=0.35, color=color)
            ax.plot(x, kde(x), color=color, lw=2)

        ax.axvline(subset.median(), color="#333", ls="--", lw=1.2,
                   label=f"Median: {subset.median():.0f}")
        ax.axvline(subset.mean(),   color="#888", ls=":",  lw=1.2,
                   label=f"Mean:   {subset.mean():.0f}")
        ax.set_xlabel(col.replace("_", " ").title())
        ax.set_ylabel("Density")
        ax.set_title(f"{group_label} – {col.replace('_', ' ')}")
        ax.legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout(pad=2)
    return fig


def plot_unfair_breakdown(
    stats: pd.DataFrame,
    figsize: Tuple[int, int] = (9, 5),
) -> plt.Figure:
    """Stacked bar showing the composition of unfair clauses only."""
    set_style()
    unfair = stats[stats["is_unfair"]].copy()

    fig, ax = plt.subplots(figsize=figsize)
    colors = CLASS_COLORS[: len(unfair)]
    ax.bar(
        unfair["label_name"],
        unfair["count"],
        color=colors,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.set_xlabel("Unfair Clause Type")
    ax.set_ylabel("Sample Count")
    ax.set_title("Distribution Among Unfair Clause Types", pad=12)
    plt.xticks(rotation=35, ha="right")

    for i, (_, row) in enumerate(unfair.iterrows()):
        ax.text(i, row["count"] + 5, str(int(row["count"])),
                ha="center", va="bottom", fontsize=8, color="#333")

    fig.tight_layout(pad=2)
    return fig


def plot_text_stats_heatmap(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """Heatmap of mean word-count, char-length, sentence-count per class."""
    set_style()
    pivot = (
        df.groupby("label_name")[["word_count", "text_length", "sentence_count"]]
        .mean()
        .round(1)
    )

    norm = (pivot - pivot.min()) / (pivot.max() - pivot.min() + 1e-9)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(norm.T.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)

    ax.set_xticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.index, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.columns)))
    ax.set_yticklabels(["Word Count", "Char Length", "Sentence Count"], fontsize=9)

    for r, metric in enumerate(pivot.columns):
        for c, cls in enumerate(pivot.index):
            val = pivot.loc[cls, metric]
            ax.text(c, r, f"{val:.0f}", ha="center", va="center",
                    fontsize=7.5, color="black" if norm.loc[cls, metric] < 0.7 else "white")

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.03, label="Normalised value")
    ax.set_title("Text-Length Metrics by Class (mean)", pad=12)
    fig.tight_layout(pad=2)
    return fig


def plot_sample_balance_splits(
    splits_dict: dict,
    figsize: Tuple[int, int] = (12, 4),
) -> plt.Figure:
    """
    Grouped bars showing fair vs unfair counts across train/val/test.
    """
    set_style()
    split_names = list(splits_dict.keys())
    fair_counts   = [splits_dict[s].df["is_unfair"].eq(0).sum() for s in split_names]
    unfair_counts = [splits_dict[s].df["is_unfair"].eq(1).sum() for s in split_names]

    x = np.arange(len(split_names))
    w = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    b1 = ax.bar(x - w / 2, fair_counts,   w, label="Fair",   color=PALETTE_FAIR,   edgecolor="white")
    b2 = ax.bar(x + w / 2, unfair_counts, w, label="Unfair", color=PALETTE_UNFAIR, edgecolor="white")

    for bar in [*b1, *b2]:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 30,
                f"{bar.get_height():,}",
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in split_names])
    ax.set_ylabel("Sample Count")
    ax.set_title("Fair vs Unfair Samples Across Dataset Splits", pad=12)
    ax.legend()
    fig.tight_layout(pad=2)
    return fig


def downsample_majority_class(
    df: pd.DataFrame,
    target_ratio: float = 2.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Downsample the Fair (label=9) class so that:
        n_fair ≤ target_ratio × n_unfair_total

    Parameters
    ----------
    target_ratio : float
        How many fair samples to keep per unfair sample.
        Default 2.0 → balanced but retains some majority signal.

    Returns
    -------
    Balanced DataFrame (shuffled).
    """
    unfair_df = df[df["is_unfair"] == 1]
    fair_df   = df[df["is_unfair"] == 0]

    n_keep = int(len(unfair_df) * target_ratio)
    n_keep = min(n_keep, len(fair_df))

    fair_downsampled = fair_df.sample(n=n_keep, random_state=random_state)
    result = pd.concat([unfair_df, fair_downsampled]).sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)
    return result


def get_top_ngrams(
    texts: pd.Series,
    n: int = 2,
    top_k: int = 20,
    stop_words: Optional[List[str]] = None,
) -> List[Tuple[str, int]]:
    """Return the top-k n-grams from a text series."""
    import re

    DEFAULT_STOPS = {
        "the", "a", "an", "of", "to", "in", "and", "or", "for",
        "we", "you", "your", "our", "is", "are", "may", "will",
        "this", "that", "with", "by", "at", "be", "any",
    }
    stops = set(stop_words or []) | DEFAULT_STOPS

    counter: Counter = Counter()
    for text in texts.dropna():
        tokens = re.findall(r"\b[a-z]{2,}\b", text.lower())
        tokens = [t for t in tokens if t not in stops]
        if n == 1:
            counter.update(tokens)
        else:
            for i in range(len(tokens) - n + 1):
                counter[" ".join(tokens[i: i + n])] += 1
    return counter.most_common(top_k)


def plot_top_ngrams(
    df: pd.DataFrame,
    n: int = 2,
    top_k: int = 20,
    figsize: Tuple[int, int] = (14, 6),
) -> plt.Figure:
    """Side-by-side top-n-gram charts for fair vs unfair clauses."""
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, is_unfair_flag, label, color in [
        (axes[0], 0, "Fair Clauses",   PALETTE_FAIR),
        (axes[1], 1, "Unfair Clauses", PALETTE_UNFAIR),
    ]:
        subset = df[df["is_unfair"] == is_unfair_flag]["text"]
        ngrams = get_top_ngrams(subset, n=n, top_k=top_k)
        if not ngrams:
            ax.set_title(f"No data – {label}")
            continue
        terms, counts = zip(*ngrams)
        y_pos = range(len(terms))
        ax.barh(y_pos, counts, color=color, alpha=0.85, edgecolor="white")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(terms, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Frequency")
        ax.set_title(f"Top-{top_k} {n}-grams\n{label}", pad=10)

    fig.suptitle(f"N-gram Analysis (n={n}): Fair vs Unfair", y=1.01)
    fig.tight_layout(pad=2)
    return fig
