"""
src/imbalance.py
────────────────
Quantifies class imbalance and provides strategies for handling it.
Used in the EDA notebook to justify our downsampling decision.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .eda_utils import PALETTE_FAIR, PALETTE_UNFAIR, set_style


def imbalance_report(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """
    Comprehensive imbalance metrics per class.

    Returns a DataFrame with:
        - count, pct
        - IR  (Imbalance Ratio vs majority class)
        - strategy recommendation
    """
    vc = df[label_col].value_counts().sort_index()
    total = vc.sum()
    majority = vc.max()

    records = []
    for lbl, cnt in vc.items():
        ir = majority / cnt
        if ir <= 1.5:
            strategy = "None needed"
        elif ir <= 5:
            strategy = "Light downsampling / class weights"
        elif ir <= 20:
            strategy = "Downsampling majority + oversampling minority"
        else:
            strategy = "Aggressive resampling or synthetic data (SMOTE)"

        records.append(
            {
                "label": lbl,
                "label_name": df[df[label_col] == lbl]["label_name"].iloc[0]
                if "label_name" in df.columns
                else str(lbl),
                "count": cnt,
                "pct": round(cnt / total * 100, 2),
                "IR_vs_majority": round(ir, 1),
                "strategy": strategy,
            }
        )
    return pd.DataFrame(records)


def effective_number_of_samples(df: pd.DataFrame, beta: float = 0.9999) -> pd.Series:
    """
    Class-Balanced Loss effective number of samples per class.
    E_n = (1 - beta^n) / (1 - beta)
    Reference: Cui et al. (2019), CVPR.
    """
    counts = df["label"].value_counts().sort_index()
    en = (1.0 - beta ** counts) / (1.0 - beta)
    weights = 1.0 / en
    weights = weights / weights.sum() * len(counts)
    return weights.round(4)


def plot_imbalance_summary(
    report_df: pd.DataFrame,
    figsize: Tuple[int, int] = (13, 5),
) -> plt.Figure:
    """
    Two panels:
        Left  – Imbalance Ratio (IR) bar chart
        Right – Class-balanced weight recommendation
    """
    set_style()
    fig, (ax_ir, ax_w) = plt.subplots(1, 2, figsize=figsize)

    colors = [PALETTE_FAIR if ir <= 1.5 else PALETTE_UNFAIR
              for ir in report_df["IR_vs_majority"]]
    ax_ir.barh(
        report_df["label_name"],
        report_df["IR_vs_majority"],
        color=colors, edgecolor="white", height=0.65,
    )
    ax_ir.axvline(1, color="#555", ls="--", lw=1.2, label="Perfectly balanced")
    ax_ir.set_xlabel("Imbalance Ratio (majority / class)")
    ax_ir.set_title("Class Imbalance Ratios", pad=10)
    ax_ir.invert_yaxis()
    ax_ir.legend(fontsize=8)

    for i, row in enumerate(report_df.itertuples()):
        ax_ir.text(
            row.IR_vs_majority + 0.3, i,
            f"×{row.IR_vs_majority:.1f}",
            va="center", fontsize=8, color="#333",
        )

    strategies = report_df["strategy"].value_counts()
    ax_w.pie(
        strategies.values,
        labels=strategies.index,
        autopct="%1.0f%%",
        startangle=90,
        colors=["#2A9D8F", "#E63946", "#F4A261", "#457B9D"][: len(strategies)],
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    ax_w.set_title("Recommended Strategy Breakdown", pad=10)

    fig.suptitle("Class Imbalance Analysis – unfair_tos", y=1.01)
    fig.tight_layout(pad=2)
    return fig


def compute_class_weights_sklearn(df: pd.DataFrame) -> Dict[int, float]:
    """
    Compute balanced class weights using sklearn formula:
    w_j = n_samples / (n_classes * n_samples_j)
    Ready to pass directly to HuggingFace Trainer's `class_weight` arg.
    """
    from collections import Counter

    counts = Counter(df["label"])
    n_samples = len(df)
    n_classes = len(counts)
    return {
        cls: round(n_samples / (n_classes * cnt), 4)
        for cls, cnt in sorted(counts.items())
    }


def simulate_resampling(
    df: pd.DataFrame,
    strategy: str = "downsample",
    target_ratio: float = 2.0,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Simulate a resampling strategy and return the new DataFrame + stats.

    Parameters
    ----------
    strategy : 'downsample' | 'oversample' | 'combined'
    target_ratio : float
        For 'downsample': n_fair = target_ratio × n_unfair
        For 'oversample': n_minority = 1/target_ratio × n_majority
    """
    rng = np.random.default_rng(random_state)

    unfair = df[df["is_unfair"] == 1]
    fair   = df[df["is_unfair"] == 0]

    if strategy == "downsample":
        n_keep = int(len(unfair) * target_ratio)
        n_keep = min(n_keep, len(fair))
        fair_resampled = fair.sample(n=n_keep, random_state=random_state)
        result = pd.concat([unfair, fair_resampled])

    elif strategy == "oversample":
        n_target = int(len(fair) / target_ratio)
        unfair_resampled = unfair.sample(n=n_target, replace=True,
                                          random_state=random_state)
        result = pd.concat([fair, unfair_resampled])

    elif strategy == "combined":
        n_target = int((len(fair) + len(unfair)) / 2)
        fair_ds   = fair.sample(n=n_target, random_state=random_state)
        unfair_os = unfair.sample(n=n_target, replace=True,
                                   random_state=random_state)
        result = pd.concat([fair_ds, unfair_os])
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    result = result.sample(frac=1, random_state=random_state).reset_index(drop=True)
    stats = {
        "total": len(result),
        "fair": result["is_unfair"].eq(0).sum(),
        "unfair": result["is_unfair"].eq(1).sum(),
        "ir_after": round(result["is_unfair"].eq(0).sum() /
                          max(result["is_unfair"].eq(1).sum(), 1), 2),
    }
    return result, stats
