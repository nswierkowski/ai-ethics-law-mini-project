"""
src/ft_evaluation.py
────────────────────
Visualisation helpers that consume TrainingResult objects and produce
publication-quality comparison figures for the notebook.
 
Functions
─────────
  plot_training_curves       – loss + val macro-F1 per epoch, all experiments
  plot_experiment_comparison – bar chart of test metrics across 4 runs
  plot_confusion_matrices    – grid of confusion matrices
  plot_per_class_f1_heatmap  – heatmap of per-class F1 per experiment
  results_to_dataframe       – summary DataFrame for tabular display
"""
 
from __future__ import annotations
 
from typing import Dict, List, Optional
 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
 
from .training_config import LABEL_NAMES, BINARY_LABEL_NAMES
from .eda_utils import set_style, PALETTE_FAIR, PALETTE_UNFAIR, CLASS_COLORS
 
EXP_COLORS = {
    "A_multiclass_baseline":  "#E63946",
    "B_multiclass_balanced":  "#2A9D8F",
    "C_binary_balanced":      "#457B9D",
    "D_multiclass_focal":     "#F4A261",
}
 
EXP_SHORT = {
    "A_multiclass_baseline":  "A: Baseline",
    "B_multiclass_balanced":  "B: Balanced",
    "C_binary_balanced":      "C: Binary",
    "D_multiclass_focal":     "D: Focal",
}
 
 
 
def plot_training_curves(results: Dict, figsize=(16, 10)) -> plt.Figure:
    """
    2-row × N-col grid.  Row 0 = train/val loss.  Row 1 = val macro-F1.
    One column per experiment.
    """
    set_style()
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=figsize, sharex=True)
    if n == 1:
        axes = axes.reshape(2, 1)
 
    for col, (exp_name, res) in enumerate(results.items()):
        epochs = range(1, len(res.train_losses) + 1)
        color  = EXP_COLORS.get(exp_name, "#333")
        short  = EXP_SHORT.get(exp_name, exp_name)
 
        # Loss
        ax_loss = axes[0, col]
        ax_loss.plot(epochs, res.train_losses, "o--", color=color,
                     alpha=0.7, linewidth=1.8, markersize=5, label="Train")
        ax_loss.plot(epochs, res.val_losses,   "o-",  color=color,
                     linewidth=2.2, markersize=6, label="Val")
        ax_loss.axvline(res.best_epoch, color="#aaa", ls=":", lw=1.2)
        ax_loss.set_title(short, pad=8, fontsize=11, color=color, fontweight="bold")
        ax_loss.set_ylabel("Loss" if col == 0 else "")
        ax_loss.legend(fontsize=8)
 
        # Val macro-F1
        ax_f1 = axes[1, col]
        ax_f1.plot(epochs, res.val_macro_f1s, "s-", color=color,
                   linewidth=2.2, markersize=6)
        ax_f1.axvline(res.best_epoch, color="#aaa", ls=":", lw=1.2,
                      label=f"Best epoch {res.best_epoch}")
        ax_f1.axhline(res.best_val_macro_f1, color=color, ls="--",
                      lw=1, alpha=0.5)
        ax_f1.set_xlabel("Epoch")
        ax_f1.set_ylabel("Val Macro-F1" if col == 0 else "")
        ax_f1.set_ylim(0, 1)
        ax_f1.legend(fontsize=8)
 
    fig.suptitle("Training Curves — All Experiments", fontsize=14, y=1.01)
    fig.tight_layout(pad=2)
    return fig
 
 
 
def plot_experiment_comparison(results: Dict, figsize=(13, 5)) -> plt.Figure:
    """
    Grouped bar chart: accuracy, macro-F1, weighted-F1 for each experiment.
    """
    set_style()
    metrics  = ["test_accuracy", "test_macro_f1", "test_weighted_f1"]
    labels   = ["Accuracy", "Macro-F1", "Weighted-F1"]
    exp_names = list(results.keys())
    x = np.arange(len(exp_names))
    n_metrics = len(metrics)
    width = 0.22
 
    fig, ax = plt.subplots(figsize=figsize)
    for i, (metric, label) in enumerate(zip(metrics, labels)):
        vals = [getattr(results[e], metric) for e in exp_names]
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width,
                      label=label,
                      color=CLASS_COLORS[i],
                      edgecolor="white", linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)
 
    ax.set_xticks(x)
    ax.set_xticklabels([EXP_SHORT.get(e, e) for e in exp_names], fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Test-Set Performance — Experiment Comparison", pad=12)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout(pad=2)
    return fig
 
 
 
def plot_confusion_matrices(results: Dict, figsize=(18, 14)) -> plt.Figure:
    """One normalised confusion matrix per experiment, in a 2×2 grid."""
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
 
    for idx, (exp_name, res) in enumerate(results.items()):
        ax = axes[idx]
        cm = res.confusion_mat.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm  = np.divide(cm, row_sums, where=row_sums != 0)
 
        label_names = list(res.cfg.label_names.values())
        short_names = [n[:12] for n in label_names]
 
        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues",
                       vmin=0, vmax=1)
        ax.set_title(EXP_SHORT.get(exp_name, exp_name),
                     pad=8, fontweight="bold",
                     color=EXP_COLORS.get(exp_name, "#333"))
        ax.set_xticks(range(len(short_names)))
        ax.set_yticks(range(len(short_names)))
        ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(short_names, fontsize=7)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
 
        for r in range(cm_norm.shape[0]):
            for c in range(cm_norm.shape[1]):
                v = cm_norm[r, c]
                ax.text(c, r, f"{v:.2f}",
                        ha="center", va="center", fontsize=6.5,
                        color="white" if v > 0.55 else "#333")
 
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
 
    fig.suptitle("Normalised Confusion Matrices (row = true class)", fontsize=14, y=1.01)
    fig.tight_layout(pad=2)
    return fig
 
 
 
def plot_per_class_f1_heatmap(results: Dict, figsize=(13, 5)) -> plt.Figure:
    """
    Heatmap rows = experiments, cols = classes.
    Only multiclass experiments shown (binary has different label space).
    """
    set_style()
    # Filter to multiclass
    mc_results = {k: v for k, v in results.items() if not v.cfg.is_binary}
 
    if not mc_results:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No multiclass results", ha="center", va="center")
        return fig
 
    class_names = list(LABEL_NAMES.values())
    rows, row_labels = [], []
    for exp_name, res in mc_results.items():
        row = [res.per_class_f1.get(cn, 0.0) for cn in class_names]
        rows.append(row)
        row_labels.append(EXP_SHORT.get(exp_name, exp_name))
 
    mat = np.array(rows)
 
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
 
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels([n[:14] for n in class_names], rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)
 
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            v = mat[r, c]
            ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if v < 0.3 or v > 0.75 else "#222")
 
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="F1 Score")
    ax.set_title("Per-Class F1 Score by Experiment (multiclass only)", pad=12)
    fig.tight_layout(pad=2)
    return fig
 
  
def results_to_dataframe(results: Dict) -> pd.DataFrame:
    rows = []
    for exp_name, res in results.items():
        rows.append({
            "Experiment":        EXP_SHORT.get(exp_name, exp_name),
            "Task":              res.cfg.task,
            "Imbalance Handled": res.cfg.handle_imbalance,
            "Loss":              "Focal" if res.cfg.use_focal_loss else "CrossEntropy",
            "Downsampled":       res.cfg.downsample_ratio is not None,
            "Best Val Macro-F1": round(res.best_val_macro_f1, 4),
            "Best Epoch":        res.best_epoch,
            "Early Stopped":     getattr(res, "early_stopped", False),
            "Epochs Trained":    getattr(res, "epochs_trained", res.best_epoch),
            "Test Accuracy":     round(res.test_accuracy, 4),
            "Test Macro-F1":     round(res.test_macro_f1, 4),
            "Test Weighted-F1":  round(res.test_weighted_f1, 4),
            "Train Time (min)":  round(res.training_time_s / 60, 1),
        })
    return pd.DataFrame(rows)
 