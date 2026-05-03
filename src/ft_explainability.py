"""
src/ft_explainability.py
────────────────────────
Model explainability for the fine-tuned DistilBERT classifiers.

Three complementary lenses
──────────────────────────
1. Attention Rollout
   – Aggregates multi-head attention weights across all layers using
     the "rollout" method (Abnar & Zuidema, 2020) to produce a single
     importance score per input token.

2. Integrated Gradients  (via captum, falls back gracefully if absent)
   – Attribution of each token embedding dimension, collapsed to an L2
     norm per token. Model-agnostic, faithful to the computation graph.

3. SHAP KernelExplainer  (via shap, falls back gracefully if absent)
   – Treats the model as a black box. Produces Shapley values for each
     token position in a sentence. Most expensive but most theoretically
     principled.

All three return a standard TokenImportance namedtuple so downstream
visualisation code stays the same regardless of which method was used.

Plot helpers
────────────
  plot_token_heatmap         – colour-coded token bar chart
  plot_attention_rollout_map – 2-D attention head map for a single layer
  plot_shap_summary          – beeswarm or bar summary across a corpus sample
  plot_top_important_tokens  – global most-important tokens across N samples
"""

from __future__ import annotations

import re
import warnings
from collections import defaultdict
from typing import Dict, List, NamedTuple, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from .training_config import ExperimentConfig

try:
    import torch
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from contextlib import contextmanager

@contextmanager
def _nullctx():
    yield

def _to_np_safe(t):
    """Convert tensor or NumpyTensor stub to numpy array."""
    if hasattr(t, '_arr'):
        return np.array(t._arr)
    if hasattr(t, 'cpu'):
        t = t.cpu()
    if hasattr(t, 'numpy'):
        return t.numpy()
    return np.array(t)


class TokenImportance(NamedTuple):
    tokens:     List[str]    
    scores:     np.ndarray   
    method:     str          # "attention_rollout" | "integrated_gradients" | "shap"
    pred_label: int
    pred_prob:  float
    true_label: Optional[int] = None


def attention_rollout(
    model,
    tokenizer,
    text: str,
    true_label: Optional[int] = None,
    device: str = "cpu",
) -> TokenImportance:
    """
    Compute attention rollout scores for each token in `text`.

    Algorithm (Abnar & Zuidema, 2020):
        R_0 = I  (identity)
        R_l = R_{l-1} · (0.5*A_l + 0.5*I)
    where A_l is the mean attention matrix at layer l.
    The final score for each token is R[-1][0, 1:-1] (CLS row, skip special tokens).

    Parameters
    ----------
    model     : fine-tuned DistilBERT model (with output_attentions=True)
    tokenizer : matching AutoTokenizer
    text      : raw input clause text
    """
    model.eval()
    enc = tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=128,
        padding=False,
    ).to(device)

    def _to_np(t):
        if hasattr(t, 'cpu'):    t = t.cpu()
        if hasattr(t, 'detach'): t = t.detach()
        if hasattr(t, 'numpy'):  return t.numpy()
        return np.array(t)

    # ── Forward pass with output_attentions=True ──────────────────────────────
    # HuggingFace ≥ 4.36 defaults to attn_implementation="sdpa" (fused CUDA
    # kernel) which never materialises the attention matrix — out.attentions
    # is always None in that mode. We must switch the model to "eager" (the
    # original pure-PyTorch path) before asking for attention weights, then
    # restore whatever was there before so normal inference is unaffected.
    cfg = getattr(model, "config", None)
    _orig_attn_impl = getattr(cfg, "_attn_implementation", "eager")
    _orig_out_attn  = getattr(cfg, "output_attentions",    False)

    if cfg is not None:
        cfg._attn_implementation = "eager"
        cfg.output_attentions    = True

    _ctx = torch.no_grad() if _TORCH_AVAILABLE else _nullctx()
    with _ctx:
        out = model(
            input_ids         = enc["input_ids"],
            attention_mask    = enc["attention_mask"],
            output_attentions = True,
        )

    if cfg is not None:
        cfg._attn_implementation = _orig_attn_impl
        cfg.output_attentions    = _orig_out_attn

    attentions = out.attentions

    if attentions is None:
        raise RuntimeError(
            "out.attentions is still None after switching to eager attention.\n"
            "Verify the model is a DistilBERT/BERT variant loaded via "
            "AutoModelForSequenceClassification."
        )

    seq_len  = enc["input_ids"].shape[1]
    avg_attn = [_to_np(a).squeeze(0).mean(0) for a in attentions]

    R = np.eye(seq_len)
    for A in avg_attn:
        A_aug = 0.5 * A + 0.5 * np.eye(seq_len)
        A_aug = A_aug / (A_aug.sum(axis=-1, keepdims=True) + 1e-9)
        R = R @ A_aug

    cls_scores = R[0, 1:-1]   
    cls_scores = cls_scores / (cls_scores.max() + 1e-9)

    token_ids = enc["input_ids"].squeeze(0).cpu().tolist()
    tokens    = tokenizer.convert_ids_to_tokens(token_ids)[1:-1]

    logits_np = _to_np(out.logits).squeeze(0)
    e = np.exp(logits_np - logits_np.max())
    probs = e / e.sum()
    pred_lbl  = int(probs.argmax())
    pred_prob = float(probs.max())

    return TokenImportance(
        tokens=tokens,
        scores=cls_scores,
        method="attention_rollout",
        pred_label=pred_lbl,
        pred_prob=pred_prob,
        true_label=true_label,
    )



def integrated_gradients(
    model,
    tokenizer,
    text: str,
    true_label: Optional[int] = None,
    n_steps: int = 50,
    device: str = "cpu",
) -> TokenImportance:
    """
    Token-level attribution via Integrated Gradients (Sundararajan et al., 2017).
    Uses captum if available; falls back to a single-step gradient approximation.

    The attribution for each token is the L2 norm of its embedding gradient.
    """
    model.eval()
    enc = tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=128, padding=False,
    ).to(device)

    input_ids      = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    token_ids_list = input_ids.squeeze(0).cpu().tolist()
    tokens         = tokenizer.convert_ids_to_tokens(token_ids_list)[1:-1]

    _ctx2 = torch.no_grad() if _TORCH_AVAILABLE else _nullctx()
    with _ctx2:
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits_np2 = _to_np_safe(out.logits).squeeze(0) if hasattr(out.logits, 'numpy') or hasattr(out.logits, 'cpu') else np.array(out.logits._arr).squeeze(0)
    e2 = np.exp(logits_np2 - logits_np2.max())
    probs = e2 / e2.sum()
    pred_lbl  = int(probs.argmax())
    pred_prob = float(probs.max())
    target    = pred_lbl

    try:
        from captum.attr import LayerIntegratedGradients  # type: ignore

        def forward_fn(inp_ids):
            return model(input_ids=inp_ids, attention_mask=attention_mask).logits

        lig = LayerIntegratedGradients(forward_fn, model.distilbert.embeddings)
        attrs, _ = lig.attribute(
            inputs=input_ids,
            baselines=torch.zeros_like(input_ids),
            target=target,
            n_steps=n_steps,
            return_convergence_delta=True,
        )
        scores = attrs.squeeze(0).norm(dim=-1).detach().cpu().numpy()
        scores = scores[1:-1] 

    except ImportError:
        baseline_prob = probs[target]
        scores_list   = []

        for i in range(len(tokens)):
            masked_text = " ".join(
                "[MASK]" if j == i else t
                for j, t in enumerate(tokens)
            )
            enc_m = tokenizer(masked_text, return_tensors="pt",
                              truncation=True, max_length=128, padding=False)
            _ctx5 = torch.no_grad() if _TORCH_AVAILABLE else _nullctx()
            with _ctx5:
                out_m = model(**dict(enc_m.items()))
            lg_m = _to_np_safe(out_m.logits).squeeze(0)
            e_m  = np.exp(lg_m - lg_m.max())
            p_m  = (e_m / e_m.sum())[target]
            scores_list.append(float(baseline_prob - p_m))

        scores = np.array(scores_list, dtype=np.float32)

    scores = scores / (scores.max() + 1e-9)
    return TokenImportance(
        tokens=tokens,
        scores=scores,
        method="integrated_gradients",
        pred_label=pred_lbl,
        pred_prob=pred_prob,
        true_label=true_label,
    )

def shap_token_importance(
    model,
    tokenizer,
    text: str,
    true_label: Optional[int] = None,
    n_samples: int = 64,
    device: str = "cpu",
) -> TokenImportance:
    """
    Black-box SHAP / LOO explanation at the token level.

    Each "feature" is one content token (1 = present, 0 = replaced with [MASK]).
    Uses shap.KernelExplainer when the `shap` package is installed; otherwise
    falls back to Leave-One-Out (LOO) occlusion which is equivalent for short
    sequences and requires no extra dependencies.

    All tensors are explicitly moved to `device` so this works on CUDA.
    """
    model.eval()

    enc          = tokenizer(text, return_tensors="pt",
                             truncation=True, max_length=128, padding=False)
    input_ids_np = enc["input_ids"].squeeze(0).numpy()      # (seq,)  CPU numpy
    attn_mask_np = enc["attention_mask"].squeeze(0).numpy() # (seq,)  CPU numpy
    tokens       = tokenizer.convert_ids_to_tokens(input_ids_np.tolist())[1:-1]
    n_content    = len(tokens)
    mask_id      = tokenizer.mask_token_id or tokenizer.pad_token_id

    def _run(ids_np: np.ndarray) -> np.ndarray:
        """ids_np: (seq,) int array → softmax probs (num_labels,)"""
        if _TORCH_AVAILABLE:
            ids_t  = torch.tensor(ids_np,  dtype=torch.long).unsqueeze(0).to(device)
            mask_t = torch.tensor(attn_mask_np, dtype=torch.long).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(input_ids=ids_t, attention_mask=mask_t)
            lg = out.logits.squeeze(0).float().cpu().numpy()
        else:
            out = model(
                input_ids      = ids_np[np.newaxis, :],
                attention_mask = attn_mask_np[np.newaxis, :],
            )
            lg = _to_np_safe(out.logits).squeeze(0)
        e = np.exp(lg - lg.max())
        return e / e.sum()

    probs_orig = _run(input_ids_np)
    pred_lbl   = int(probs_orig.argmax())
    pred_prob  = float(probs_orig.max())

    def predict_fn(mask_matrix: np.ndarray) -> np.ndarray:
        """
        mask_matrix: (n_instances, n_content_tokens) — 1=keep, 0=mask
        Returns:     (n_instances, num_labels) softmax probabilities
        """
        results = []
        for row in mask_matrix:
            ids = input_ids_np.copy()
            for i, keep in enumerate(row):
                if keep == 0:
                    ids[i + 1] = mask_id  
            results.append(_run(ids))
        return np.array(results)

    method_used = "loo"
    try:
        import shap 

        background = np.zeros((1, n_content))
        explainer  = shap.KernelExplainer(predict_fn, background)
        instance   = np.ones((1, n_content))
        shap_vals  = explainer.shap_values(instance, nsamples=n_samples, silent=True)

        # shap_vals is either:
        #   list[ndarray]  — one (1, n_content) array per class  (older shap)
        #   ndarray        — shape (n_content,) or (1, n_content) (newer shap)
        if isinstance(shap_vals, list):
            sv = np.array(shap_vals[pred_lbl])
        else:
            sv = np.array(shap_vals)

        sv = sv.squeeze() 

        # Sanity-check: if shape still doesn't match n_content, fall through to LOO
        if sv.shape == (n_content,):
            scores     = sv
            method_used = "shap_kernel"
        else:
            raise ValueError(
                f"Unexpected shap_values shape {sv.shape}, expected ({n_content},). "
                "Falling back to LOO."
            )

    except (ImportError, ValueError) as _shap_err:
        # ── Leave-one-out occlusion fallback ──────────────────────────────────
        # For each token position i: mask token i → measure drop in pred_lbl prob.
        # score[i] > 0  → token i supported the prediction (removing it hurt)
        # score[i] < 0  → token i hurt the prediction  (removing it helped)
        baseline_prob = probs_orig[pred_lbl]
        scores        = np.zeros(n_content, dtype=np.float32)
        for i in range(n_content):
            mask_row      = np.ones((1, n_content))
            mask_row[0, i] = 0
            p_masked      = predict_fn(mask_row)[0, pred_lbl]
            scores[i]     = baseline_prob - p_masked

    s_abs       = np.abs(scores)
    scores_norm = s_abs / (s_abs.max() + 1e-9)

    assert scores_norm.shape == (n_content,), (
        f"scores_norm shape {scores_norm.shape} != n_content {n_content}. "
        f"tokens={tokens[:5]}..."
    )

    return TokenImportance(
        tokens=tokens,
        scores=scores_norm,
        method=method_used,
        pred_label=pred_lbl,
        pred_prob=pred_prob,
        true_label=true_label,
    )

def plot_token_heatmap(
    ti: TokenImportance,
    label_names: Dict[int, str],
    title: Optional[str] = None,
    figsize: Tuple = (14, 2.2),
    max_tokens: int = 40,
) -> plt.Figure:
    """
    Horizontal bar chart where each bar = one token, coloured by importance.
    Green = low, Yellow = medium, Red = high importance.
    """
    tokens = ti.tokens[:max_tokens]
    scores = ti.scores[:max_tokens]

    cmap   = plt.cm.RdYlGn_r
    colors = [cmap(float(s)) for s in scores]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(range(len(tokens)), scores, color=colors, edgecolor="white", lw=0.5)

    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("Importance")
    ax.set_ylim(0, 1.15)
    ax.axhline(scores.mean(), color="#555", ls="--", lw=1, alpha=0.7,
               label=f"Mean={scores.mean():.2f}")
    ax.legend(fontsize=8)

    pred_name = label_names.get(ti.pred_label, str(ti.pred_label))
    true_name = label_names.get(ti.true_label, "?") if ti.true_label is not None else "?"
    correct   = "✓" if ti.pred_label == ti.true_label else "✗"
    default_title = (
        f"Token Importance ({ti.method})  |  "
        f"Pred: {pred_name} ({ti.pred_prob:.2%}) {correct}  True: {true_name}"
    )
    ax.set_title(title or default_title, pad=10, fontsize=10)
    fig.tight_layout()
    return fig


def plot_attention_head_map(
    model,
    tokenizer,
    text: str,
    layer: int = 5,
    device: str = "cpu",
    figsize: Tuple = (10, 8),
) -> plt.Figure:
    """
    Visualise the raw multi-head attention matrix for a chosen layer.
    Rows = query tokens, Cols = key tokens. One sub-plot per head.
    """
    model.eval()
    enc = tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=64, padding=False,
    ).to(device)

    cfg2 = getattr(model, "config", None)
    _orig2_impl    = getattr(cfg2, "_attn_implementation", "eager")
    _orig2_out     = getattr(cfg2, "output_attentions",    False)
    if cfg2 is not None:
        cfg2._attn_implementation = "eager"
        cfg2.output_attentions    = True

    _ctx6 = torch.no_grad() if _TORCH_AVAILABLE else _nullctx()
    with _ctx6:
        out = model(
            input_ids         = enc["input_ids"],
            attention_mask    = enc["attention_mask"],
            output_attentions = True,
        )

    if cfg2 is not None:
        cfg2._attn_implementation = _orig2_impl
        cfg2.output_attentions    = _orig2_out

    if out.attentions is None:
        raise RuntimeError(
            "out.attentions is still None — model does not expose attention weights."
        )

    attn_raw   = out.attentions[layer]
    attn_layer = _to_np_safe(attn_raw).squeeze(0)  # (n_heads, seq, seq)
    n_heads = attn_layer.shape[0]
    ids_raw = enc["input_ids"]
    if hasattr(ids_raw, '_arr'):
        ids_list = ids_raw._arr.squeeze().tolist()
    elif hasattr(ids_raw, 'cpu'):
        ids_list = ids_raw.squeeze(0).cpu().tolist()
    else:
        ids_list = np.array(ids_raw).squeeze().tolist()
    tokens  = tokenizer.convert_ids_to_tokens(ids_list)
    short_tokens = [t[:8] for t in tokens]

    cols = 2
    rows = (n_heads + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols,
        figsize=figsize,
        gridspec_kw={"hspace": 0.45, "wspace": 0.35},
    )
    axes = np.array(axes).flatten()

    seq = len(short_tokens)
    tick_step  = max(1, seq // 20)
    tick_pos   = list(range(0, seq, tick_step))
    tick_lbls  = [short_tokens[i] for i in tick_pos]

    for h in range(n_heads):
        ax  = axes[h]
        mat = attn_layer[h]
        im  = ax.imshow(mat, cmap="Blues", vmin=0, vmax=mat.max(), aspect="auto")

        ax.set_title(f"Head {h}", fontsize=11, fontweight="bold", pad=6)

        ax.set_xticks(tick_pos)
        ax.set_yticks(tick_pos)
        ax.set_xticklabels(tick_lbls, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(tick_lbls, fontsize=8)

        ax.set_xlabel("Key token",   fontsize=8, labelpad=3)
        ax.set_ylabel("Query token", fontsize=8, labelpad=3)

        for row_idx in range(seq):
            peak = int(mat[row_idx].argmax())
            ax.add_patch(plt.Rectangle(
                (peak - 0.5, row_idx - 0.5), 1, 1,
                linewidth=1.2, edgecolor="#E63946", facecolor="none",
            ))

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format="%.2f")

    for h in range(n_heads, len(axes)):
        axes[h].axis("off")

    fig.suptitle(
        (f"Attention Heads — Layer {layer}  |  "
         f"\"{text[:70]}{'...' if len(text) > 70 else ''}\""),
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    return fig


def plot_top_important_tokens(
    token_importances: List[TokenImportance],
    top_k: int = 25,
    figsize: Tuple = (13, 5),
) -> plt.Figure:
    """
    Aggregates token importance scores across many examples.
    Shows the globally most influential tokens for UNFAIR vs FAIR predictions.
    """
    from .eda_utils import set_style, PALETTE_UNFAIR, PALETTE_FAIR

    set_style()
    unfair_scores: Dict[str, List[float]] = defaultdict(list)
    fair_scores:   Dict[str, List[float]] = defaultdict(list)

    for ti in token_importances:
        for tok, score in zip(ti.tokens, ti.scores):
            clean = re.sub(r"^##", "", tok).lower().strip()
            if len(clean) < 2 or clean in ("[cls]", "[sep]", "[pad]"):
                continue
            bucket = unfair_scores if ti.pred_label != 9 else fair_scores
            bucket[clean].append(float(score))

    def top_mean(d, k):
        means = {t: np.mean(v) for t, v in d.items() if len(v) >= 2}
        return sorted(means.items(), key=lambda x: -x[1])[:k]

    unfair_top = top_mean(unfair_scores, top_k)
    fair_top   = top_mean(fair_scores,   top_k)

    fig, (ax_u, ax_f) = plt.subplots(1, 2, figsize=figsize)

    for ax, top_list, color, group in [
        (ax_u, unfair_top, PALETTE_UNFAIR, "Unfair Predictions"),
        (ax_f, fair_top,   PALETTE_FAIR,   "Fair Predictions"),
    ]:
        if not top_list:
            ax.set_title(f"No data – {group}")
            continue
        terms, vals = zip(*top_list)
        y = range(len(terms))
        ax.barh(y, vals, color=color, alpha=0.85, edgecolor="white")
        ax.set_yticks(y)
        ax.set_yticklabels(terms, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Mean Importance")
        ax.set_title(f"Top-{top_k} Tokens\n{group}", pad=8)
        ax.set_xlim(0, 1)

    fig.suptitle("Global Token Importance — Most Influential Words", y=1.01)
    fig.tight_layout(pad=2)
    return fig


def plot_comparative_explanations(
    results: Dict,
    sample_texts: List[str],
    sample_labels: List[int],
    label_names: Dict[int, str],
    method: str = "attention_rollout",
    figsize: Tuple = (16, 10),
) -> plt.Figure:
    """
    Side-by-side token importance for the same input across multiple models.
    Rows = samples, Cols = experiments.
    """
    from .eda_utils import set_style
    set_style()

    n_samples = min(len(sample_texts), 3)
    n_models  = len(results)
    fig, axes = plt.subplots(n_samples, n_models,
                             figsize=figsize,
                             squeeze=False)

    dispatch = {
        "attention_rollout":   attention_rollout,
        "integrated_gradients": integrated_gradients,
        "shap":                shap_token_importance,
    }
    explain_fn = dispatch.get(method, attention_rollout)

    cmap = plt.cm.RdYlGn_r

    for row, (text, true_lbl) in enumerate(zip(sample_texts[:n_samples],
                                               sample_labels[:n_samples])):
        for col, (exp_name, res) in enumerate(results.items()):
            ax = axes[row, col]
            try:
                ti = explain_fn(
                    res.model, res.tokenizer, text,
                    true_label=true_lbl,
                    device="cpu",
                )
                toks   = ti.tokens[:30]
                scores = ti.scores[:30]
                colors = [cmap(float(s)) for s in scores]
                ax.bar(range(len(toks)), scores, color=colors,
                       edgecolor="white", lw=0.4)
                ax.set_xticks(range(len(toks)))
                ax.set_xticklabels(toks, rotation=60, ha="right", fontsize=6)
                ax.set_ylim(0, 1.1)

                if row == 0:
                    from .ft_evaluation import EXP_SHORT
                    ax.set_title(EXP_SHORT.get(exp_name, exp_name),
                                 fontsize=9, fontweight="bold",
                                 color="#333", pad=6)
                if col == 0:
                    short_text = text[:35] + "…" if len(text) > 35 else text
                    ax.set_ylabel(short_text, fontsize=7, labelpad=4)

                pred_name = label_names.get(ti.pred_label, "?")
                ok = "✓" if ti.pred_label == ti.true_label else "✗"
                ax.set_xlabel(f"{ok} {pred_name}", fontsize=7)

            except Exception as e:
                ax.text(0.5, 0.5, f"Error:\n{str(e)[:60]}",
                        ha="center", va="center", fontsize=7,
                        transform=ax.transAxes, color="red")

    fig.suptitle(f"Comparative Token Importance ({method}) — Same Inputs, All Models",
                 fontsize=12, y=1.01)
    fig.tight_layout(pad=1.5)
    return fig