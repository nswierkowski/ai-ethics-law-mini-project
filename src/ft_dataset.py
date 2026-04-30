"""
src/ft_dataset.py
─────────────────
PyTorch Dataset wrapper + tokenisation utilities for the unfair_tos task.

Responsibilities
────────────────
  • ToSDataset        – torch.utils.data.Dataset over a pd.DataFrame
  • build_dataloaders – tokenise, batch, return train/val/test DataLoaders
  • compute_class_weights_tensor – sklearn-balanced weights as torch.Tensor
  • FocalLoss         – class-weighted focal loss (Lin et al., 2017)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from .training_config import ExperimentConfig

# Lazy heavy imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ── Dataset ───────────────────────────────────────────────────────────────────

class ToSDataset(Dataset):
    """
    Wraps a pd.DataFrame (text, label) as a PyTorch Dataset.

    For binary task, maps original 10-class label → {0=Fair, 1=Unfair}.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_length: int = 128,
        is_binary: bool = False,
    ) -> None:
        self.texts = df["text"].tolist()
        if is_binary:
            self.labels = (df["label"] != 9).astype(int).tolist()
        else:
            self.labels = df["label"].tolist()
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }

    def get_texts(self) -> List[str]:
        return self.texts

    def get_labels(self) -> List[int]:
        return self.labels


# ── DataLoaders ───────────────────────────────────────────────────────────────

def build_dataloaders(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    cfg: ExperimentConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Tokenise splits and return (train_loader, val_loader, test_loader).
    """
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    train_ds = ToSDataset(train_df, tokenizer, cfg.max_length, cfg.is_binary)
    val_ds   = ToSDataset(val_df,   tokenizer, cfg.max_length, cfg.is_binary)
    test_ds  = ToSDataset(test_df,  tokenizer, cfg.max_length, cfg.is_binary)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size * 2, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size * 2, shuffle=False)

    return train_loader, val_loader, test_loader


# ── Class weights ─────────────────────────────────────────────────────────────

def compute_class_weights_tensor(
    df: pd.DataFrame,
    num_labels: int,
    is_binary: bool = False,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Sklearn-balanced weights as a float32 Tensor ready for nn.CrossEntropyLoss.
    """
    if is_binary:
        labels = (df["label"] != 9).astype(int).values
    else:
        labels = df["label"].values

    counts = np.bincount(labels, minlength=num_labels).astype(float)
    counts = np.where(counts == 0, 1, counts)   # avoid div/0
    n_samples = len(labels)
    weights = n_samples / (num_labels * counts)
    return torch.tensor(weights, dtype=torch.float32).to(device)


# ── Focal Loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification (Lin et al., ICCV 2017).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Parameters
    ----------
    weight : torch.Tensor, optional
        Per-class weights (alpha_t). Shape: (num_classes,)
    gamma  : float
        Focusing parameter. gamma=0 → standard cross-entropy.
    reduction : str
        'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.register_buffer("weight", weight)
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, C), targets: (B,)
        log_probs = F.log_softmax(logits, dim=-1)              # (B, C)
        probs     = torch.exp(log_probs)                        # (B, C)

        # Gather true-class log-prob and prob
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)
        pt     = probs.gather(1, targets.unsqueeze(1)).squeeze(1)      # (B,)

        focal_term = (1 - pt) ** self.gamma
        loss       = -focal_term * log_pt                               # (B,)

        if self.weight is not None:
            alpha = self.weight[targets]
            loss  = alpha * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
