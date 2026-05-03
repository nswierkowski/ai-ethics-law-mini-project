"""
src/ft_trainer.py
─────────────────
Model construction, training loop, evaluation, and checkpoint saving.

Changes vs initial version
───────────────────────────
  • Dropout   – classifier_dropout is applied to DistilBERT's classification
                head (pre_classifier → Dropout → classifier).  The value is
                passed via model config so it affects *all* dropout layers in
                the head, not just one manually inserted layer.
  • Early stopping – EarlyStopping monitors val macro-F1 with configurable
                patience and min_delta.  Best weights are always saved and
                restored at the end regardless of whether ES fired.
  • Bug fix   – _evaluate_test now passes labels=all_label_ids to every
                sklearn call, preventing the "N classes ≠ M target_names"
                crash when a rare class is absent from the test split.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from .training_config import ExperimentConfig, LABEL_NAMES

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from .ft_dataset import FocalLoss, compute_class_weights_tensor
_TORCH_AVAILABLE = True



class EarlyStopping:
    """
    Monitors a validation metric (higher = better) and signals when training
    should stop because the model has stopped improving.

    Usage
    -----
    >>> es = EarlyStopping(patience=3, min_delta=0.001)
    >>> for epoch in ...:
    ...     val_f1 = evaluate(...)
    ...     if es.step(val_f1):
    ...         break   # patience exhausted
    >>> best_epoch = es.best_epoch

    Attributes
    ----------
    best_score  : float  – best value seen so far
    best_epoch  : int    – epoch at which best_score was recorded (1-indexed)
    wait        : int    – epochs elapsed since last improvement
    stopped     : bool   – True once patience is exhausted
    """

    def __init__(self, patience: int = 3, min_delta: float = 0.001) -> None:
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_score = -np.inf
        self.best_epoch = 0
        self.wait       = 0
        self.stopped    = False
        self._epoch     = 0

    def step(self, score: float) -> bool:
        """
        Call once per epoch with the current validation metric.
        Returns True when training should stop (patience exhausted).
        """
        self._epoch += 1
        if score >= self.best_score + self.min_delta:
            self.best_score = score
            self.best_epoch = self._epoch
            self.wait       = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            self.stopped = True
            return True
        return False

    def __repr__(self) -> str:
        return (
            f"EarlyStopping(patience={self.patience}, min_delta={self.min_delta}, "
            f"best={self.best_score:.4f} @ epoch {self.best_epoch}, "
            f"wait={self.wait}, stopped={self.stopped})"
        )



@dataclass
class TrainingResult:
    """Holds everything produced by one fine-tuning run."""
    cfg: ExperimentConfig
    train_losses:       List[float] = field(default_factory=list)
    val_losses:         List[float] = field(default_factory=list)
    val_macro_f1s:      List[float] = field(default_factory=list)
    best_val_macro_f1:  float = 0.0
    best_epoch:         int   = 0
    early_stopped:      bool  = False
    epochs_trained:     int   = 0
    test_macro_f1:      float = 0.0
    test_weighted_f1:   float = 0.0
    test_accuracy:      float = 0.0
    classification_rep: str   = ""
    confusion_mat:      Optional[np.ndarray] = None
    per_class_f1:       Dict[str, float] = field(default_factory=dict)
    training_time_s:    float = 0.0
    model:     Optional[object] = None
    tokenizer: Optional[object] = None
    test_preds:  Optional[np.ndarray] = None
    test_labels: Optional[np.ndarray] = None



class DistilBertTrainer:
    """
    Self-contained fine-tuning engine for one ExperimentConfig.

    Dropout
    -------
    cfg.classifier_dropout overrides DistilBERT's seq_classif_dropout via
    AutoConfig *before* weights are loaded.  This controls the Dropout layer
    in the classification head:

        hidden_state → pre_classifier (Linear+ReLU) → Dropout(p) → classifier

    Typical range: 0.1–0.4.  Default here: 0.3.

    Early Stopping
    --------------
    Tracks val macro-F1.  After `cfg.es_patience` epochs with no improvement
    of at least `cfg.es_min_delta`, training halts and the best checkpoint is
    restored.  `result.early_stopped` records whether ES fired.
    """

    def __init__(
        self,
        cfg:      ExperimentConfig,
        train_df: pd.DataFrame,
        val_df:   pd.DataFrame,
        test_df:  pd.DataFrame,
        device:   Optional[str] = None,
    ) -> None:
        self.cfg      = cfg
        self.train_df = train_df
        self.val_df   = val_df
        self.test_df  = test_df
        self.device   = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Device            : {self.device}")
        print(f"  Classifier dropout: {cfg.classifier_dropout}")
        print(f"  Early stopping    : patience={cfg.es_patience}, "
              f"min_delta={cfg.es_min_delta}")


    def run(self) -> TrainingResult:
        result = TrainingResult(cfg=self.cfg)
        t0 = time.time()

        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        result.tokenizer = tokenizer

        from .ft_dataset import build_dataloaders
        train_df_used = self._maybe_downsample(self.train_df)
        train_loader, val_loader, test_loader = build_dataloaders(
            train_df_used, self.val_df, self.test_df, self.cfg
        )

        model = self._build_model()
        result.model = model

        criterion = self._build_criterion(train_df_used)

        optimizer = AdamW(
            model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        total_steps  = len(train_loader) * self.cfg.num_epochs
        warmup_steps = int(total_steps * self.cfg.warmup_ratio)
        scheduler    = get_linear_schedule_with_warmup(
            optimizer, warmup_steps, total_steps
        )

        es         = EarlyStopping(self.cfg.es_patience, self.cfg.es_min_delta)
        best_state: Optional[dict] = None

        for epoch in range(1, self.cfg.num_epochs + 1):
            train_loss          = self._train_epoch(
                model, train_loader, criterion, optimizer, scheduler
            )
            val_loss, val_f1    = self._eval_epoch(model, val_loader, criterion)

            result.train_losses.append(train_loss)
            result.val_losses.append(val_loss)
            result.val_macro_f1s.append(val_f1)
            result.epochs_trained = epoch

            improved = val_f1 >= es.best_score + es.min_delta
            if improved:
                best_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }

            stop = es.step(val_f1)

            marker = " ◀ best" if improved else ""
            es_tag = (f"  [ES wait {es.wait}/{es.patience}]"
                      if not improved else "")
            print(
                f"  Epoch {epoch:>2}/{self.cfg.num_epochs}  "
                f"train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"val_macro_f1={val_f1:.4f}"
                f"{marker}{es_tag}"
            )

            if stop:
                print(
                    f"\n  ⏹  Early stopping triggered after epoch {epoch}  "
                    f"(no improvement for {self.cfg.es_patience} epochs)."
                    f"\n     Best val macro-F1 = {es.best_score:.4f} "
                    f"@ epoch {es.best_epoch}"
                )
                result.early_stopped = True
                break

        if best_state is not None:
            model.load_state_dict(
                {k: v.to(self.device) for k, v in best_state.items()}
            )
            print(f"  ↩  Restored best weights from epoch {es.best_epoch}")

        result.best_val_macro_f1 = float(es.best_score)
        result.best_epoch        = es.best_epoch

        self._evaluate_test(model, test_loader, result)

        self._save_checkpoint(model, tokenizer)

        result.training_time_s = time.time() - t0
        print(
            f"\n  ✅ Done in {result.training_time_s / 60:.1f} min  "
            f"| epochs={result.epochs_trained}/{self.cfg.num_epochs}  "
            f"| early_stopped={result.early_stopped}  "
            f"| best val macro-F1={result.best_val_macro_f1:.4f}  "
            f"| test macro-F1={result.test_macro_f1:.4f}"
        )
        return result


    def _build_model(self) -> "nn.Module":
        """
        Load DistilBERT and override the classification-head dropout.

        DistilBERT's config exposes `seq_classif_dropout` which controls the
        Dropout layer inserted between pre_classifier and classifier.
        Setting it via AutoConfig before from_pretrained means the head is
        constructed with our value from the start.
        """
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(
            self.cfg.model_name,
            num_labels=self.cfg.num_labels,
        )
        config.seq_classif_dropout = self.cfg.classifier_dropout

        return AutoModelForSequenceClassification.from_pretrained(
            self.cfg.model_name,
            config=config,
            ignore_mismatched_sizes=True,
            # Force eager attention so attention weights are always available
            # for explainability (SDPA / flash_attention_2 do not return them).
            attn_implementation="eager",
        ).to(self.device)

    def _maybe_downsample(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.cfg.downsample_ratio is None:
            return df
        from .eda_utils import downsample_majority_class
        return downsample_majority_class(
            df,
            target_ratio=self.cfg.downsample_ratio,
            random_state=self.cfg.seed,
        )

    def _build_criterion(self, train_df: pd.DataFrame) -> "nn.Module":
        if not self.cfg.handle_imbalance:
            return nn.CrossEntropyLoss()
        weights = compute_class_weights_tensor(
            train_df,
            num_labels=self.cfg.num_labels,
            is_binary=self.cfg.is_binary,
            device=self.device,
        )
        if self.cfg.use_focal_loss:
            return FocalLoss(weight=weights, gamma=self.cfg.focal_gamma)
        return nn.CrossEntropyLoss(weight=weights)

    def _train_epoch(
        self,
        model:     "nn.Module",
        loader:    "DataLoader",
        criterion: "nn.Module",
        optimizer: "AdamW",
        scheduler,
    ) -> float:
        model.train()
        total_loss = 0.0
        for batch in loader:
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels         = batch["labels"].to(self.device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss    = criterion(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        return total_loss / len(loader)

    def _eval_epoch(
        self,
        model:     "nn.Module",
        loader:    "DataLoader",
        criterion: "nn.Module",
    ) -> Tuple[float, float]:
        model.eval()
        total_loss    = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in loader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["labels"].to(self.device)

                outputs     = model(input_ids=input_ids, attention_mask=attention_mask)
                loss        = criterion(outputs.logits, labels)
                total_loss += loss.item()

                preds = outputs.logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        all_label_ids = list(self.cfg.label_names.keys())
        macro_f1 = f1_score(
            all_labels, all_preds,
            labels=all_label_ids,
            average="macro",
            zero_division=0,
        )
        return total_loss / len(loader), macro_f1

    def _evaluate_test(
        self,
        model:  "nn.Module",
        loader: "DataLoader",
        result: TrainingResult,
    ) -> None:
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in loader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["labels"].to(self.device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds   = outputs.logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        preds  = np.array(all_preds)
        labels = np.array(all_labels)

        # Always use the full label set from config — never infer from data.
        # This prevents the sklearn crash when a rare class is absent from
        # the test split after downsampling.
        all_label_ids = list(result.cfg.label_names.keys())
        label_names   = list(result.cfg.label_names.values())

        result.test_preds       = preds
        result.test_labels      = labels
        result.test_macro_f1    = f1_score(
            labels, preds, labels=all_label_ids,
            average="macro",    zero_division=0)
        result.test_weighted_f1 = f1_score(
            labels, preds, labels=all_label_ids,
            average="weighted", zero_division=0)
        result.test_accuracy    = float((preds == labels).mean())
        result.confusion_mat    = confusion_matrix(
            labels, preds, labels=all_label_ids)
        result.classification_rep = classification_report(
            labels, preds,
            labels=all_label_ids,
            target_names=label_names,
            zero_division=0,
        )
        per_class = f1_score(
            labels, preds, labels=all_label_ids,
            average=None, zero_division=0)
        result.per_class_f1 = {
            result.cfg.label_names[lbl]: float(per_class[i])
            for i, lbl in enumerate(all_label_ids)
        }

    def _save_checkpoint(
        self, model: "nn.Module", tokenizer: "AutoTokenizer"
    ) -> None:
        save_path = self.cfg.model_save_path
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"  💾 Saved → {save_path}")