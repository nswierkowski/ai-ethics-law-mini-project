"""
src/training_config.py
──────────────────────
Central configuration for all four fine-tuning experiments.
Import this everywhere so hyper-parameters are never duplicated.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUTS_DIR   = os.path.join(ROOT_DIR, "outputs")
MODELS_DIR    = os.path.join(ROOT_DIR, "models")
FIGURES_DIR   = os.path.join(OUTPUTS_DIR, "figures")

for _d in (OUTPUTS_DIR, MODELS_DIR, FIGURES_DIR):
    os.makedirs(_d, exist_ok=True)


# ── Label maps (copied here so config is self-contained) ──────────────────────

LABEL_NAMES: Dict[int, str] = {
    0: "Arbitration",
    1: "Content Removal",
    2: "Copyright/IP",
    3: "Jurisdiction",
    4: "Governing Law",
    5: "Limitation of Liability",
    6: "Unilateral Termination",
    7: "Broad Data Use",
    8: "Privacy Change",
    9: "OK / Fair",
}

BINARY_LABEL_NAMES: Dict[int, str] = {0: "Fair", 1: "Unfair"}


# ── Experiment registry ────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    """
    Holds every hyper-parameter and flag for one fine-tuning run.

    Four experiments are pre-defined:
        A  – Multiclass, NO imbalance handling   (baseline)
        B  – Multiclass, WITH imbalance handling (weighted loss + downsampling)
        C  – Binary (fair/unfair), WITH imbalance handling
        D  – Multiclass, WITH imbalance + focal loss
    """
    name: str
    description: str
    task: str                           # "multiclass" | "binary"
    num_labels: int
    label_names: Dict[int, str]
    handle_imbalance: bool = False
    use_focal_loss: bool   = False
    downsample_ratio: Optional[float] = None   # None = use full data
    focal_gamma: float = 2.0

    model_name: str      = "distilbert-base-uncased"
    max_length: int      = 128
    batch_size: int      = 16
    num_epochs: int      = 30   # higher ceiling — early stopping will cut it short
    learning_rate: float = 2e-5
    warmup_ratio: float  = 0.1
    weight_decay: float  = 0.01
    seed: int            = 42

    classifier_dropout: float = 0.3

    es_patience:  int   = 3      
    es_min_delta: float = 0.001  

    @property
    def model_save_path(self) -> str:
        return os.path.join(MODELS_DIR, self.name)

    @property
    def is_binary(self) -> bool:
        return self.task == "binary"


EXPERIMENTS: Dict[str, ExperimentConfig] = {
    "A_multiclass_baseline": ExperimentConfig(
        name="A_multiclass_baseline",
        description="10-class DistilBERT — NO imbalance handling (pure baseline)",
        task="multiclass",
        num_labels=10,
        label_names=LABEL_NAMES,
        handle_imbalance=False,
        use_focal_loss=False,
        downsample_ratio=None,
    ),
    "B_multiclass_balanced": ExperimentConfig(
        name="B_multiclass_balanced",
        description="10-class DistilBERT — downsampling 2:1 + weighted cross-entropy",
        task="multiclass",
        num_labels=10,
        label_names=LABEL_NAMES,
        handle_imbalance=True,
        use_focal_loss=False,
        downsample_ratio=2.0,
    ),
    "C_binary_balanced": ExperimentConfig(
        name="C_binary_balanced",
        description="Binary (Fair/Unfair) DistilBERT — downsampling 2:1 + weighted CE",
        task="binary",
        num_labels=2,
        label_names=BINARY_LABEL_NAMES,
        handle_imbalance=True,
        use_focal_loss=False,
        downsample_ratio=2.0,
    ),
    "D_multiclass_focal": ExperimentConfig(
        name="D_multiclass_focal",
        description="10-class DistilBERT — downsampling 2:1 + Focal Loss (γ=2)",
        task="multiclass",
        num_labels=10,
        label_names=LABEL_NAMES,
        handle_imbalance=True,
        use_focal_loss=True,
        downsample_ratio=2.0,
        focal_gamma=2.0,
    ),
}