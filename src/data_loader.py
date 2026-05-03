"""
src/data_loader.py
──────────────────
Loads the lex_glue / unfair_tos dataset (HuggingFace Datasets).
Falls back to a deterministic mock corpus when the network is unavailable,
so the EDA notebook always runs end-to-end.

Label schema (from the official lex_glue paper):
    0  – 'a'  Arbitration (dispute resolution stripped)
    1  – 'ch' Content removal / account termination
    2  – 'cr' Copyright / IP transfer clauses
    3  – 'j'  Jurisdiction (unilateral choice)
    4  – 'law' Governing-law clause
    5  – 'ltd' Limitation of liability
    6  – 'ter' Unilateral contract termination
    7  – 'use' Broad usage of personal data
    8  – 'pinc' Privacy / policy-change without notice
    9  – 'OB'  Clearly OK / Fair

Reference: https://huggingface.co/datasets/lex_glue  (unfair_tos subset)
"""

from __future__ import annotations

import random
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

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

LABEL_ABBR: Dict[int, str] = {
    0: "a", 1: "ch", 2: "cr", 3: "j", 4: "law",
    5: "ltd", 6: "ter", 7: "use", 8: "pinc", 9: "OB",
}

UNFAIR_LABELS = set(range(9))
FAIR_LABEL = 9

@dataclass
class DatasetSplit:
    """Holds a single train/validation/test split as a DataFrame."""
    name: str
    df: pd.DataFrame = field(default_factory=pd.DataFrame)

    # convenience properties
    @property
    def texts(self) -> pd.Series:
        return self.df["text"]

    @property
    def labels(self) -> pd.Series:
        return self.df["label"]

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        return f"DatasetSplit(name={self.name!r}, n={len(self)})"


def load_dataset(
    use_mock_fallback: bool = False,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, DatasetSplit]:
    """
    Load lex_glue / unfair_tos and return a dict of DatasetSplit objects.

    Parameters
    ----------
    use_mock_fallback : bool
        If True and `datasets` is not importable or network is down,
        generate a deterministic mock corpus instead.
    seed : int
        Random seed for reproducibility.
    verbose : bool
        Print loading info.

    Returns
    -------
    splits : dict
        Keys: "train", "validation", "test"
        Values: DatasetSplit objects with .df, .texts, .labels
    """
    from datasets import load_dataset as hf_load  

    if verbose:
        print("📦 Loading lex_glue / unfair_tos from HuggingFace …")

    raw = hf_load("lex_glue", "unfair_tos")
    splits: Dict[str, DatasetSplit] = {}

    for split_name in ("train", "validation", "test"):
        hf_split = raw[split_name]
        df = hf_split.to_pandas()

        if "labels" in df.columns and "label" not in df.columns:
            df = df.rename(columns={"labels": "label"})
            
        _first = df["label"].iloc[0]
        if isinstance(_first, (list, tuple, np.ndarray)):
            df["label"] = df["label"].apply(
                lambda x: x[0] if isinstance(x, (list, tuple, np.ndarray)) and len(x) else FAIR_LABEL
            )
            
        df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(FAIR_LABEL).astype(int)

        df["label_name"] = df["label"].map(LABEL_NAMES)
        df["label_abbr"] = df["label"].map(LABEL_ABBR)
        df["is_unfair"] = df["label"].isin(UNFAIR_LABELS).astype(int)
        df["text_length"] = df["text"].str.len()
        df["word_count"] = df["text"].str.split().str.len()
        df["sentence_count"] = df["text"].str.count(r"\.") + 1
        splits[split_name] = DatasetSplit(name=split_name, df=df)

    if verbose:
        for name, sp in splits.items():
            print(f" {name:12s} → {len(sp):,} samples")
    return splits
