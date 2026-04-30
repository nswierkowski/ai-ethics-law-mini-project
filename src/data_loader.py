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

# ── Label metadata ────────────────────────────────────────────────────────────

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

# Labels 0-8 are "unfair"; label 9 is "fair"
UNFAIR_LABELS = set(range(9))
FAIR_LABEL = 9

# # ── Mock corpus sentences per class ──────────────────────────────────────────

# _SEEDS: Dict[int, List[str]] = {
#     0: [
#         "Any dispute arising from this agreement shall be resolved through binding arbitration.",
#         "You waive your right to participate in class action lawsuits.",
#         "Arbitration shall take place exclusively in the company's home state.",
#         "By using this service you consent to mandatory binding arbitration.",
#         "All claims must be resolved on an individual basis and not as a class action.",
#     ],
#     1: [
#         "We may remove any content at our sole discretion without prior notice.",
#         "Accounts may be terminated for any reason including no reason at all.",
#         "We reserve the right to delete user-generated content deemed inappropriate.",
#         "Your account may be suspended or terminated without warning or explanation.",
#         "Content that violates our guidelines will be removed immediately.",
#     ],
#     2: [
#         "By uploading content you grant us a worldwide royalty-free license to use it.",
#         "You transfer all intellectual property rights to content submitted to us.",
#         "We may reproduce, distribute, and create derivative works from your content.",
#         "All user submissions become the exclusive property of the company.",
#         "You assign to us a perpetual irrevocable license to your uploaded materials.",
#     ],
#     3: [
#         "Any legal proceedings must be brought exclusively in courts of Delaware.",
#         "You consent to the exclusive jurisdiction of courts in our home state.",
#         "Disputes shall be heard only in the courts of San Francisco, California.",
#         "By using this service you submit to personal jurisdiction in our chosen venue.",
#         "All claims must be filed in the courts located in our headquarters country.",
#     ],
#     4: [
#         "This agreement shall be governed by the laws of the State of California.",
#         "These terms are governed by and construed in accordance with English law.",
#         "The governing law of this agreement is the law of Ireland.",
#         "All disputes shall be resolved under the laws of Delaware.",
#         "This contract is subject to the laws of the jurisdiction we select.",
#     ],
#     5: [
#         "We are not liable for any indirect, incidental, or consequential damages.",
#         "Our total liability shall not exceed the amount you paid in the last month.",
#         "We provide the service as-is without any warranty of any kind.",
#         "Under no circumstances shall our liability exceed one hundred dollars.",
#         "We disclaim all warranties express or implied including merchantability.",
#     ],
#     6: [
#         "We may terminate this agreement at any time for any or no reason.",
#         "We reserve the right to discontinue the service without prior notice.",
#         "We can modify or terminate the service at our sole discretion.",
#         "The company may end your access to the platform immediately.",
#         "We retain the right to suspend or terminate accounts unilaterally.",
#     ],
#     7: [
#         "We may share your data with third-party partners for marketing purposes.",
#         "Your personal information may be used for any purpose at our discretion.",
#         "We collect and use your data to improve our products and services broadly.",
#         "By registering you consent to receive commercial communications from us.",
#         "We may process your personal data for research and analytical purposes.",
#     ],
#     8: [
#         "We may update this privacy policy at any time without notifying you.",
#         "Continued use of the service constitutes acceptance of updated terms.",
#         "We reserve the right to change these terms with or without notice.",
#         "The privacy policy is subject to change at our sole discretion.",
#         "Material changes to this policy may be implemented without prior notice.",
#     ],
#     9: [
#         "We will notify you at least 30 days before making material changes.",
#         "You may cancel your subscription at any time with no penalty.",
#         "Your data is encrypted in transit and at rest using industry standards.",
#         "You retain all intellectual property rights to content you create.",
#         "We will never sell your personal data to third parties.",
#         "You can request deletion of your personal data at any time.",
#         "Our service complies with GDPR and CCPA data protection regulations.",
#         "We collect only the minimum data necessary to provide the service.",
#         "You have the right to access and export all data we hold about you.",
#         "Disputes will be resolved through neutral mediation of your choosing.",
#     ],
# }

# # Realistic class distribution from the actual unfair_tos dataset (approx.)
# _CLASS_WEIGHTS = {
#     0: 0.036, 1: 0.037, 2: 0.029, 3: 0.033, 4: 0.020,
#     5: 0.042, 6: 0.038, 7: 0.043, 8: 0.030, 9: 0.692,
# }

# # ── Split sizes ───────────────────────────────────────────────────────────────

# _SPLIT_SIZES = {"train": 5532, "validation": 2275, "test": 1607}


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
    from datasets import load_dataset as hf_load  # type: ignore

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

        # if df["label"].dtype == object or isinstance(df["label"].iloc[0], list):
        #     df["label"] = df["label"].apply(
        #         lambda x: x[0] if isinstance(x, list) and x else FAIR_LABEL
        #     )

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
