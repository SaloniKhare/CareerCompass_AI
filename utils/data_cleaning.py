"""Phase 3 - Data cleaning and text normalization.

Machine learning models cannot read raw messy text. Before training (and
again before every prediction) we must normalize skills into a consistent
format. Doing this in ONE module guarantees the exact same cleaning is applied
during training and during inference - if they differ, the model silently
gets worse.

What we clean
-------------
* lower-case everything
* replace separators (/, |, ;, newlines) with commas
* strip punctuation that is not part of a skill (keep + # . for c++, c#, node.js)
* collapse extra whitespace
* map common aliases to a canonical name (e.g. "ml" -> "machine learning")

Inputs : raw strings / a pandas DataFrame
Outputs: cleaned strings / a cleaned DataFrame
"""

from __future__ import annotations

import re
from typing import List

import pandas as pd

# Common ways students write the same skill -> the canonical form we train on.
SKILL_ALIASES = {
    "ml": "machine learning",
    "dl": "deep learning",
    "ai": "artificial intelligence",
    "js": "javascript",
    "ts": "typescript",
    "nodejs": "node.js",
    "node": "node.js",
    "reactjs": "react",
    "powerbi": "power bi",
    "postgres": "sql",
    "postgresql": "sql",
    "mysql": "sql",
    "nlp": "nlp",
    "cv": "computer vision",
    "oops": "oop",
    "k8s": "kubernetes",
    "tf": "tensorflow",
}


def normalize_skill(token: str) -> str:
    """Clean and canonicalize a single skill token.

    Args:
        token: a raw skill word, e.g. "  ReactJS ".

    Returns:
        The canonical, lower-cased skill, e.g. "react".
    """
    token = token.strip().lower()
    # Keep letters, numbers, spaces and a few skill-specific symbols.
    token = re.sub(r"[^a-z0-9+#.\s]", "", token)
    token = re.sub(r"\s+", " ", token).strip()
    return SKILL_ALIASES.get(token, token)


def clean_skill_text(raw: str) -> str:
    """Turn any messy skills string into a clean, comma-separated string.

    Args:
        raw: e.g. "Python / ML; reactjs".

    Returns:
        e.g. "python, machine learning, react".
    """
    if not isinstance(raw, str):
        return ""
    # Unify all common separators into commas.
    unified = re.sub(r"[\/|;\n\t]+", ",", raw)
    parts = [normalize_skill(p) for p in unified.split(",")]
    # Drop empties and duplicates while preserving order.
    seen = set()
    cleaned: List[str] = []
    for p in parts:
        if p and p not in seen:
            seen.add(p)
            cleaned.append(p)
    return ", ".join(cleaned)


def skills_to_list(raw: str) -> List[str]:
    """Return a clean list of individual skills from a raw string."""
    cleaned = clean_skill_text(raw)
    return [s for s in cleaned.split(", ") if s]


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a career DataFrame in place-safe manner.

    Args:
        df: DataFrame with at least a 'skills' column and a 'career' column.

    Returns:
        A cleaned copy: normalized skills, no null/empty rows, no duplicates.
    """
    df = df.copy()
    df["skills"] = df["skills"].apply(clean_skill_text)
    if "career" in df.columns:
        df["career"] = df["career"].astype(str).str.strip()
    # Remove rows that ended up with no skills.
    df = df[df["skills"].str.len() > 0]
    df = df.drop_duplicates().reset_index(drop=True)
    return df


if __name__ == "__main__":
    # Quick manual test.
    samples = ["Python / ML; reactjs", "AWS | Docker,K8S", "  HTML,CSS,,js  "]
    for s in samples:
        print(f"{s!r:35} -> {clean_skill_text(s)!r}")
