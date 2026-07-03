"""Phase 10 - Career prediction (model inference).

Loads the pickled pipeline produced by models/train_model.py and turns a raw
student profile into a career prediction with a confidence score and the full
probability distribution across all careers.

The model file is loaded once and cached so we don't hit disk on every call.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List, Tuple

import joblib
import pandas as pd

from utils.config import MODEL_PATH
from utils.data_cleaning import clean_skill_text


@lru_cache(maxsize=1)
def _load_artifact() -> dict:
    """Load and cache the trained model artifact.

    Raises:
        FileNotFoundError: if the model has not been trained yet.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model not found. Train it first with: python -m models.train_model"
        )
    return joblib.load(MODEL_PATH)


def model_is_ready() -> bool:
    """Return True if a trained model file exists on disk."""
    return os.path.exists(MODEL_PATH)


def predict_career(skills: str, cgpa: float) -> Tuple[str, float, Dict[str, float]]:
    """Predict the best-fit career for a student.

    Args:
        skills: raw comma-separated skills string (any format - it is cleaned).
        cgpa: the student's CGPA.

    Returns:
        A tuple of:
            * predicted career (str)
            * confidence for that career, 0-1 (float)
            * dict {career: probability} sorted high -> low
    """
    artifact = _load_artifact()
    pipeline = artifact["pipeline"]

    cleaned = clean_skill_text(skills)
    X = pd.DataFrame([{"skills": cleaned, "cgpa": float(cgpa)}])

    prediction = pipeline.predict(X)[0]

    # Build the probability map (all our candidate models expose predict_proba).
    proba = pipeline.predict_proba(X)[0]
    classes: List[str] = list(pipeline.classes_)
    prob_map = {c: float(p) for c, p in zip(classes, proba)}
    prob_map = dict(sorted(prob_map.items(), key=lambda kv: kv[1], reverse=True))

    confidence = prob_map[prediction]
    return prediction, confidence, prob_map


if __name__ == "__main__":
    # Manual smoke test using the trained model.
    demo = "python, tensorflow, deep learning, nlp, neural networks"
    career, conf, dist = predict_career(demo, 8.5)
    print(f"Skills : {demo}")
    print(f"Career : {career}  (confidence {conf:.1%})")
    print("Top 3  :", list(dist.items())[:3])
