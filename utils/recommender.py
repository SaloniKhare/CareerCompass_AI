"""Phases 12 & 13 - Course and interview-question recommendation.

Both features are CSV-driven (no APIs): we filter the datasets by the
predicted career and return tidy DataFrames the UI can render directly.

Datasets used:
    datasets/courses.csv            -> course recommendations
    datasets/interview_questions.csv -> interview questions
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

import pandas as pd

from utils.config import COURSES_PATH, INTERVIEW_PATH


@lru_cache(maxsize=1)
def _load_courses() -> pd.DataFrame:
    """Load and cache the courses dataset."""
    return pd.read_csv(COURSES_PATH)


@lru_cache(maxsize=1)
def _load_questions() -> pd.DataFrame:
    """Load and cache the interview questions dataset."""
    return pd.read_csv(INTERVIEW_PATH)


def recommend_courses(
    career: str, difficulty: Optional[str] = None, limit: int = 10
) -> pd.DataFrame:
    """Recommend courses for a career (Phase 12).

    Args:
        career: predicted career to filter by.
        difficulty: optional filter ("Beginner"/"Intermediate"/"Advanced").
        limit: maximum number of courses to return.

    Returns:
        A DataFrame of matching courses (may be empty if none found).
    """
    df = _load_courses()
    result = df[df["career"] == career]
    if difficulty and difficulty != "All":
        result = result[result["difficulty"] == difficulty]
    return result.head(limit).reset_index(drop=True)


def recommend_questions(
    career: str, difficulty: Optional[str] = None, limit: int = 12
) -> pd.DataFrame:
    """Recommend interview questions for a career (Phase 13).

    Args:
        career: predicted career to filter by.
        difficulty: optional filter ("Easy"/"Medium"/"Hard").
        limit: maximum number of questions to return.

    Returns:
        A DataFrame of matching questions (may be empty if none found).
    """
    df = _load_questions()
    result = df[df["career"] == career]
    if difficulty and difficulty != "All":
        result = result[result["difficulty"] == difficulty]
    return result.head(limit).reset_index(drop=True)


def course_difficulties() -> List[str]:
    """Return the distinct difficulty levels present in the courses dataset."""
    return ["All"] + sorted(_load_courses()["difficulty"].unique().tolist())


if __name__ == "__main__":
    print(recommend_courses("Data Scientist").to_string(index=False))
    print()
    print(recommend_questions("Data Scientist", "Medium").to_string(index=False))
