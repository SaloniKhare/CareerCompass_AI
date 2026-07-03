"""Phase 11 - Skill gap analysis.

Once we know the target career, we compare the student's skills against that
career's ideal skill set (from utils.config.CAREER_SKILLS) to find:
    * matched skills   (skills they already have)
    * missing skills   (skills they should learn)
    * a match percentage (a simple readiness score)

This is pure set arithmetic on cleaned skill lists - no ML needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from utils.config import CAREER_SKILLS
from utils.data_cleaning import skills_to_list


@dataclass
class SkillGapResult:
    """Container for the outcome of a skill-gap analysis."""

    career: str
    matched: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    match_percentage: float = 0.0


def analyze_skill_gap(user_skills: str, career: str) -> SkillGapResult:
    """Compare a user's skills to a career's required skills.

    Args:
        user_skills: raw skills string (cleaned internally).
        career: the predicted / selected career.

    Returns:
        A SkillGapResult with matched skills, missing skills and a score.
    """
    required = CAREER_SKILLS.get(career, [])
    user_set = set(skills_to_list(user_skills))

    matched = [s for s in required if s in user_set]
    missing = [s for s in required if s not in user_set]

    pct = (len(matched) / len(required) * 100) if required else 0.0

    return SkillGapResult(
        career=career,
        matched=matched,
        missing=missing,
        match_percentage=round(pct, 1),
    )


if __name__ == "__main__":
    result = analyze_skill_gap("python, pandas, sql", "Data Scientist")
    print("Career         :", result.career)
    print("Matched        :", result.matched)
    print("Missing        :", result.missing)
    print("Match %        :", result.match_percentage)
