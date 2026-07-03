"""Phase 15 - Career report generation.

Collects the outputs of every other module (prediction, skill gap, course &
interview recommendations) into a single, human-readable text report. The
report is returned as a string AND written to the reports/ folder so it can be
downloaded or re-opened later.

Inputs : the student profile + all analysis results.
Outputs: the report text (str) and a saved .txt file path.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List

import pandas as pd

from utils.config import REPORTS_DIR
from utils.skill_gap import SkillGapResult


def _bullet(items: List[str]) -> str:
    """Format a list as indented bullet points (or a dash if empty)."""
    if not items:
        return "   - none"
    return "\n".join(f"   - {item}" for item in items)


def build_suggestions(gap: SkillGapResult, confidence: float) -> List[str]:
    """Generate simple, rule-based guidance based on the analysis."""
    tips: List[str] = []
    if confidence < 0.5:
        tips.append(
            "Your profile fits several careers. Add more specialized skills "
            "to strengthen a clear direction."
        )
    if gap.missing:
        tips.append(
            f"Focus first on: {', '.join(gap.missing[:3])} to close your biggest gaps."
        )
    if gap.match_percentage >= 70:
        tips.append(
            "You are well aligned with this career - start applying and "
            "practicing interview questions."
        )
    else:
        tips.append(
            "Build 1-2 portfolio projects using the recommended courses to "
            "demonstrate the missing skills."
        )
    return tips


def generate_report(
    profile: Dict,
    career: str,
    confidence: float,
    gap: SkillGapResult,
    courses: pd.DataFrame,
    questions: pd.DataFrame,
    save: bool = True,
) -> Dict[str, str]:
    """Build the full career report.

    Args:
        profile: the student profile dict (needs at least 'name').
        career: predicted career.
        confidence: prediction confidence 0-1.
        gap: SkillGapResult from the skill-gap analysis.
        courses: DataFrame of recommended courses.
        questions: DataFrame of recommended interview questions.
        save: whether to write the report to the reports/ folder.

    Returns:
        A dict with keys 'content' (the report text) and 'path' (file path or "").
    """
    name = profile.get("name", "Student")
    suggestions = build_suggestions(gap, confidence)

    course_lines = (
        [f"{r.course_name} ({r.platform}, {r.difficulty}, {r.duration}) - {r.link}"
         for r in courses.itertuples()]
        if not courses.empty
        else []
    )
    question_lines = (
        [f"[{r.difficulty}] {r.question}" for r in questions.itertuples()]
        if not questions.empty
        else []
    )

    lines = [
        "=" * 62,
        "            AI CAREER COUNSELLING - CAREER REPORT",
        "=" * 62,
        f"Name        : {name}",
        f"Degree      : {profile.get('degree', '-')} ({profile.get('branch', '-')})",
        f"CGPA        : {profile.get('cgpa', '-')}",
        f"Generated   : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "-" * 62,
        "1. CAREER PREDICTION",
        "-" * 62,
        f"Recommended Career : {career}",
        f"Confidence Score   : {confidence:.1%}",
        "",
        "-" * 62,
        "2. SKILL ANALYSIS",
        "-" * 62,
        f"Skill Match        : {gap.match_percentage}%",
        "Existing Skills    :",
        _bullet(gap.matched),
        "Missing Skills     :",
        _bullet(gap.missing),
        "",
        "-" * 62,
        "3. RECOMMENDED COURSES",
        "-" * 62,
        _bullet(course_lines),
        "",
        "-" * 62,
        "4. INTERVIEW QUESTIONS",
        "-" * 62,
        _bullet(question_lines),
        "",
        "-" * 62,
        "5. SUGGESTIONS",
        "-" * 62,
        _bullet(suggestions),
        "",
        "=" * 62,
        "End of report - generated locally by the AI Career Counselling System",
        "=" * 62,
    ]
    content = "\n".join(lines)

    path = ""
    if save:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        safe_name = "".join(c for c in name if c.isalnum() or c in (" ", "_")).strip()
        safe_name = safe_name.replace(" ", "_") or "student"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(REPORTS_DIR, f"{safe_name}_{stamp}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    return {"content": content, "path": path}


if __name__ == "__main__":
    from utils.skill_gap import analyze_skill_gap
    from utils.recommender import recommend_courses, recommend_questions

    demo_profile = {"name": "Test Student", "degree": "B.Tech", "branch": "CSE", "cgpa": 8.4}
    demo_gap = analyze_skill_gap("python, pandas, sql", "Data Scientist")
    out = generate_report(
        demo_profile, "Data Scientist", 0.82, demo_gap,
        recommend_courses("Data Scientist"), recommend_questions("Data Scientist"),
        save=False,
    )
    print(out["content"])
