"""Central configuration for the AI Career Counselling System.

This module is the single source of truth for:
    * the list of careers the model can predict
    * the skills each career typically requires
    * useful file-system paths

Keeping this data in one place means every other module (dataset
generator, model trainer, skill-gap analyzer, UI) stays in sync.
"""

from __future__ import annotations

import os
from typing import Dict, List

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# BASE_DIR points to the project root (the CareerCounsellor/ folder), no matter
# where a script is launched from. We build every other path relative to it.
BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASETS_DIR: str = os.path.join(BASE_DIR, "datasets")
MODELS_DIR: str = os.path.join(BASE_DIR, "models")
DATABASE_DIR: str = os.path.join(BASE_DIR, "database")
REPORTS_DIR: str = os.path.join(BASE_DIR, "reports")

CAREER_DATASET_PATH: str = os.path.join(DATASETS_DIR, "career_dataset.csv")
COURSES_PATH: str = os.path.join(DATASETS_DIR, "courses.csv")
INTERVIEW_PATH: str = os.path.join(DATASETS_DIR, "interview_questions.csv")

MODEL_PATH: str = os.path.join(MODELS_DIR, "career_model.pkl")
DB_PATH: str = os.path.join(DATABASE_DIR, "career.db")

# ---------------------------------------------------------------------------
# Careers and their required skills
# ---------------------------------------------------------------------------
# The keys are the labels the ML model predicts. The values are the "ideal"
# skill sets used both to generate synthetic training data AND to run the
# skill-gap analysis later on.
CAREER_SKILLS: Dict[str, List[str]] = {
    "AI Engineer": [
        "python", "machine learning", "deep learning", "tensorflow",
        "pytorch", "nlp", "computer vision", "neural networks",
        "mathematics", "data preprocessing",
    ],
    "Data Scientist": [
        "python", "statistics", "machine learning", "pandas", "numpy",
        "data visualization", "sql", "scikit-learn", "feature engineering",
        "hypothesis testing",
    ],
    "Data Analyst": [
        "excel", "sql", "python", "data visualization", "statistics",
        "power bi", "tableau", "pandas", "reporting", "data cleaning",
    ],
    "Software Developer": [
        "java", "python", "data structures", "algorithms", "oop",
        "git", "problem solving", "software design", "debugging", "testing",
    ],
    "Backend Developer": [
        "node.js", "python", "java", "sql", "api development", "databases",
        "django", "flask", "authentication", "system design",
    ],
    "Frontend Developer": [
        "html", "css", "javascript", "react", "ui design", "typescript",
        "responsive design", "redux", "accessibility", "web performance",
    ],
    "Cyber Security Analyst": [
        "networking", "linux", "cryptography", "ethical hacking",
        "security", "penetration testing", "firewalls", "risk assessment",
        "incident response", "siem",
    ],
    "Cloud Engineer": [
        "aws", "azure", "docker", "kubernetes", "linux", "networking",
        "devops", "terraform", "ci/cd", "monitoring",
    ],
}

# Convenience list of all career names.
CAREERS: List[str] = list(CAREER_SKILLS.keys())

# A flat, de-duplicated master list of every skill known to the system.
ALL_SKILLS: List[str] = sorted({s for skills in CAREER_SKILLS.values() for s in skills})

# Interest areas offered in the profile form (used as a light feature/label hint).
INTERESTS: List[str] = [
    "Artificial Intelligence", "Data & Analytics", "Web Development",
    "Software Engineering", "Cyber Security", "Cloud & DevOps",
    "Mathematics", "Design",
]
