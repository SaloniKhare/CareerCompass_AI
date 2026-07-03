"""Phase 2 - Synthetic dataset generator.

Public career datasets that map "skill set -> job role" are hard to find and
usually messy, so we generate a realistic *synthetic* dataset instead.

How it works
------------
For every career we know its ideal skill set (see utils/config.CAREER_SKILLS).
To create one training sample we:
    1. pick a career,
    2. keep a random subset of that career's real skills (so rows differ),
    3. occasionally sprinkle in 1-2 skills from *other* careers (real-world
       noise - people rarely fit one box perfectly),
    4. attach a plausible CGPA.
The label is the career we started from.

Running this file writes datasets/career_dataset.csv.

Usage:
    python -m utils.generate_datasets
"""

from __future__ import annotations

import csv
import os
import random
from typing import List

from utils.config import CAREER_SKILLS, CAREERS, CAREER_DATASET_PATH, DATASETS_DIR

# Fixing the seed makes the generated dataset reproducible across runs.
random.seed(42)

SAMPLES_PER_CAREER: int = 250  # 8 careers -> 2000 rows total


def _make_skill_string(career: str) -> str:
    """Build a realistic comma-separated skill string for one sample.

    Args:
        career: the target career whose skills we sample from.

    Returns:
        A comma-separated skill string, e.g. "python, pandas, sql".
    """
    ideal = CAREER_SKILLS[career]

    # Keep 60-100% of the career's real skills.
    k = random.randint(max(3, int(len(ideal) * 0.6)), len(ideal))
    chosen: List[str] = random.sample(ideal, k)

    # 35% of the time add mild noise from another career.
    if random.random() < 0.35:
        other = random.choice([c for c in CAREERS if c != career])
        noise = random.sample(CAREER_SKILLS[other], random.randint(1, 2))
        chosen.extend(noise)

    random.shuffle(chosen)
    # De-duplicate while preserving order.
    seen = set()
    unique = [s for s in chosen if not (s in seen or seen.add(s))]
    return ", ".join(unique)


def generate() -> str:
    """Generate the career dataset and write it to disk.

    Returns:
        The path of the CSV file that was written.
    """
    os.makedirs(DATASETS_DIR, exist_ok=True)

    rows = []
    for career in CAREERS:
        for _ in range(SAMPLES_PER_CAREER):
            skills = _make_skill_string(career)
            # CGPA loosely correlated with role but mostly random (6.0 - 9.9).
            cgpa = round(random.uniform(6.0, 9.9), 2)
            rows.append({"skills": skills, "cgpa": cgpa, "career": career})

    # Shuffle so careers are not grouped together (important before training).
    random.shuffle(rows)

    with open(CAREER_DATASET_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["skills", "cgpa", "career"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {CAREER_DATASET_PATH}")
    return CAREER_DATASET_PATH


if __name__ == "__main__":
    generate()
