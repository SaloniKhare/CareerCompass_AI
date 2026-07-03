"""Phase 14 - SQLite persistence layer.

A small, dependency-free data layer built on Python's built-in sqlite3.
It stores three things the app needs to remember between sessions:
    * students     - the profile a user fills in
    * predictions  - every career prediction made (history)
    * reports      - the full generated career reports

Design notes
------------
* We use parameterized queries everywhere (the "?" placeholders) to prevent
  SQL injection.
* Each connection is opened, used and closed inside a context manager so we
  never leak file handles.
* `init_db()` is safe to call repeatedly (CREATE TABLE IF NOT EXISTS).
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, Iterator, List, Optional

from utils.config import DATABASE_DIR, DB_PATH


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """Yield a SQLite connection with row access by column name."""
    os.makedirs(DATABASE_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # rows behave like dicts
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create all tables if they do not already exist."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS students (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                name      TEXT NOT NULL,
                age       INTEGER,
                degree    TEXT,
                branch    TEXT,
                cgpa      REAL,
                skills    TEXT,
                interests TEXT,
                created_at TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id  INTEGER,
                name        TEXT,
                career      TEXT,
                confidence  REAL,
                created_at  TEXT,
                FOREIGN KEY (student_id) REFERENCES students (id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS reports (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id  INTEGER,
                name        TEXT,
                career      TEXT,
                content     TEXT,
                created_at  TEXT,
                FOREIGN KEY (student_id) REFERENCES students (id)
            )
            """
        )


# ---------------------------------------------------------------------------
# Students
# ---------------------------------------------------------------------------
def save_student(profile: Dict) -> int:
    """Insert a student profile and return its new row id."""
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO students (name, age, degree, branch, cgpa, skills,
                                  interests, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                profile.get("name"),
                profile.get("age"),
                profile.get("degree"),
                profile.get("branch"),
                profile.get("cgpa"),
                profile.get("skills"),
                profile.get("interests"),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        return int(cur.lastrowid)


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------
def save_prediction(
    student_id: Optional[int], name: str, career: str, confidence: float
) -> int:
    """Record a prediction in the history table."""
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO predictions (student_id, name, career, confidence, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                student_id,
                name,
                career,
                float(confidence),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        return int(cur.lastrowid)


def get_predictions(limit: int = 50) -> List[Dict]:
    """Return the most recent predictions as a list of dicts."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def save_report(
    student_id: Optional[int], name: str, career: str, content: str
) -> int:
    """Persist a full career report."""
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO reports (student_id, name, career, content, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                student_id,
                name,
                career,
                content,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        return int(cur.lastrowid)


def get_reports(limit: int = 50) -> List[Dict]:
    """Return the most recent reports as a list of dicts."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM reports ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


if __name__ == "__main__":
    init_db()
    print(f"Initialized database at {DB_PATH}")
