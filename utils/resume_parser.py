"""Phase 9 - Resume parsing.

We accept a PDF resume, extract its raw text, then use regular expressions and
keyword matching (NLP-lite) to pull out structured information:
    * skills         (matched against our master skill list)
    * education       (lines under an EDUCATION heading)
    * projects        (lines under a PROJECTS heading)
    * certifications  (lines under a CERTIFICATIONS heading)

Text extraction tries pdfplumber first (better layout handling) and falls back
to PyPDF2. No external APIs are used.

Inputs : a file path OR a file-like object (Streamlit upload).
Outputs: a ParsedResume dataclass.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Union

from utils.config import ALL_SKILLS
from utils.data_cleaning import normalize_skill

# Section headings we look for (case-insensitive). Maps heading -> field name.
SECTION_PATTERNS = {
    "education": ["education", "academic"],
    "projects": ["projects", "project work", "personal projects"],
    "certifications": ["certifications", "certificates", "certification", "courses"],
    "skills": ["skills", "technical skills", "core competencies"],
}


@dataclass
class ParsedResume:
    """Structured data extracted from a resume."""

    raw_text: str = ""
    skills: List[str] = field(default_factory=list)
    education: List[str] = field(default_factory=list)
    projects: List[str] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)

    def skills_string(self) -> str:
        """Return skills as a comma-separated string (for the model)."""
        return ", ".join(self.skills)


def _extract_text(source: Union[str, "IO"]) -> str:
    """Extract raw text from a PDF using pdfplumber, falling back to PyPDF2.

    Args:
        source: a file path or a file-like object.

    Returns:
        The extracted text (empty string if extraction fails).
    """
    text = ""
    # --- Attempt 1: pdfplumber ---
    try:
        import pdfplumber

        with pdfplumber.open(source) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception:
        text = ""

    # --- Attempt 2: PyPDF2 (only if pdfplumber produced nothing) ---
    if not text.strip():
        try:
            from PyPDF2 import PdfReader

            # PyPDF2 needs the stream rewound if it's a file-like object.
            if hasattr(source, "seek"):
                source.seek(0)
            reader = PdfReader(source)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            text = ""

    return text


def _extract_skills(text: str) -> List[str]:
    """Match known skills against the resume text.

    We normalize the text and check each master skill as a whole word/phrase.
    """
    lowered = " " + re.sub(r"[^a-z0-9+#.\s]", " ", text.lower()) + " "
    found: List[str] = []
    for skill in ALL_SKILLS:
        # Word-boundary-ish match so "java" doesn't match "javascript".
        pattern = r"(?<![a-z0-9])" + re.escape(skill) + r"(?![a-z0-9])"
        if re.search(pattern, lowered):
            found.append(normalize_skill(skill))
    return sorted(set(found))


def _extract_section(text: str, keywords: List[str]) -> List[str]:
    """Grab the lines that belong to a section identified by its heading.

    We find a line that *is* one of the keywords (a heading), then collect the
    following non-empty lines until the next heading-like line appears.
    """
    lines = [ln.strip() for ln in text.splitlines()]
    all_headings = [kw for kws in SECTION_PATTERNS.values() for kw in kws]

    collected: List[str] = []
    capturing = False
    for line in lines:
        low = line.lower().strip(" :")
        if not line:
            continue
        # Is this line one of OUR target headings?
        if any(low == kw or low.startswith(kw) for kw in keywords) and len(line) < 40:
            capturing = True
            continue
        # Hit a different heading -> stop capturing.
        if capturing and any(low == h for h in all_headings) and low not in keywords:
            break
        if capturing:
            collected.append(line)
        if len(collected) >= 8:  # keep sections reasonably short
            break
    return collected


def parse_resume(source: Union[str, "IO"]) -> ParsedResume:
    """Parse a resume PDF into structured fields.

    Args:
        source: a PDF file path or a Streamlit UploadedFile / file-like object.

    Returns:
        A ParsedResume populated as best we can from the document.
    """
    text = _extract_text(source)
    if not text.strip():
        return ParsedResume(raw_text="")

    return ParsedResume(
        raw_text=text,
        skills=_extract_skills(text),
        education=_extract_section(text, SECTION_PATTERNS["education"]),
        projects=_extract_section(text, SECTION_PATTERNS["projects"]),
        certifications=_extract_section(text, SECTION_PATTERNS["certifications"]),
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        parsed = parse_resume(sys.argv[1])
        print("Skills        :", parsed.skills)
        print("Education     :", parsed.education)
        print("Projects      :", parsed.projects)
        print("Certifications:", parsed.certifications)
    else:
        print("Usage: python -m utils.resume_parser <path-to-resume.pdf>")
