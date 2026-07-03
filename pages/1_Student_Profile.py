"""Phase 8 - Student Profile page.

Collects the student's academic details, skills and interests in a form, then
stores the profile in st.session_state (so other pages can use it) and in the
SQLite database (so it is remembered between sessions).
"""

from __future__ import annotations

import streamlit as st

from database.db import save_student
from utils.config import ALL_SKILLS, INTERESTS
from utils.data_cleaning import clean_skill_text

st.set_page_config(page_title="Student Profile", page_icon="📝", layout="wide")
st.title("📝 Student Profile")
st.write("Tell us about yourself. These details feed the career prediction model.")

# Pre-fill the form with any existing profile so edits are easy.
existing = st.session_state.get("profile") or {}

with st.form("profile_form"):
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Full name", value=existing.get("name", ""))
        age = st.number_input(
            "Age", min_value=16, max_value=60, value=int(existing.get("age", 21))
        )
        degree = st.text_input("Degree", value=existing.get("degree", "B.Tech"))
    with col2:
        branch = st.text_input("Branch", value=existing.get("branch", "CSE"))
        cgpa = st.number_input(
            "CGPA", min_value=0.0, max_value=10.0,
            value=float(existing.get("cgpa", 8.0)), step=0.1,
        )

    # Multiselect from the known skill list keeps input clean and consistent.
    default_skills = clean_skill_text(existing.get("skills", "")).split(", ")
    default_skills = [s for s in default_skills if s in ALL_SKILLS]
    skills = st.multiselect(
        "Skills (pick all that apply)", options=ALL_SKILLS, default=default_skills
    )
    extra_skills = st.text_input(
        "Other skills (comma-separated, optional)",
        help="Anything not in the list above.",
    )

    interests = st.multiselect(
        "Interests", options=INTERESTS,
        default=[i for i in existing.get("interests", "").split(", ") if i in INTERESTS],
    )

    submitted = st.form_submit_button("Save profile", type="primary")

if submitted:
    if not name.strip():
        st.error("Please enter your name.")
    elif not skills and not extra_skills.strip():
        st.error("Please select or enter at least one skill.")
    else:
        # Merge picked skills + free-text skills into one clean string.
        combined = ", ".join(skills)
        if extra_skills.strip():
            combined = f"{combined}, {extra_skills}" if combined else extra_skills
        combined = clean_skill_text(combined)

        profile = {
            "name": name.strip(),
            "age": int(age),
            "degree": degree.strip(),
            "branch": branch.strip(),
            "cgpa": float(cgpa),
            "skills": combined,
            "interests": ", ".join(interests),
        }

        # Persist to DB and remember the new row id in session.
        student_id = save_student(profile)
        profile["id"] = student_id
        st.session_state["profile"] = profile
        # A new profile invalidates any previous prediction.
        st.session_state["prediction"] = None

        st.success(f"Profile saved for {name}! (id #{student_id})")
        st.write("**Skills recorded:**", combined)
        st.info("Next: upload a resume, or go straight to **Career Prediction**.")
