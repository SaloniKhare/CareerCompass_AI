"""Phase 15 - Career Report page.

Assembles the prediction, skill gap, course and interview recommendations into
a single downloadable report, saves it to the reports/ folder and the SQLite
reports table, and offers a download button.
"""

from __future__ import annotations

import streamlit as st

from database.db import save_report
from utils.recommender import recommend_courses, recommend_questions
from utils.report_generator import generate_report

st.set_page_config(page_title="Career Report", page_icon="📈", layout="wide")
st.title("📈 Career Report")

profile = st.session_state.get("profile")
prediction = st.session_state.get("prediction")

if not profile:
    st.warning("Please create your **Student Profile** first.")
    st.stop()
if not prediction:
    st.warning("Please run the **Career Prediction** first.")
    st.stop()

career = prediction["career"]
confidence = prediction["confidence"]
gap = prediction["gap"]

st.write(f"Generate a complete career report for **{profile['name']}**.")

if st.button("Generate report", type="primary"):
    courses = recommend_courses(career)
    questions = recommend_questions(career)
    result = generate_report(
        profile=profile,
        career=career,
        confidence=confidence,
        gap=gap,
        courses=courses,
        questions=questions,
        save=True,
    )
    # Persist the report content in the database too.
    save_report(profile.get("id"), profile["name"], career, result["content"])
    st.session_state["report"] = result

report = st.session_state.get("report")
if report:
    st.success(f"Report generated and saved to: `{report['path']}`")
    st.download_button(
        "⬇️ Download report (.txt)",
        data=report["content"],
        file_name=f"{profile['name'].replace(' ', '_')}_career_report.txt",
        mime="text/plain",
        type="primary",
    )
    with st.expander("Preview report", expanded=True):
        st.text(report["content"])
