"""Phases 10-13 - Career Prediction, Skill Gap, Course & Interview page.

This is the core analysis screen. It:
    10. predicts the best-fit career + confidence,
    11. shows the skill-gap analysis (matched vs missing),
    12. recommends courses,
    13. recommends interview questions.
Results are cached in session_state so the Report page can reuse them, and the
prediction is logged to the SQLite history table.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from database.db import save_prediction
from utils.predictor import model_is_ready, predict_career
from utils.recommender import (
    course_difficulties,
    recommend_courses,
    recommend_questions,
)
from utils.skill_gap import analyze_skill_gap

st.set_page_config(page_title="Career Prediction", page_icon="🎯", layout="wide")
st.title("🎯 Career Prediction & Analysis")

profile = st.session_state.get("profile")
if not profile:
    st.warning("Please create your **Student Profile** first.")
    st.stop()
if not model_is_ready():
    st.error("Model not trained. Run `python -m models.train_model` and refresh.")
    st.stop()

st.write(f"Analyzing profile for **{profile['name']}**")
st.caption(f"Skills: {profile.get('skills', '-')}")

if st.button("Run career analysis", type="primary"):
    career, confidence, dist = predict_career(profile["skills"], profile["cgpa"])
    gap = analyze_skill_gap(profile["skills"], career)
    # Log to history and cache for the report page.
    save_prediction(profile.get("id"), profile["name"], career, confidence)
    st.session_state["prediction"] = {
        "career": career,
        "confidence": confidence,
        "distribution": dist,
        "gap": gap,
    }

prediction = st.session_state.get("prediction")
if not prediction:
    st.info("Click **Run career analysis** to see your results.")
    st.stop()

career = prediction["career"]
confidence = prediction["confidence"]
dist = prediction["distribution"]
gap = prediction["gap"]

# ---------------------------------------------------------------------------
# Phase 10 - prediction result
# ---------------------------------------------------------------------------
st.divider()
c1, c2 = st.columns([1, 1])
with c1:
    st.subheader("Recommended Career")
    st.markdown(f"## {career}")
    st.metric("Confidence", f"{confidence:.1%}")
with c2:
    st.subheader("Match probability by career")
    top = dict(list(dist.items())[:5])  # show the top 5
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.barh(list(top.keys())[::-1], [v * 100 for v in list(top.values())[::-1]],
            color="#2563eb")
    ax.set_xlabel("Probability (%)")
    ax.set_xlim(0, 100)
    fig.tight_layout()
    st.pyplot(fig)

# ---------------------------------------------------------------------------
# Phase 11 - skill gap
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Skill Gap Analysis")
st.progress(int(gap.match_percentage), text=f"Skill match: {gap.match_percentage}%")
g1, g2 = st.columns(2)
with g1:
    st.markdown("**✅ Skills you already have**")
    st.write("\n".join(f"- {s}" for s in gap.matched) or "None matched yet.")
with g2:
    st.markdown("**📌 Skills to learn**")
    st.write("\n".join(f"- {s}" for s in gap.missing) or "You have them all!")

# ---------------------------------------------------------------------------
# Phases 12 & 13 - recommendations
# ---------------------------------------------------------------------------
st.divider()
tab_courses, tab_interview = st.tabs(["📚 Recommended Courses", "💬 Interview Questions"])

with tab_courses:
    diff = st.selectbox("Filter by difficulty", course_difficulties(), key="course_diff")
    courses = recommend_courses(career, difficulty=diff)
    if courses.empty:
        st.info("No courses match this filter.")
    else:
        st.dataframe(
            courses,
            hide_index=True,
            use_container_width=True,
            column_config={"link": st.column_config.LinkColumn("link")},
        )

with tab_interview:
    qdiff = st.selectbox("Filter by difficulty", ["All", "Easy", "Medium", "Hard"],
                         key="q_diff")
    questions = recommend_questions(career, difficulty=qdiff)
    if questions.empty:
        st.info("No questions match this filter.")
    else:
        for row in questions.itertuples():
            with st.expander(f"[{row.difficulty}] {row.question}"):
                st.caption("Practice answering this out loud, then research the ideal answer.")

st.divider()
st.success("Analysis complete! Go to the **Career Report** page to download a full report.")
