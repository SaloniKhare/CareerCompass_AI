"""Phase 9 - Resume Upload & Parsing page.

Lets the student upload a PDF resume. We extract structured fields (skills,
education, projects, certifications) and offer to merge the detected skills
into their profile so the prediction becomes more accurate.
"""

from __future__ import annotations

import streamlit as st

from utils.data_cleaning import clean_skill_text
from utils.resume_parser import parse_resume

st.set_page_config(page_title="Resume Upload", page_icon="📄", layout="wide")
st.title("📄 Resume Upload & Parsing")

profile = st.session_state.get("profile")
if not profile:
    st.warning("Please create your **Student Profile** first.")
    st.stop()

st.write("Upload a PDF resume and we will automatically extract your skills.")

uploaded = st.file_uploader("Choose a PDF resume", type=["pdf"])

if uploaded is not None:
    with st.spinner("Parsing resume..."):
        parsed = parse_resume(uploaded)
    st.session_state["parsed_resume"] = parsed

    if not parsed.raw_text.strip():
        st.error(
            "Could not read any text from this PDF. It may be a scanned image. "
            "Try a text-based PDF or enter skills manually on the Profile page."
        )
    else:
        st.success("Resume parsed successfully!")

        tab1, tab2, tab3, tab4 = st.tabs(
            ["Skills", "Education", "Projects", "Certifications"]
        )
        with tab1:
            st.write(", ".join(parsed.skills) if parsed.skills else "No known skills detected.")
        with tab2:
            st.write("\n".join(f"- {e}" for e in parsed.education) or "Not found.")
        with tab3:
            st.write("\n".join(f"- {p}" for p in parsed.projects) or "Not found.")
        with tab4:
            st.write("\n".join(f"- {c}" for c in parsed.certifications) or "Not found.")

        with st.expander("View extracted raw text"):
            st.text(parsed.raw_text[:3000])

        st.divider()
        if parsed.skills:
            st.subheader("Merge detected skills into your profile?")
            st.write("Detected:", ", ".join(parsed.skills))
            if st.button("Add these skills to my profile", type="primary"):
                merged = clean_skill_text(
                    f"{profile.get('skills', '')}, {parsed.skills_string()}"
                )
                profile["skills"] = merged
                st.session_state["profile"] = profile
                st.session_state["prediction"] = None  # invalidate old prediction
                st.success("Skills merged into your profile!")
                st.write("**Updated skills:**", merged)
