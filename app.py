"""Phase 7 - Streamlit application entry point (Home page).

Run the whole app from the project root with:

    streamlit run app.py

Streamlit automatically turns every file in the pages/ folder into a page in
the left sidebar, so this file only needs to render the Home screen and do
one-time setup (initialize the database, prepare shared session state).
"""

from __future__ import annotations

import streamlit as st

from database.db import get_predictions, init_db
from utils.config import CAREERS
from utils.predictor import model_is_ready

# ---------------------------------------------------------------------------
# Page configuration - must be the first Streamlit call.
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CareerCompass AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


def bootstrap() -> None:
    """One-time setup: create DB tables and seed shared session_state keys."""
    init_db()
    if not model_is_ready():
        try:
            # Import your training function directly from your models module
            from models.train_model import train_and_select
            
            # Show a brief status message on startup so users know it's initializing
            st.info("Initializing application files and training the machine learning model. Please wait...")
            
            # Run the full train, evaluate, and save workflow
            train_and_select()
            
            st.success("Initialization complete! Model trained successfully.")
        except Exception as e:
            st.error(f"Failed to automatically train the model on startup: {e}")
    # Shared state used across pages. We set defaults only once.
    st.session_state.setdefault("profile", None)
    st.session_state.setdefault("parsed_resume", None)
    st.session_state.setdefault("prediction", None)


bootstrap()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🎓 Career Counsellor")
    st.caption("A local, ML-powered career guidance system.")
    st.divider()
    if model_is_ready():
        st.success("Model: ready")
    else:
        st.error("Model not trained.\nRun: `python -m models.train_model`")
    profile = st.session_state.get("profile")
    st.write("**Profile:**", profile["name"] if profile else "not set")
    st.divider()
    st.caption("Use the pages above in order: Profile → Resume → Prediction → Report.")

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.title("CareerCompass AI")
st.write(
    "Discover the tech career that best matches your skills, close your skill "
    "gaps, and prepare for interviews - all running locally with a real "
    "machine learning model."
)

col1, col2, col3 = st.columns(3)
col1.metric("Careers Covered", len(CAREERS))
col2.metric("Model", "Trained" if model_is_ready() else "Missing")
col3.metric("Predictions Made", len(get_predictions(limit=100000)))

st.divider()

st.subheader("How it works")
steps = st.columns(4)
steps[0].info("**1. Profile**\n\nEnter your academic details and skills.")
steps[1].info("**2. Resume**\n\nUpload a PDF to auto-extract skills.")
steps[2].info("**3. Prediction**\n\nGet your best-fit career + skill gap.")
steps[3].info("**4. Report**\n\nDownload a full career report.")

st.divider()

with st.expander("Careers this system can recommend"):
    st.write(", ".join(CAREERS))

if not model_is_ready():
    st.warning(
        "The ML model has not been trained yet. Open a terminal in the project "
        "root and run **`python -m models.train_model`**, then refresh this page."
    )
else:
    st.success("Everything is ready. Head to the **Student Profile** page to begin!")
