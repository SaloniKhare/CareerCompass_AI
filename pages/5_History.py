"""History page - view stored predictions and reports from SQLite.

Demonstrates reading back the data persisted by the SQLite layer (Phase 14).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from database.db import get_predictions, get_reports

st.set_page_config(page_title="History", page_icon="🗂️", layout="wide")
st.title("🗂️ History")
st.caption("Everything below is loaded from the local SQLite database.")

tab_pred, tab_reports = st.tabs(["Prediction history", "Saved reports"])

with tab_pred:
    predictions = get_predictions(limit=100)
    if not predictions:
        st.info("No predictions yet. Run a career analysis first.")
    else:
        df = pd.DataFrame(predictions)
        df["confidence"] = (df["confidence"] * 100).round(1).astype(str) + "%"
        st.dataframe(
            df[["id", "name", "career", "confidence", "created_at"]],
            hide_index=True,
            use_container_width=True,
        )

with tab_reports:
    reports = get_reports(limit=50)
    if not reports:
        st.info("No reports saved yet. Generate one on the Career Report page.")
    else:
        for r in reports:
            with st.expander(f"#{r['id']} - {r['name']} - {r['career']} ({r['created_at']})"):
                st.text(r["content"])
