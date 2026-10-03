"""Shared sidebar filters. Values live in st.session_state so they carry across pages."""

import pandas as pd
import streamlit as st

from lib.data import CONFIDENCE_ORDER

VERDICTS = ["Fresh and rotten", "Fresh only", "Rotten only"]
CRITICS = ["All critics", "Top critics only", "Other critics only"]


def sidebar_filters() -> None:
    """Draw the filters. Call once per run, from app.py."""
    st.session_state.setdefault("f_confidence", "low")
    st.sidebar.header("Filters")
    st.sidebar.radio("Verdict", VERDICTS, key="f_verdict")
    st.sidebar.radio("Critics", CRITICS, key="f_critics")
    st.sidebar.select_slider(
        "Minimum annotator confidence",
        options=CONFIDENCE_ORDER,
        key="f_confidence",
        help="'low' keeps every tuple. 143 tuples are low confidence, many of them possibly "
        "about a different film the critic was comparing Parasite to.",
    )


def current_filters() -> dict:
    return {
        "verdict": st.session_state.get("f_verdict", VERDICTS[0]),
        "critics": st.session_state.get("f_critics", CRITICS[0]),
        "confidence": st.session_state.get("f_confidence", "low"),
    }


def apply_filters(df: pd.DataFrame, filters: dict | None = None) -> pd.DataFrame:
    """Keep rows matching the filters. Works on any frame with the review metadata columns."""
    f = filters or current_filters()
    mask = pd.Series(True, index=df.index)
    if f["verdict"] == "Fresh only":
        mask &= df["reviewState"] == "fresh"
    elif f["verdict"] == "Rotten only":
        mask &= df["reviewState"] == "rotten"
    if f["critics"] == "Top critics only":
        mask &= df["isTopCritic"]
    elif f["critics"] == "Other critics only":
        mask &= ~df["isTopCritic"]
    if "confidence" in df.columns:
        floor = CONFIDENCE_ORDER.index(f["confidence"])
        allowed = CONFIDENCE_ORDER[floor:]
        mask &= df["confidence"].astype(str).isin(allowed)
    return df[mask]


def is_filtered(filters: dict | None = None) -> bool:
    f = filters or current_filters()
    return (f["verdict"], f["critics"], f["confidence"]) != (VERDICTS[0], CRITICS[0], "low")
