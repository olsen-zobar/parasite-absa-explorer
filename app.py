"""Parasite ABSA Explorer: entry point. Run with `streamlit run app.py`."""

import streamlit as st

from lib.auth import require_password
from lib.filters import sidebar_filters

st.set_page_config(page_title="Parasite ABSA Explorer", page_icon="🎬", layout="wide")

require_password()

pages = [
    st.Page("pages/home.py", title="Home", icon="🏠", default=True),
    st.Page("pages/methods.py", title="Methods & Data", icon="📚"),
    st.Page("pages/cast_crew.py", title="Cast & Crew", icon="🎭"),
]
nav = st.navigation(pages)
sidebar_filters()
nav.run()
