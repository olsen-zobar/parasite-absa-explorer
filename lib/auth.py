"""Password gate. The full review text is copyrighted, so every page sits behind it."""

import hmac

import streamlit as st


def _expected_password() -> str | None:
    try:
        return st.secrets["password"]
    except (KeyError, FileNotFoundError):
        return None


def _check(entered: str, expected: str) -> bool:
    return hmac.compare_digest(entered.encode("utf-8"), expected.encode("utf-8"))


def require_password() -> None:
    """Stop the script unless the visitor has entered the class password."""
    if st.session_state.get("authenticated"):
        return

    expected = _expected_password()
    if expected is None:
        st.error(
            "No password is configured. Copy `.streamlit/secrets.toml.example` to "
            "`.streamlit/secrets.toml` and set `password`."
        )
        st.stop()

    st.title("Parasite ABSA Explorer")
    st.write("This site quotes copyrighted film criticism, so it is limited to the class.")
    with st.form("login"):
        entered = st.text_input("Class password", type="password")
        submitted = st.form_submit_button("Enter")

    if submitted:
        if _check(entered, expected):
            st.session_state["authenticated"] = True
            st.rerun()
        st.error("That password is not correct.")
    st.stop()
