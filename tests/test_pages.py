"""AppTest checks: the password gate, both pages, and the Home headline numbers."""

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "app.py")
PASSWORD = "test-password"


def fresh_app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.secrets["password"] = PASSWORD
    return at


def logged_in() -> AppTest:
    at = fresh_app()
    at.run()
    at.text_input[0].input(PASSWORD)
    at.button[0].click()
    at.run()
    assert not at.exception
    return at


def metrics(at: AppTest) -> dict[str, str]:
    return {m.label: m.value for m in at.metric}


def test_gate_blocks_without_password():
    at = fresh_app()
    at.run()
    assert not at.exception
    assert len(at.text_input) == 1
    assert not at.metric  # no page content leaks before login


def test_gate_rejects_wrong_password():
    at = fresh_app()
    at.run()
    at.text_input[0].input("wrong")
    at.button[0].click()
    at.run()
    assert any("not correct" in e.value for e in at.error)
    assert not at.metric


def test_home_loads_with_password():
    at = logged_in()
    assert at.title[0].value == "Parasite ABSA Explorer"
    assert len(at.metric) == 8


def test_home_headline_numbers_match_data():
    reviews = pd.read_csv(ROOT / "data/reviews.csv")
    reviews = reviews[reviews["n_tuples"] > 0]
    opinions = pd.read_csv(ROOT / "data/opinions.csv")
    sentences = pd.read_csv(ROOT / "data/sentences.csv")
    sentences = sentences[sentences["review_id"].isin(reviews["reviewId"])]
    n_fresh = (reviews["reviewState"] == "fresh").sum()
    n_rotten = (reviews["reviewState"] == "rotten").sum()

    expected = {
        "Reviews": str(len(reviews)),
        "Opinion tuples": str(len(opinions)),
        "Categories used": str(opinions["category"].nunique()),
        "Fresh / rotten": f"{n_fresh} / {n_rotten}",
        "Publications": str(reviews["publicatioName"].nunique()),
        "Top critics": str(reviews["isTopCritic"].sum()),
        "Sentences": str(len(sentences)),
        "Sentences with an opinion": str(sentences["has_opinion"].sum()),
    }
    assert metrics(logged_in()) == expected
    # and the figures the project documents in data/SUMMARY.md
    assert expected["Reviews"] == "107"
    assert expected["Opinion tuples"] == "2045"
    assert expected["Categories used"] == "22"
    assert expected["Fresh / rotten"] == "102 / 5"


def test_methods_page_loads():
    at = logged_in()
    at.switch_page("pages/methods.py")
    at.run()
    assert not at.exception
    assert at.title[0].value == "Methods & Data"
    headers = [h.value for h in at.header]
    for section in ["The annotation schema", "Data dictionary", "Limitations", "Download the data"]:
        assert section in headers


@pytest.mark.parametrize(
    "filters, expected",
    [
        ({"verdict": "Rotten only", "critics": "All critics", "confidence": "low"}, 84),
        ({"verdict": "Fresh and rotten", "critics": "All critics", "confidence": "medium"}, 1902),
    ],
)
def test_filters(filters, expected):
    from lib.data import load_opinions
    from lib.filters import apply_filters

    assert len(apply_filters(load_opinions(), filters)) == expected
