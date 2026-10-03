"""Page 6 (Stars vs. Words) and page 7 (Keyword in context): helpers and AppTest checks."""

import math
import re
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from lib.kwic import concordance, keyword_pattern
from lib.stars import outliers, rating_to_unit, stars_vs_words

ROOT = Path(__file__).resolve().parent.parent
PASSWORD = "test-password"


def logged_in_at(page: str) -> AppTest:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.secrets["password"] = PASSWORD
    at.run()
    at.text_input[0].input(PASSWORD)
    at.button[0].click()
    at.run()
    at.switch_page(page)
    at.run()
    assert not at.exception
    return at


def corpus():
    reviews = pd.read_csv(ROOT / "data/reviews.csv").rename(columns={"reviewId": "review_id"})
    reviews = reviews[reviews["n_tuples"] > 0]
    return reviews, pd.read_csv(ROOT / "data/opinions.csv")


# --- Ratings ----------------------------------------------------------------
@pytest.mark.parametrize(
    "score, expected",
    [
        ("4/5", 0.8),
        ("9/10", 0.9),
        ("3.5/4", 0.875),
        ("2.0/5", 0.4),
        ("4/4", 1.0),
        ("5", 1.0),
        ("A+", 1.0),
        ("A", 11 / 12),
        ("A-", 10 / 12),
        ("B", 8 / 12),
        ("F", 0.0),
    ],
)
def test_rating_to_unit(score, expected):
    assert rating_to_unit(score) == pytest.approx(expected)


@pytest.mark.parametrize("score", [None, float("nan"), "", "great", "6/5", "2026-05-04 00:00:00"])
def test_rating_to_unit_rejects_junk(score):
    assert math.isnan(rating_to_unit(score))


def test_every_recovered_rating_is_readable():
    reviews, _ = corpus()
    scores = reviews["originalScore_recovered"].dropna()
    assert len(scores) == 64  # CLAUDE.md: 64 reviews have ratings
    assert scores.map(rating_to_unit).notna().all()


def test_stars_vs_words_matches_review_means():
    reviews, opinions = corpus()
    rated = stars_vs_words(reviews, opinions)
    assert len(rated) == 64
    # mean polarity comes from opinions.csv (reviews.csv's per-review counts are stale)
    means = opinions.groupby("review_id")["polarity_score"].mean()
    assert rated["mean_polarity"].to_numpy() == pytest.approx(
        means.loc[rated["review_id"]].to_numpy()
    )
    assert rated["residual"].sum() == pytest.approx(0, abs=1e-9)  # least-squares residuals
    worst = outliers(rated, 5)
    assert len(worst) == 5
    assert worst["residual"].abs().min() >= rated["residual"].abs().drop(worst.index).max()


# --- Keyword search ---------------------------------------------------------
def test_keyword_pattern_whole_word_and_case():
    p = keyword_pattern("Class", whole_word=True)
    assert p.search("a CLASS satire") and not p.search("a classic")
    assert keyword_pattern("class", whole_word=False).search("a classic")
    assert keyword_pattern("   ") is None


def test_keyword_pattern_treats_query_as_text():
    p = keyword_pattern("a.b (c)", whole_word=False)
    assert p.search("x a.b (c) y") and not p.search("aXb c")


def test_concordance_contexts():
    s = pd.DataFrame(
        {"review_id": [1], "sentence_id": [0], "text": ["Rich and poor; the poor stay poor."]}
    )
    hits = concordance(s, keyword_pattern("poor"), width=5)
    assert len(hits) == 3  # every occurrence, not just the first
    first = hits.iloc[0]
    assert first["keyword"] == "poor"
    assert first["left"] == "… and " and first["right"] == "; the…"
    assert s["text"][0][first["start"] : first["end"]] == "poor"


# --- Pages ------------------------------------------------------------------
def test_stars_page_loads():
    at = logged_in_at("pages/stars.py")
    assert at.title[0].value == "Stars vs. Words"
    assert any("Pearson r = 0.77" in m.value for m in at.markdown)
    assert len(at.dataframe) == 1  # the outlier table


def test_stars_page_survives_rotten_filter():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.secrets["password"] = PASSWORD
    at.session_state["authenticated"] = True
    at.session_state["f_verdict"] = "Rotten only"
    at.switch_page("pages/stars.py")
    at.run()
    assert not at.exception
    assert any("too few to draw a trend" in w.value for w in at.warning)


def test_kwic_page_default_search():
    at = logged_in_at("pages/kwic.py")
    assert at.title[0].value == "Keyword in context"
    sentences = pd.read_csv(ROOT / "data/sentences.csv")
    reviews, _ = corpus()
    sentences = sentences[sentences["review_id"].isin(reviews["review_id"])]
    expected = sentences["text"].str.count(re.compile(r"(?<!\w)class(?!\w)", re.I)).sum()
    got = {m.label: m.value for m in at.metric}
    assert got["Matches"] == str(expected)


def test_kwic_page_new_query_and_no_match():
    at = logged_in_at("pages/kwic.py")
    at.text_input(key="kwic_query").input("Song Kang-ho")
    at.run()
    assert not at.exception
    assert int({m.label: m.value for m in at.metric}["Matches"]) > 0
    at.text_input(key="kwic_query").input("zzqxj")
    at.run()
    assert not at.exception
    assert any("No sentence contains" in w.value for w in at.warning)
