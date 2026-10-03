"""AppTest checks for page 2, Review Explorer, plus the lib/review.py helpers it uses."""

import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "app.py")
PAGE = "pages/review_explorer.py"
PASSWORD = "test-password"
# Peter Bradshaw, Guardian: fresh, top critic, rated 4/5. opinions.csv has 12 tuples for it,
# 2 negative. (reviews.csv says n_tuples = 15: its per-review counts are stale, so the page
# counts from opinions.csv.)
BRADSHAW = 2588703


def open_page(query: dict | None = None, filters: dict | None = None) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.secrets["password"] = PASSWORD
    at.session_state["authenticated"] = True
    for key, value in (filters or {}).items():
        at.session_state[key] = value
    for key, value in (query or {}).items():
        at.query_params[key] = value
    at.switch_page(PAGE)
    at.run()
    assert not at.exception, at.exception
    return at


def gauge_values(at: AppTest) -> list[float]:
    figs = [json.loads(c.proto.spec) for c in at.get("plotly_chart")]
    return [d["value"] for f in figs for d in f["data"] if d["type"] == "indicator"]


def data():
    reviews = pd.read_csv(ROOT / "data/reviews.csv")
    reviews = reviews[reviews["n_tuples"] > 0]
    opinions = pd.read_csv(ROOT / "data/opinions.csv")
    sentences = pd.read_csv(ROOT / "data/sentences.csv")
    return reviews, opinions, sentences


def test_page_loads_and_lists_every_review():
    at = open_page()
    assert at.title[0].value == "Review Explorer"
    picker = at.selectbox(key="review_pick")
    assert len(picker.options) == 107
    assert len(at.get("plotly_chart")) == 3  # two gauges and the arc


def test_deep_link_opens_review_and_numbers_match_data():
    reviews, opinions, sentences = data()
    at = open_page(query={"review": str(BRADSHAW)})
    assert at.selectbox(key="review_pick").value == BRADSHAW
    assert "Peter Bradshaw" in at.header[0].value

    mine = opinions[opinions["review_id"] == BRADSHAW]
    mine_sentences = sentences[sentences["review_id"] == BRADSHAW]
    metrics = {m.label: m.value for m in at.metric}
    assert metrics == {
        "Sentences": str(len(mine_sentences)),
        "Sentences with an opinion": str(mine_sentences["has_opinion"].sum()),
        "Opinion tuples": str(len(mine)),
        "Categories": str(mine["category"].nunique()),
    }
    assert metrics["Opinion tuples"] == "12"

    mean, pct_negative = gauge_values(at)
    assert mean == pytest.approx(mine["polarity_score"].mean())
    assert pct_negative == pytest.approx(100 * 2 / 12)
    # the recovered rating, not the date the spreadsheet turned "4/5" into
    assert any("Rating: **4/5**" in m.value for m in at.markdown)


def test_picking_a_review_updates_the_page_and_link():
    at = open_page()
    at.selectbox(key="review_pick").set_value(BRADSHAW).run()
    assert not at.exception
    assert "Peter Bradshaw" in at.header[0].value
    assert at.query_params["review"] == str(BRADSHAW)


def test_verdict_filter_narrows_the_picker():
    at = open_page(filters={"f_verdict": "Rotten only"})
    picker = at.selectbox(key="review_pick")
    assert len(picker.options) == 5
    assert all("(rotten)" in o for o in picker.options)


def test_confidence_filter_hides_tuples():
    _, opinions, _ = data()
    at = open_page(query={"review": str(BRADSHAW)}, filters={"f_confidence": "high"})
    kept = opinions[(opinions["review_id"] == BRADSHAW) & (opinions["confidence"] == "high")]
    assert {m.label: m.value for m in at.metric}["Opinion tuples"] == str(len(kept))


def test_only_opinion_sentences_toggle():
    at = open_page(query={"review": str(BRADSHAW)})
    text_before = max((m.value for m in at.markdown), key=len)
    at.toggle[0].set_value(True).run()
    assert not at.exception
    text_after = max((m.value for m in at.markdown), key=len)
    assert len(text_after) < len(text_before)


# --- lib/review.py ----------------------------------------------------------


def test_every_highlight_lands_on_its_terms():
    """Across the whole corpus, every span cut out of its sentence matches the CSV terms."""
    from lib.review import sentence_spans

    _, opinions, sentences = data()
    merged = opinions.merge(sentences, on=["review_id", "sentence_id"])
    for (_, _), group in merged.groupby(["review_id", "sentence_id"]):
        text, start = group["text"].iloc[0], int(group["char_start"].iloc[0])
        expected = set(group["opinion_term"]) | set(group["aspect_term"].dropna())
        found = {text[s:e] for s, e, _ in sentence_spans(group, start)}
        assert found == expected


def test_sentence_arc_keeps_every_sentence_in_order():
    from lib.review import sentence_arc

    _, opinions, sentences = data()
    mine_sentences = sentences[sentences["review_id"] == BRADSHAW]
    arc = sentence_arc(mine_sentences, opinions[opinions["review_id"] == BRADSHAW])
    assert len(arc) == len(mine_sentences)
    assert arc["number"].tolist() == list(range(1, len(arc) + 1))
    assert arc["mean_polarity"].notna().sum() == mine_sentences["has_opinion"].sum()
    assert arc["n_tuples"].sum() == 12


def test_polarity_stats_on_empty_frame():
    from lib.review import polarity_stats

    assert polarity_stats(pd.DataFrame({"polarity_score": []})) == {
        "n": 0,
        "mean": None,
        "pct_negative": None,
    }
