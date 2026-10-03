"""Page 3, Aspect Explorer: the page runs, and its tables agree with plain pandas."""

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from lib.aspects import (
    IMPLICIT,
    category_polarity,
    diverging_bars,
    evidence_sentences,
    sentence_spans,
    top_aspect_terms,
    top_opinion_terms,
)
from lib.charts import POLARITY_LABELS
from lib.data import SMALL_N, load_opinions

ROOT = Path(__file__).resolve().parent.parent
PASSWORD = "test-password"


def aspect_page() -> AppTest:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.secrets["password"] = PASSWORD
    at.run()
    at.text_input[0].input(PASSWORD)
    at.button[0].click()
    at.run()
    at.switch_page("pages/aspects.py")
    at.run()
    assert not at.exception
    return at


@pytest.fixture(scope="module")
def opinions() -> pd.DataFrame:
    return load_opinions()


def test_page_loads():
    at = aspect_page()
    assert at.title[0].value == "Aspect Explorer"
    headers = [h.value for h in at.header]
    for section in ["Praise and criticism by category", "The words critics used", "Read the evidence"]:
        assert section in headers
    assert any("draws the most criticism" in m.value for m in at.markdown)
    # evidence sentences link back to the original articles
    assert any("read the original review" in c.value for c in at.caption)


def test_page_survives_filters():
    at = aspect_page()
    at.radio(key="ax_scale").set_value("Number of opinions").run()
    at.multiselect(key="ax_entities").set_value(["STORY"]).run()
    at.toggle(key="ax_implicit").set_value(False).run()
    at.selectbox(key="ax_ev_category").set_value("STORY#ENDING").run()
    at.multiselect(key="ax_ev_polarity").set_value(["negative", "very_negative"]).run()
    assert not at.exception
    assert any("Filters are on" in c.value for c in at.caption)


def test_page_handles_no_matches():
    at = aspect_page()
    at.multiselect(key="ax_ev_polarity").set_value([]).run()
    assert not at.exception
    assert any("No sentences match" in i.value for i in at.info)


def test_category_counts_match_pandas(opinions):
    table = category_polarity(opinions).set_index("category")
    raw = pd.read_csv(ROOT / "data/opinions.csv")
    expected = raw.groupby(["category", "polarity"]).size().unstack(fill_value=0)
    for category, row in expected.iterrows():
        for polarity, n in row.items():
            assert table.loc[category, polarity] == n
    assert table["n"].sum() == len(raw) == 2045
    assert table.loc["MOVIE#GENERAL", "rank"] == 1


def test_small_categories_last_and_unranked(opinions):
    table = category_polarity(opinions)
    small = table[table["small"]]
    assert set(small["category"]) == {
        "MOVIE#REWATCHABILITY",
        "EDITING#GENERAL",
        "TRANSLATION#SUBTITLES",
    }
    assert (small["n"] < SMALL_N).all()
    assert small["rank"].isna().all()
    assert list(small.index) == list(range(len(table) - len(small), len(table)))
    ranked = table[~table["small"]]
    assert list(ranked["rank"]) == list(range(1, len(ranked) + 1))
    assert ranked["n"].is_monotonic_decreasing


def test_diverging_bars_signs_and_shares(opinions):
    table = category_polarity(opinions)
    fig = diverging_bars(table, share=True)
    by_name = {
        polarity: next(t for t in fig.data if t.name == label)
        for polarity, label in POLARITY_LABELS.items()
    }
    assert all(x <= 0 for x in by_name["very_negative"].x)
    assert all(x <= 0 for x in by_name["negative"].x)
    assert all(x >= 0 for x in by_name["positive"].x)
    assert all(x >= 0 for x in by_name["very_positive"].x)
    # in share mode the four polarities of each category add up to 100%
    totals = sum(abs(pd.Series(t.x)) for t in fig.data)
    assert totals.round(6).eq(100).all()


def test_top_terms_leave_out_implicit(opinions):
    terms = top_aspect_terms(opinions)
    assert IMPLICIT not in set(terms["aspect_term_norm"])
    assert terms["aspect_term_norm"].iloc[0] == "the film"
    assert terms.groupby("aspect_term_norm")["count"].sum().max() == 368
    assert terms["aspect_term_norm"].nunique() == 10


def test_top_opinion_words(opinions):
    praise = top_opinion_terms(opinions, ["positive", "very_positive"])
    assert set(praise.iloc[:2]["opinion words"]) == {"masterful", "brilliant"}
    assert praise["tuples"].is_monotonic_decreasing


def test_highlight_spans_land_on_the_words(opinions):
    sample = opinions.sample(200, random_state=0)
    for row in sample.itertuples():
        rows = opinions[
            (opinions["review_id"] == row.review_id) & (opinions["sentence_id"] == row.sentence_id)
        ]
        spans = sentence_spans(rows)
        opinion_span = (
            row.opinion_start - row.sentence_start,
            row.opinion_end - row.sentence_start,
            str(row.polarity),
        )
        assert opinion_span in spans
        assert row.sentence[opinion_span[0] : opinion_span[1]] == row.opinion_term


def test_evidence_one_row_per_sentence(opinions):
    pool = opinions[opinions["category"] == "STORY#ENDING"]
    evidence = evidence_sentences(pool)
    assert evidence["tuples"].sum() == len(pool)
    assert not evidence.duplicated(["review_id", "sentence_id"]).any()
