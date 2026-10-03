"""AppTest checks for page 5, Cast & Crew, and its roster and photo credits."""

import pandas as pd
import pytest

from lib.people import CREDIT_COLUMNS, CREDITS_PATH, NO_PHOTO, PEOPLE_DIR, roster
from tests.test_pages import ROOT, logged_in

PAGE = "pages/cast_crew.py"
NON_PEOPLE = {"Kim family", "Park family", "Moon-gwang", "the cast", "the ensemble"}


def cast_crew_page():
    at = logged_in()
    at.switch_page(PAGE)
    at.run()
    assert not at.exception
    return at


def test_page_loads():
    at = cast_crew_page()
    assert at.title[0].value == "Cast & Crew"
    headers = [h.value for h in at.header]
    for section in [
        "Who gets the praise?",
        "The crew",
        "The cast",
        "What critics said about one person",
        "Characters and groups",
        "Photo credits",
    ]:
        assert section in headers


def test_roster_covers_every_person_in_the_data():
    targets = set(pd.read_csv(ROOT / "data/opinions.csv")["target"].dropna())
    names = set(roster()["target"])
    assert names <= targets, "roster names must match the data's spelling"
    assert targets - names == NON_PEOPLE


def test_person_picker_shows_matching_evidence():
    at = cast_crew_page()
    picker = at.selectbox(key="cast_person")
    assert picker.value == "Bong Joon-ho"  # the most discussed person comes first in the roster
    picker.set_value("Hong Kyung-pyo").run()
    assert not at.exception
    assert any("53 tuples" in c.value for c in at.caption)
    assert any("Hong Kyung-pyo" in s.value for s in at.subheader)


def test_small_people_are_listed_not_ranked():
    at = cast_crew_page()
    small = at.dataframe[0].value  # the greyed table of people under 10 opinions
    assert (small["n"] < 10).all()
    assert list(small["target"]) == sorted(small["target"])


def test_filters_change_the_counts():
    at = cast_crew_page()
    at.session_state["f_verdict"] = "Rotten only"
    at.run()
    assert not at.exception
    assert any("Sidebar filters are on" in c.value for c in at.caption)
    picker = at.selectbox(key="cast_person")
    assert "Song Kang-ho" not in picker.options  # no rotten review mentions him


def test_credits_file_is_complete():
    if not CREDITS_PATH.exists():
        pytest.skip("no photos fetched yet")
    credits = pd.read_csv(CREDITS_PATH, dtype=str).fillna("")
    assert list(credits.columns) == CREDIT_COLUMNS
    assert set(credits["person"]) <= set(roster()["target"]) - NO_PHOTO
    for row in credits.itertuples():
        assert (PEOPLE_DIR / row.file).exists(), row.file
        assert row.author and row.source_url.startswith("https://commons.wikimedia.org/")
        assert row.licence.lower().startswith(("cc", "public domain", "pd")), row.licence
