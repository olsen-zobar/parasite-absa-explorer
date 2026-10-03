"""Cached data loaders. Every page gets its data from here, never from the CSVs directly."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ASSETS = ROOT / "assets"
SCHEMA_PATH = DATA / "parasite_absa_schema_v1.1.json"

POLARITY_ORDER = ["very_negative", "negative", "positive", "very_positive"]
CONFIDENCE_ORDER = ["low", "medium", "high"]
SMALL_N = 10  # categories with fewer tuples are shown greyed out and never ranked


@st.cache_data
def load_reviews() -> pd.DataFrame:
    """One row per review. Drops the review with no opinion tuples (not a Parasite review)."""
    reviews = pd.read_csv(DATA / "reviews.csv").rename(columns={"reviewId": "review_id"})
    reviews = reviews[reviews["n_tuples"] > 0].reset_index(drop=True)
    # creationDate is when Rotten Tomatoes added the review, not when it was published
    reviews = reviews.rename(columns={"creationDate": "rt_added_date"})
    return reviews


@st.cache_data
def load_sentences() -> pd.DataFrame:
    """One row per sentence, for the annotated reviews only."""
    sentences = pd.read_csv(DATA / "sentences.csv")
    keep = load_reviews()["review_id"]
    return sentences[sentences["review_id"].isin(keep)].reset_index(drop=True)


@st.cache_data
def load_opinions() -> pd.DataFrame:
    """One row per opinion tuple, with its sentence and the review metadata the filters need."""
    opinions = pd.read_csv(DATA / "opinions.csv")
    reviews = load_reviews()[
        ["review_id", "criticName", "publicatioName", "isTopCritic", "reviewState", "reviewUrl"]
    ]
    sentences = load_sentences()[["review_id", "sentence_id", "char_start", "text"]]
    sentences = sentences.rename(columns={"text": "sentence", "char_start": "sentence_start"})
    opinions = opinions.merge(reviews, on="review_id", how="inner")
    opinions = opinions.merge(sentences, on=["review_id", "sentence_id"], how="left")
    opinions["polarity"] = pd.Categorical(opinions["polarity"], POLARITY_ORDER, ordered=True)
    opinions["confidence"] = pd.Categorical(
        opinions["confidence"], CONFIDENCE_ORDER, ordered=True
    )
    return opinions


@st.cache_data
def load_schema() -> dict | None:
    """The category definitions, or None while the schema file is not in the repo."""
    if not SCHEMA_PATH.exists():
        return None
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def raw_csv_bytes(name: str) -> bytes:
    """A data file exactly as stored, for the download buttons."""
    return (DATA / name).read_bytes()
