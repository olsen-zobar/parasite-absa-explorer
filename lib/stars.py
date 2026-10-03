"""Star ratings on one scale, and how far each review's words stray from its stars.

Critics grade in fractions (4/5, 9/10, 3.5/4) and letters (A-, B). To compare them we map
every rating to 0..1. Ratings come from `originalScore_recovered`, which was reconstructed
from a damaged export, so treat these numbers as provisional.
"""

import re

import numpy as np
import pandas as pd

# F to A+ in 13 equal steps: F=0, D-=1/12, ... A=11/12, A+=1
LETTERS = ["F", "D-", "D", "D+", "C-", "C", "C+", "B-", "B", "B+", "A-", "A", "A+"]
LETTER_SCALE = {g: i / (len(LETTERS) - 1) for i, g in enumerate(LETTERS)}
BARE_NUMBER_OUT_OF = 5  # a bare "5" is read as 5 out of 5

_FRACTION = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*$")
_NUMBER = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*$")


def rating_to_unit(score) -> float:
    """'4/5' -> 0.8, 'A-' -> 0.83, '5' -> 1.0. Anything unreadable -> NaN."""
    if not isinstance(score, str):
        return np.nan
    if m := _FRACTION.match(score):
        top, bottom = float(m.group(1)), float(m.group(2))
        return top / bottom if bottom and top <= bottom else np.nan
    if m := _NUMBER.match(score):
        value = float(m.group(1))
        return value / BARE_NUMBER_OUT_OF if value <= BARE_NUMBER_OUT_OF else np.nan
    return LETTER_SCALE.get(score.strip().upper(), np.nan)


def stars_vs_words(reviews: pd.DataFrame, opinions: pd.DataFrame) -> pd.DataFrame:
    """One row per rated review: its rating (0..1), the mean polarity of its tuples, and how
    far that mean sits from the straight-line fit (residual). Reviews whose tuples were all
    filtered out are dropped."""
    words = (
        opinions.groupby("review_id")
        .agg(mean_polarity=("polarity_score", "mean"), tuples=("polarity_score", "size"))
        .reset_index()
    )
    rated = reviews.assign(rating=reviews["originalScore_recovered"].map(rating_to_unit))
    rated = rated.dropna(subset=["rating"]).merge(words, on="review_id", how="inner")
    if len(rated) >= 2 and rated["rating"].nunique() > 1:
        slope, intercept = np.polyfit(rated["rating"], rated["mean_polarity"], 1)
    else:
        slope, intercept = 0.0, rated["mean_polarity"].mean() if len(rated) else 0.0
    rated["predicted"] = intercept + slope * rated["rating"]
    rated["residual"] = rated["mean_polarity"] - rated["predicted"]
    return rated.reset_index(drop=True)


def outliers(rated: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """The n reviews whose words are furthest from what their stars predict."""
    return rated.reindex(rated["residual"].abs().sort_values(ascending=False).index).head(n)
