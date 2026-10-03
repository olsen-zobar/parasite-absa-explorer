"""Helpers for the Review Explorer page: picking a review, its numbers, arc and highlights."""

import pandas as pd
import plotly.graph_objects as go

from lib.charts import POLARITY_COLORS

GAUGE_BAR = "#1F2933"


def review_label(row) -> str:
    """How a review appears in the picker: "Critic, Publication (fresh)"."""
    return f"{row.criticName}, {row.publicatioName} ({row.reviewState})"


def star_rating(row) -> str | None:
    """The critic's own rating. Some raw scores were mangled into dates by a spreadsheet
    ("4/5" became 2026-05-04), so use the recovered value, which fixes those."""
    value = row.originalScore_recovered
    return None if pd.isna(value) else str(value)


def polarity_stats(tuples: pd.DataFrame) -> dict:
    """Mean polarity score (−2..+2) and share of negative tuples."""
    n = len(tuples)
    if n == 0:
        return {"n": 0, "mean": None, "pct_negative": None}
    return {
        "n": n,
        "mean": float(tuples["polarity_score"].mean()),
        "pct_negative": float(100 * (tuples["polarity_score"] < 0).mean()),
    }


def polarity_class(score: float) -> str:
    """The polarity name nearest to a (mean) score, for colouring."""
    if score <= -1.5:
        return "very_negative"
    if score < 0:
        return "negative"
    if score < 1.5:
        return "positive"
    return "very_positive"


def sentence_arc(sentences: pd.DataFrame, tuples: pd.DataFrame) -> pd.DataFrame:
    """One row per sentence of a review, in order, with the mean polarity of its tuples
    (NaN for sentences that carry no opinion)."""
    per_sentence = tuples.groupby("sentence_id").agg(
        mean_polarity=("polarity_score", "mean"), n_tuples=("polarity_score", "size")
    )
    arc = sentences[["sentence_id", "text"]].sort_values("sentence_id").reset_index(drop=True)
    arc = arc.merge(per_sentence, on="sentence_id", how="left")
    arc["n_tuples"] = arc["n_tuples"].fillna(0).astype(int)
    arc["number"] = arc.index + 1  # what readers see: sentence 1, 2, 3 …
    return arc


def sentence_spans(tuples: pd.DataFrame, sentence_start: int) -> list[tuple[int, int, str]]:
    """Highlight spans for one sentence. Offsets in opinions.csv count from the start of the
    review, so subtract the sentence's own start to get positions inside the sentence."""
    spans = []
    for row in tuples.itertuples():
        spans.append(
            (int(row.opinion_start) - sentence_start, int(row.opinion_end) - sentence_start,
             str(row.polarity))
        )
        if pd.notna(row.aspect_start):
            spans.append(
                (int(row.aspect_start) - sentence_start, int(row.aspect_end) - sentence_start,
                 "aspect")
            )
    return list(dict.fromkeys(spans))  # one aspect can carry several opinions


def gauge(value: float, low: float, high: float, corpus: float, title: str, suffix: str = ""):
    """A Plotly gauge with a threshold marker at the corpus value."""
    if suffix == "%":
        steps = [
            {"range": [0, 25], "color": POLARITY_COLORS["positive"]},
            {"range": [25, 50], "color": "#D1E5F0"},
            {"range": [50, 75], "color": "#FDDBC7"},
            {"range": [75, 100], "color": POLARITY_COLORS["negative"]},
        ]
    else:
        steps = [
            {"range": [-2, -1], "color": POLARITY_COLORS["very_negative"]},
            {"range": [-1, 0], "color": POLARITY_COLORS["negative"]},
            {"range": [0, 1], "color": POLARITY_COLORS["positive"]},
            {"range": [1, 2], "color": POLARITY_COLORS["very_positive"]},
        ]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": suffix, "valueformat": ".0f" if suffix == "%" else "+.2f"},
            title={"text": title},
            gauge={
                "axis": {"range": [low, high]},
                "bar": {"color": GAUGE_BAR, "thickness": 0.25},
                "steps": steps,
                "threshold": {"line": {"color": "black", "width": 4}, "value": corpus},
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=30, r=30, t=60, b=10))
    return fig


def arc_chart(arc: pd.DataFrame):
    """Bars of mean polarity per sentence, in reading order. Gaps are sentences with no
    opinion."""
    rated = arc.dropna(subset=["mean_polarity"])
    fig = go.Figure(
        go.Bar(
            x=rated["number"],
            y=rated["mean_polarity"],
            marker_color=[POLARITY_COLORS[polarity_class(v)] for v in rated["mean_polarity"]],
            customdata=list(zip(rated["n_tuples"], rated["text"].str.slice(0, 90))),
            hovertemplate="Sentence %{x}: mean %{y:+.2f} from %{customdata[0]} tuple(s)"
            "<br>%{customdata[1]}…<extra></extra>",
        )
    )
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), bargap=0.15)
    fig.update_xaxes(title="sentence (in reading order)", range=[0.5, len(arc) + 0.5])
    fig.update_yaxes(title="mean polarity", range=[-2.2, 2.2], zeroline=True, zerolinewidth=2)
    return fig
