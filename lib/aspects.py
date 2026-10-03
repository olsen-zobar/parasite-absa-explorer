"""Tables and charts for the Aspect Explorer page (page 3)."""

import textwrap

import pandas as pd
import plotly.graph_objects as go

from lib.charts import GREY, POLARITY_COLORS, POLARITY_LABELS
from lib.data import POLARITY_ORDER, SMALL_N

IMPLICIT = "(implicit)"  # aspect_term_norm when no words name the aspect
NEGATIVE = ["very_negative", "negative"]
POSITIVE = ["positive", "very_positive"]


def category_polarity(opinions: pd.DataFrame) -> pd.DataFrame:
    """One row per category: tuple count per polarity, total n, mean score, share negative.
    Categories with at least SMALL_N tuples come first, largest first, and get a rank;
    smaller ones follow alphabetically with no rank."""
    counts = (
        opinions.groupby(["category", "polarity"], observed=False)
        .size()
        .unstack("polarity", fill_value=0)
        .reindex(columns=POLARITY_ORDER, fill_value=0)
    )
    counts.columns = [str(c) for c in counts.columns]
    table = counts.reset_index()
    table["n"] = table[POLARITY_ORDER].sum(axis=1)
    table = table[table["n"] > 0]
    means = opinions.groupby("category", observed=True)["polarity_score"].mean()
    table["mean_polarity"] = table["category"].map(means)
    table["pct_negative"] = 100 * table[NEGATIVE].sum(axis=1) / table["n"]
    table["small"] = table["n"] < SMALL_N

    big = table[~table["small"]].sort_values(["n", "category"], ascending=[False, True])
    small = table[table["small"]].sort_values("category")
    big = big.assign(rank=range(1, len(big) + 1))
    small = small.assign(rank=pd.NA)
    return pd.concat([big, small], ignore_index=True)


def diverging_bars(table: pd.DataFrame, share: bool, definitions: dict[str, str] | None = None):
    """Negative opinions extend left of zero, positive right. Small categories are grey.
    share=True plots each polarity as a % of the category's tuples, else raw counts."""
    definitions = definitions or {}
    labels = [
        f"{row.category}  (n={row.n})" if row.small else f"{row.rank}. {row.category}  ({row.n})"
        for row in table.itertuples()
    ]
    hover_defs = [
        "<br>".join(textwrap.wrap(definitions.get(c, ""), 70)) for c in table["category"]
    ]

    fig = go.Figure()
    # Traces nearest zero first, so 'negative' sits between zero and 'very negative'
    for polarity in ["negative", "very_negative", "positive", "very_positive"]:
        values = table[polarity].astype(float)
        if share:
            values = 100 * values / table["n"]
        sign = -1 if polarity in NEGATIVE else 1
        colours = [GREY if small else POLARITY_COLORS[polarity] for small in table["small"]]
        fig.add_bar(
            name=POLARITY_LABELS[polarity],
            y=labels,
            x=sign * values,
            orientation="h",
            marker_color=colours,
            marker_opacity=[0.45 if small else 1.0 for small in table["small"]],
            customdata=list(zip(table[polarity], values.round(1), hover_defs)),
            hovertemplate=(
                f"<b>%{{y}}</b><br>{POLARITY_LABELS[polarity]}: %{{customdata[0]}} tuples "
                "(%{customdata[1]}" + ("%" if share else "") + ")<br>"
                "<i>%{customdata[2]}</i><extra></extra>"
            ),
        )
    fig.update_layout(
        barmode="relative",
        height=max(320, 26 * len(table) + 90),
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, traceorder="normal"),
        xaxis_title="% of the category's opinions" if share else "opinion tuples",
    )
    fig.update_yaxes(categoryorder="array", categoryarray=labels, autorange="reversed")
    fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor="#1F2933")
    if share:
        # label the axis in plain percentages on both sides of zero
        ticks = [-100, -75, -50, -25, 0, 25, 50, 75, 100]
        fig.update_xaxes(range=[-100, 100], tickvals=ticks, ticktext=[f"{abs(t)}%" for t in ticks])
    return fig


def top_aspect_terms(opinions: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """The n most frequent normalised aspect terms (implicit aspects left out), one row per
    term and polarity, with columns aspect_term_norm, polarity, count, total."""
    named = opinions[opinions["aspect_term_norm"] != IMPLICIT]
    totals = named["aspect_term_norm"].value_counts().head(n)
    top = named[named["aspect_term_norm"].isin(totals.index)]
    long = (
        top.groupby(["aspect_term_norm", "polarity"], observed=True)
        .size()
        .reset_index(name="count")
    )
    long["polarity"] = long["polarity"].astype(str)
    long["total"] = long["aspect_term_norm"].map(totals)
    return long.sort_values(["total", "aspect_term_norm"], ascending=[False, True])


def term_bars(long: pd.DataFrame):
    """Stacked horizontal bars of top aspect terms, coloured by polarity."""
    order = list(dict.fromkeys(long["aspect_term_norm"]))
    fig = go.Figure()
    for polarity in POLARITY_ORDER:
        part = long[long["polarity"] == polarity]
        fig.add_bar(
            name=POLARITY_LABELS[polarity],
            y=part["aspect_term_norm"],
            x=part["count"],
            orientation="h",
            marker_color=POLARITY_COLORS[polarity],
        )
    fig.update_layout(
        barmode="stack",
        height=max(260, 28 * len(order) + 80),
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        xaxis_title="opinion tuples",
    )
    fig.update_yaxes(categoryorder="array", categoryarray=order, autorange="reversed")
    return fig


def top_opinion_terms(opinions: pd.DataFrame, polarities: list[str], n: int = 10) -> pd.DataFrame:
    """The n most frequent opinion words (lower-cased) among tuples of the given polarities."""
    words = opinions[opinions["polarity"].astype(str).isin(polarities)]["opinion_term"]
    counts = words.str.lower().str.strip().value_counts().head(n)
    return counts.rename_axis("opinion words").reset_index(name="tuples")


def sentence_spans(rows: pd.DataFrame) -> list[tuple[int, int, str]]:
    """Highlight spans for the tuples of one sentence, as positions inside that sentence.
    Offsets in opinions.csv count from the start of the review, so subtract the sentence's."""
    offset = int(rows["sentence_start"].iloc[0])
    spans = []
    for row in rows.itertuples():
        spans.append((int(row.opinion_start) - offset, int(row.opinion_end) - offset, str(row.polarity)))
        if pd.notna(row.aspect_start):
            spans.append((int(row.aspect_start) - offset, int(row.aspect_end) - offset, "aspect"))
    return list(dict.fromkeys(spans))  # one aspect can carry several opinions


def evidence_sentences(opinions: pd.DataFrame) -> pd.DataFrame:
    """One row per sentence that has at least one of the given tuples, in review order."""
    return (
        opinions.groupby(["review_id", "sentence_id"], observed=True)
        .agg(
            sentence=("sentence", "first"),
            criticName=("criticName", "first"),
            publicatioName=("publicatioName", "first"),
            reviewUrl=("reviewUrl", "first"),
            tuples=("polarity", "size"),
        )
        .reset_index()
        .sort_values(["publicatioName", "review_id", "sentence_id"])
        .reset_index(drop=True)
    )


def category_definitions(schema: dict | None) -> dict[str, str]:
    """category id -> 'Name: definition', from the schema JSON (empty if it is missing)."""
    if not schema:
        return {}
    return {c["id"]: f"{c['name']}: {c['definition']}" for c in schema.get("categories", [])}
