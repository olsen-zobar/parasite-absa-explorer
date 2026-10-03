"""Colours and the takeaway / show-the-code pattern every chart uses."""

import html

import pandas as pd
import plotly.express as px
import streamlit as st

from lib.data import POLARITY_ORDER

POLARITY_COLORS = {
    "very_negative": "#B2182B",
    "negative": "#EF8A62",
    "positive": "#67A9CF",
    "very_positive": "#2166AC",
}
POLARITY_LABELS = {
    "very_negative": "very negative (−2)",
    "negative": "negative (−1)",
    "positive": "positive (+1)",
    "very_positive": "very positive (+2)",
}
GREY = "#9AA5B1"


def takeaway(text: str) -> None:
    """The one-sentence point of the chart, shown above it."""
    st.markdown(f"**{text}**")


def show_code(code: str) -> None:
    """The few lines of pandas behind a chart or number, shown below it."""
    with st.expander("Show the code"):
        st.code(code.strip(), language="python")


def polarity_bar(counts: pd.DataFrame):
    """Horizontal bar of tuple counts per polarity. Expects columns polarity, n."""
    fig = px.bar(
        counts,
        x="n",
        y="polarity",
        orientation="h",
        color="polarity",
        color_discrete_map=POLARITY_COLORS,
        category_orders={"polarity": POLARITY_ORDER},
        text="n",
        labels={"n": "opinion tuples", "polarity": ""},
    )
    fig.update_layout(showlegend=False, height=260, margin=dict(l=10, r=10, t=10, b=10))
    fig.update_yaxes(ticktext=[POLARITY_LABELS[p] for p in POLARITY_ORDER], tickvals=POLARITY_ORDER)
    return fig


def highlight_sentence(sentence: str, spans: list[tuple[int, int, str]]) -> str:
    """HTML for a sentence with spans marked. spans: (start, end, kind) where kind is a
    polarity name (filled in its colour) or "aspect" (underlined)."""
    out, pos = [], 0
    for start, end, kind in sorted(spans):
        if start < pos:  # overlapping span: skip rather than garble the text
            continue
        out.append(html.escape(sentence[pos:start]))
        piece = html.escape(sentence[start:end])
        if kind == "aspect":
            out.append(f'<span style="border-bottom:3px solid #1F2933">{piece}</span>')
        else:
            colour = POLARITY_COLORS[kind]
            out.append(
                f'<span style="background:{colour};color:white;padding:0 4px;'
                f'border-radius:3px">{piece}</span>'
            )
        pos = end
    out.append(html.escape(sentence[pos:]))
    return "".join(out)
