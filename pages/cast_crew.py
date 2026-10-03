"""Page 5: Cast & Crew. How critics judge the people behind the film (the `target` column)."""

import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.charts import GREY, POLARITY_COLORS, POLARITY_LABELS, highlight_sentence, show_code, takeaway
from lib.data import POLARITY_ORDER, SMALL_N, load_opinions
from lib.filters import apply_filters, is_filtered
from lib.people import load_credits, photo_for, roster

opinions = load_opinions()
shown = apply_filters(opinions)
people = roster()

st.title("Cast & Crew")
st.markdown(
    "When a critic praises *Parasite*, who gets the credit? Every opinion whose subject is a "
    "named person carries that person in the **target** column, whether the critic wrote "
    "\"Bong\", \"the director\" or \"his camera\". This page counts those opinions per person."
)
if is_filtered():
    st.caption(f"Sidebar filters are on: showing {len(shown)} of {len(opinions)} tuples.")


def person_stats(df: pd.DataFrame) -> pd.DataFrame:
    """One row per target: tuple count, review count, mean polarity, count per polarity."""
    stats = df.groupby("target").agg(
        n=("polarity_score", "size"),
        reviews=("review_id", "nunique"),
        mean_polarity=("polarity_score", "mean"),
    )
    counts = pd.crosstab(df["target"], df["polarity"]).reindex(columns=POLARITY_ORDER, fill_value=0)
    return stats.join(counts).reset_index()


stats = people.merge(person_stats(shown[shown["target"].notna()]), on="target", how="left")
stats[["n", "reviews", *POLARITY_ORDER]] = stats[["n", "reviews", *POLARITY_ORDER]].fillna(0).astype(int)
corpus_mean = shown["polarity_score"].mean()
ranked = stats[stats["n"] >= SMALL_N].sort_values("n", ascending=False)
small = stats[stats["n"] < SMALL_N].sort_values("target")

# --- Praise by person -------------------------------------------------------
st.header("Who gets the praise?")
if ranked.empty:
    st.info("No one has 10 or more opinions under the current filters.")
else:
    top = ranked.iloc[0]
    total = max(int(stats["n"].sum()), 1)
    negative = int(stats[["very_negative", "negative"]].sum().sum())
    takeaway(
        f"{top['target']} draws {top['n'] / total:.0%} of all opinions about named people, "
        f"and only {negative / total:.0%} of those opinions are negative."
    )
    fig = go.Figure()
    for polarity in POLARITY_ORDER:
        sign = -1 if polarity in ("very_negative", "negative") else 1
        fig.add_bar(
            y=ranked["target"],
            x=sign * ranked[polarity],
            orientation="h",
            name=POLARITY_LABELS[polarity],
            marker_color=POLARITY_COLORS[polarity],
            customdata=ranked[polarity],
            hovertemplate="%{y}: %{customdata} " + POLARITY_LABELS[polarity] + "<extra></extra>",
        )
    fig.update_layout(
        barmode="relative",
        height=80 + 40 * len(ranked),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=-0.15),
        xaxis_title="opinion tuples (negative to the left, positive to the right)",
    )
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width="stretch")
show_code(
    """
people = opinions[opinions["target"].notna()]
counts = pd.crosstab(people["target"], people["polarity"])
counts[["very_negative", "negative"]] *= -1      # negatives point left
counts = counts[counts.sum(axis=1).abs() >= 10]   # rank only people with 10+ opinions
counts.plot.barh(stacked=True)
"""
)

if not small.empty:
    st.markdown(
        f"**Fewer than {SMALL_N} opinions.** Too few to compare, so these are listed "
        "alphabetically, not ranked."
    )
    st.dataframe(
        small[["target", "role", "n", "mean_polarity"]].style.set_properties(color=GREY),
        hide_index=True,
        width="stretch",
        column_config={
            "target": "person",
            "n": "opinions",
            "mean_polarity": st.column_config.NumberColumn("mean polarity", format="%+.2f"),
        },
    )


# --- Cards ------------------------------------------------------------------
def initials_tile(name: str) -> str:
    letters = "".join(part[0] for part in name.replace("-", " ").split()[:2])
    return (
        '<div style="aspect-ratio:3/4;background:#E4E7EB;color:#52606D;display:flex;'
        'align-items:center;justify-content:center;font-size:2.5rem;border-radius:6px">'
        f"{html.escape(letters)}</div>"
    )


def card(person: pd.Series) -> None:
    path, credit = photo_for(person["target"])
    if path is not None:
        st.image(str(path), width="stretch")
        st.caption(
            f"Photo: {credit['author']}, [{credit['licence']}]({credit['licence_url'] or credit['source_url']}), "
            f"via [Wikimedia Commons]({credit['source_url']})"
        )
    else:
        st.markdown(initials_tile(person["target"]), unsafe_allow_html=True)
        st.caption("No free photo available")
    st.markdown(f"**{person['target']}**  \n{person['role']}")
    if person["n"] == 0:
        st.markdown("No opinions under these filters")
    else:
        colour = "" if person["n"] >= SMALL_N else f"color:{GREY}"
        st.markdown(
            f'<span style="{colour}">{person["n"]} opinions in {person["reviews"]} reviews · '
            f"mean {person['mean_polarity']:+.2f}</span>",
            unsafe_allow_html=True,
        )


for group, title in [("Crew", "The crew"), ("Cast", "The cast")]:
    st.header(title)
    members = stats[stats["group"] == group]
    for start in range(0, len(members), 4):
        cols = st.columns(4)
        for col, (_, person) in zip(cols, members.iloc[start : start + 4].iterrows()):
            with col:
                card(person)


# --- One person in detail ---------------------------------------------------
def gauge(value: float, threshold: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"valueformat": "+.2f"},
            gauge={
                "axis": {"range": [-2, 2]},
                "bar": {"color": POLARITY_COLORS["very_positive" if value >= 0 else "very_negative"]},
                "threshold": {"line": {"color": "#1F2933", "width": 3}, "value": threshold},
            },
        )
    )
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=30, b=10))
    return fig


st.header("What critics said about one person")
names = stats.loc[stats["n"] > 0, "target"].tolist()
if names:
    name = st.selectbox("Person", names, key="cast_person")
    person = stats[stats["target"] == name].iloc[0]
    evidence = shown[shown["target"] == name]

    left, right = st.columns([1, 2])
    with left:
        st.plotly_chart(gauge(person["mean_polarity"], corpus_mean), width="stretch")
        st.caption(
            f"Mean polarity of {person['n']} opinions about {name}. The black mark is the mean "
            f"of every opinion in the corpus ({corpus_mean:+.2f})."
        )
        if person["n"] < SMALL_N:
            st.warning(f"Only {person['n']} opinions: too few to compare with others.")
    with right:
        by_category = (
            evidence.groupby("category").size().sort_values(ascending=False).rename("opinions")
        )
        takeaway(f"Critics mostly judge {name} under {by_category.index[0]}.")
        st.dataframe(by_category.reset_index(), hide_index=True, width="stretch")
    show_code(
        f"""
about = opinions[opinions["target"] == "{name}"]
about["polarity_score"].mean()            # the gauge
about["category"].value_counts()          # the table
"""
    )

    st.subheader(f"The sentences behind {name}'s numbers")
    st.caption(
        f"{len(evidence)} tuples. Showing up to 30, most negative first so criticism is not "
        "buried. Coloured: the opinion words. Underlined: the aspect term."
    )
    for row in evidence.sort_values(["polarity_score", "review_id"]).head(30).itertuples():
        offset = row.sentence_start
        spans = [(row.opinion_start - offset, row.opinion_end - offset, row.polarity)]
        if pd.notna(row.aspect_start):
            spans.append((int(row.aspect_start) - offset, int(row.aspect_end) - offset, "aspect"))
        st.markdown(
            f"{highlight_sentence(row.sentence, spans)}<br>"
            f'<small>{POLARITY_LABELS[row.polarity]} · {html.escape(row.category)} · '
            f"{html.escape(row.criticName)}, <i>{html.escape(row.publicatioName)}</i> · "
            f'<a href="{html.escape(row.reviewUrl)}" target="_blank">original review</a></small>',
            unsafe_allow_html=True,
        )

# --- Characters and groups --------------------------------------------------
st.header("Characters and groups")
st.markdown(
    "Some targets are not one real person: a family of characters, or \"the cast\" as a "
    "whole. They are kept apart from the people above."
)
others = person_stats(shown[shown["target"].notna() & ~shown["target"].isin(people["target"])])
st.dataframe(
    others.sort_values("n", ascending=False)[["target", "n", "reviews", "mean_polarity"]],
    hide_index=True,
    width="stretch",
    column_config={
        "n": "opinions",
        "mean_polarity": st.column_config.NumberColumn("mean polarity", format="%+.2f"),
    },
)

# --- Photo credits ----------------------------------------------------------
st.header("Photo credits")
credits = load_credits()
if credits.empty:
    st.markdown("No photos have been fetched yet. Run `python scripts/fetch_people_photos.py`.")
else:
    st.markdown(
        "Every photo is freely licensed and comes from Wikimedia Commons. The full list, with "
        "licence links, is in `assets/people/credits.csv`."
    )
    st.dataframe(
        credits[["person", "author", "licence", "source_url"]],
        hide_index=True,
        width="stretch",
        column_config={"source_url": st.column_config.LinkColumn("source")},
    )
