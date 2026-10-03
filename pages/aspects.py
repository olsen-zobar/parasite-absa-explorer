"""Page 3: Aspect Explorer. Polarity per category, the most frequent terms, and the sentences."""

import streamlit as st

from lib.aspects import (
    IMPLICIT,
    NEGATIVE,
    POSITIVE,
    category_definitions,
    category_polarity,
    diverging_bars,
    evidence_sentences,
    sentence_spans,
    term_bars,
    top_aspect_terms,
    top_opinion_terms,
)
from lib.charts import POLARITY_LABELS, highlight_sentence, show_code, takeaway
from lib.data import POLARITY_ORDER, SMALL_N, load_opinions, load_schema
from lib.filters import apply_filters, is_filtered

PAGE_SIZE = 20
ALL = "All categories"

opinions = load_opinions()
definitions = category_definitions(load_schema())

st.title("Aspect Explorer")
st.markdown(
    "Which parts of *Parasite* did critics praise, and which did they criticise? Each opinion "
    "tuple is filed under one **ENTITY#ATTRIBUTE** category (ACTING#GENERAL, STORY#ENDING, …). "
    "Compare the categories, see the words critics used, then read the sentences behind them."
)

# --- Page filters (the sidebar filters apply as well) -----------------------
entities = sorted(opinions["entity"].unique())
cols = st.columns([3, 2])
picked = cols[0].multiselect(
    "Entities",
    entities,
    key="ax_entities",
    placeholder="All entities",
    help="The part before the # in a category. Leave empty to keep every entity.",
)
with_implicit = cols[1].toggle(
    "Include implicit aspects",
    value=True,
    key="ax_implicit",
    help="An implicit aspect is one the critic judges without naming it, e.g. 'It is a "
    "masterpiece.' 27% of tuples have one.",
)

shown = apply_filters(opinions)
if picked:
    shown = shown[shown["entity"].isin(picked)]
if not with_implicit:
    shown = shown[shown["aspect_explicit"]]

if is_filtered() or picked or not with_implicit:
    st.caption(f"Filters are on: counting {len(shown)} of {len(opinions)} tuples.")
if shown.empty:
    st.warning("No opinion tuples match these filters. Loosen them in the sidebar or above.")
    st.stop()

# --- Diverging bars per category ---------------------------------------------
st.header("Praise and criticism by category")
table = category_polarity(shown)
ranked = table[~table["small"]]
small = table[table["small"]]

share = (
    st.radio(
        "Bar length",
        ["Share of the category's opinions", "Number of opinions"],
        horizontal=True,
        key="ax_scale",
    )
    == "Share of the category's opinions"
)

if len(ranked):
    most_negative = ranked.sort_values(["pct_negative", "n"], ascending=[False, False]).iloc[0]
    if most_negative["pct_negative"] > 0:
        takeaway(
            f"Critics are positive about every part of the film, but {most_negative['category']} "
            f"draws the most criticism: {most_negative['pct_negative']:.0f}% of its "
            f"{most_negative['n']} opinions are negative."
        )
    else:
        takeaway(f"None of the {len(ranked)} ranked categories has a negative opinion here.")
else:
    takeaway(f"Every category here has fewer than {SMALL_N} tuples, so none is ranked.")

st.plotly_chart(diverging_bars(table, share, definitions), width="stretch")
st.caption(
    "Negative opinions extend left of the line, positive ones right. Categories are numbered "
    f"by size. Grey bars at the bottom have fewer than {SMALL_N} tuples: shown with their count, "
    "never ranked. Hover a bar for the category's definition."
)
show_code(
    """
counts = (
    opinions.groupby(["category", "polarity"])
    .size()
    .unstack("polarity", fill_value=0)
)
counts["n"] = counts.sum(axis=1)
share = counts.div(counts["n"], axis=0) * 100          # % of each category
big = counts[counts["n"] >= 10].sort_values("n", ascending=False)
"""
)
if definitions:
    with st.expander("What each category means"):
        for category in table["category"]:
            st.markdown(f"**{category}**: {definitions.get(category, '(not in the schema)')}")

# --- Top terms ----------------------------------------------------------------
st.header("The words critics used")
category_options = [ALL] + list(table["category"])
term_category = st.selectbox("Category", category_options, key="ax_term_category")
in_category = shown if term_category == ALL else shown[shown["category"] == term_category]

terms = top_aspect_terms(in_category)
left, right = st.columns([3, 2])
with left:
    st.subheader("Most-judged aspects")
    if terms.empty:
        st.info("Every tuple here has an implicit aspect, so there are no aspect terms to count.")
    else:
        top = terms.iloc[0]
        takeaway(
            f"'{top['aspect_term_norm']}' is the most-judged aspect"
            + ("" if term_category == ALL else f" in {term_category}")
            + f", with {top['total']} opinions."
        )
        st.plotly_chart(term_bars(terms), width="stretch")
        n_implicit = int((in_category["aspect_term_norm"] == IMPLICIT).sum())
        st.caption(
            f"Normalised aspect terms ('film' and 'movie' both count as 'the film'). "
            f"{n_implicit} tuples with an implicit aspect are left out."
        )
        show_code(
            """
named = opinions[opinions["aspect_term_norm"] != "(implicit)"]
top = named["aspect_term_norm"].value_counts().head(10)
by_polarity = (
    named[named["aspect_term_norm"].isin(top.index)]
    .groupby(["aspect_term_norm", "polarity"]).size()
)
"""
        )
with right:
    st.subheader("Most-used opinion words")
    praise = top_opinion_terms(in_category, POSITIVE)
    criticism = top_opinion_terms(in_category, NEGATIVE)
    if len(praise):
        best = praise[praise["tuples"] == praise["tuples"].iloc[0]]["opinion words"]
        words = " and ".join(f"'{w}'" for w in best) if len(best) <= 3 else f"'{best.iloc[0]}'"
        takeaway(f"The commonest praise: {words} "
            f"({praise['tuples'].iloc[0]} times{' each' if len(best) > 1 else ''}).")
    st.markdown("Praise (positive and very positive)")
    st.dataframe(praise, hide_index=True, width="stretch")
    st.markdown("Criticism (negative and very negative)")
    if criticism.empty:
        st.caption("No negative opinions here.")
    else:
        st.dataframe(criticism, hide_index=True, width="stretch")
    show_code(
        """
praise = opinions[opinions["polarity_score"] > 0]
praise["opinion_term"].str.lower().value_counts().head(10)
"""
    )

# --- Evidence -----------------------------------------------------------------
st.header("Read the evidence")
st.markdown(
    "Every bar above is made of sentences like these. Pick a category and polarity to read them. "
    "Underlined: aspect term. Coloured: opinion term, in its polarity colour."
)
cols = st.columns([2, 2, 2])
ev_category = cols[0].selectbox("Category", category_options, key="ax_ev_category")
ev_polarities = cols[1].multiselect(
    "Polarity",
    POLARITY_ORDER,
    default=POLARITY_ORDER,
    key="ax_ev_polarity",
    format_func=lambda p: POLARITY_LABELS[p],
)
pool = shown if ev_category == ALL else shown[shown["category"] == ev_category]
pool = pool[pool["polarity"].astype(str).isin(ev_polarities)]
term_options = ["Any aspect"] + sorted(
    t for t in pool["aspect_term_norm"].dropna().unique() if t != IMPLICIT
)
ev_term = cols[2].selectbox("Aspect term", term_options, key="ax_ev_term")
if ev_term != "Any aspect":
    pool = pool[pool["aspect_term_norm"] == ev_term]

evidence = evidence_sentences(pool)
if evidence.empty:
    st.info("No sentences match. Try another polarity or category.")
else:
    n_pages = (len(evidence) - 1) // PAGE_SIZE + 1
    page = 1
    if n_pages > 1:
        page = st.number_input(
            f"Page (of {n_pages})", min_value=1, max_value=n_pages, value=1, key="ax_ev_page"
        )
    first = (page - 1) * PAGE_SIZE
    st.caption(
        f"{len(pool)} tuples in {len(evidence)} sentences. "
        f"Showing sentences {first + 1}–{min(first + PAGE_SIZE, len(evidence))}."
    )
    for row in evidence.iloc[first : first + PAGE_SIZE].itertuples():
        rows = pool[(pool["review_id"] == row.review_id) & (pool["sentence_id"] == row.sentence_id)]
        labels = "; ".join(
            f"{r.category} · {POLARITY_LABELS[str(r.polarity)]}" for r in rows.itertuples()
        )
        st.markdown(
            f"<blockquote>{highlight_sentence(row.sentence, sentence_spans(rows))}</blockquote>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"{row.criticName}, *{row.publicatioName}* · [read the original review]"
            f"({row.reviewUrl}) · {labels}"
        )
    show_code(
        """
evidence = opinions[
    (opinions["category"] == "STORY#ENDING") & (opinions["polarity_score"] < 0)
]
evidence[["sentence", "aspect_term", "opinion_term", "polarity", "publicatioName"]]
"""
    )
