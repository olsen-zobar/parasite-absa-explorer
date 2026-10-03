"""Page 2: Review Explorer. Pick one review and see how its opinions are spread through it."""

import html

import streamlit as st

from lib.charts import GREY, highlight_sentence, show_code, takeaway
from lib.data import load_opinions, load_reviews, load_sentences
from lib.filters import apply_filters, current_filters, is_filtered
from lib.review import (
    arc_chart,
    gauge,
    polarity_stats,
    review_label,
    sentence_arc,
    sentence_spans,
    star_rating,
)

reviews = load_reviews()
sentences = load_sentences()
opinions = load_opinions()

st.title("Review Explorer")
st.markdown(
    "Pick one critic's review and read it the way the annotation sees it: every opinion "
    "coloured by polarity, every aspect underlined, and a chart of how the mood moves "
    "from the first sentence to the last."
)

# --- Pick a review ----------------------------------------------------------
# The verdict and critic filters narrow the list; the confidence filter applies to tuples
listed = apply_filters(reviews).sort_values(["criticName", "publicatioName"])
if listed.empty:
    st.info("No review matches the sidebar filters. Loosen them to pick a review.")
    st.stop()

options = listed["review_id"].tolist()
labels = {row.review_id: review_label(row) for row in listed.itertuples()}

# A link such as ?review=2588703 opens that review (other pages use this to cite evidence)
requested = st.query_params.get("review")
if "review_pick" not in st.session_state and requested and requested.isdigit():
    st.session_state["review_pick"] = int(requested)
if st.session_state.get("review_pick") not in options:
    st.session_state["review_pick"] = options[0]


def _remember_pick() -> None:
    st.query_params["review"] = str(st.session_state["review_pick"])


review_id = st.selectbox(
    f"Review ({len(options)} available)",
    options,
    format_func=labels.get,
    key="review_pick",
    on_change=_remember_pick,
)
if is_filtered():
    st.caption(f"Sidebar filters are on: {len(options)} of {len(reviews)} reviews listed.")

review = reviews[reviews["review_id"] == review_id].iloc[0]
review_sentences = sentences[sentences["review_id"] == review_id]
all_tuples = opinions[opinions["review_id"] == review_id]
tuples = apply_filters(all_tuples)

# --- Header card ------------------------------------------------------------
with st.container(border=True):
    st.header(f"{review['criticName']}, *{review['publicatioName']}*")
    badges = [":tomato: Fresh" if review["reviewState"] == "fresh" else ":green_apple: Rotten"]
    if review["isTopCritic"]:
        badges.append(":star: Top critic")
    rating = star_rating(review)
    badges.append(f"Rating: **{rating}**" if rating else "No star rating")
    st.markdown(" · ".join(badges))
    st.caption(
        f"Added to Rotten Tomatoes on {review['rt_added_date']} (not necessarily the "
        f"publication date) · [Read the original review]({review['reviewUrl']})"
    )
    cols = st.columns(4)
    cols[0].metric("Sentences", len(review_sentences))
    cols[1].metric("Sentences with an opinion", int(review_sentences["has_opinion"].sum()))
    cols[2].metric("Opinion tuples", len(tuples))
    cols[3].metric("Categories", tuples["category"].nunique())
    if len(tuples) < len(all_tuples):
        st.caption(
            f"The confidence filter hides {len(all_tuples) - len(tuples)} of this review's "
            f"{len(all_tuples)} tuples."
        )

if tuples.empty:
    st.info("This review has no tuples at the chosen confidence level. Lower it in the sidebar.")
    st.stop()

# --- Gauges -----------------------------------------------------------------
st.subheader("How positive is this review?")
mine = polarity_stats(tuples)
# The corpus line uses every review, at the same confidence level as the review itself
corpus_filters = {**current_filters(), "verdict": "Fresh and rotten", "critics": "All critics"}
corpus = polarity_stats(apply_filters(opinions, corpus_filters))
direction = "above" if mine["mean"] >= corpus["mean"] else "below"
takeaway(
    f"This review's opinions average {mine['mean']:+.2f} on the −2 to +2 scale, {direction} the "
    f"corpus mean of {corpus['mean']:+.2f}; {mine['pct_negative']:.0f}% of them are negative "
    f"(corpus: {corpus['pct_negative']:.0f}%)."
)
cols = st.columns(2)
cols[0].plotly_chart(gauge(mine["mean"], -2, 2, corpus["mean"], "Mean polarity"), width="stretch")
cols[1].plotly_chart(
    gauge(mine["pct_negative"], 0, 100, corpus["pct_negative"], "Negative opinions", "%"),
    width="stretch",
)
st.caption("The black line on each gauge marks the whole corpus, for comparison.")
show_code(
    f"""
opinions = pd.read_csv("data/opinions.csv")
mine = opinions[opinions["review_id"] == {review_id}]

mine["polarity_score"].mean()               # mean polarity, -2 .. +2
(mine["polarity_score"] < 0).mean() * 100   # % negative
opinions["polarity_score"].mean()           # the corpus line on the gauge
"""
)

# --- Review arc -------------------------------------------------------------
st.subheader("The review's arc")
arc = sentence_arc(review_sentences, tuples)
negative = arc.loc[arc["mean_polarity"] < 0, "number"].tolist()
if not negative:
    takeaway(
        "Every opinion in this review is positive; the bars show where the praise is "
        "strongest."
    )
else:
    where = ", ".join(str(n) for n in negative[:6]) + (" …" if len(negative) > 6 else "")
    takeaway(
        f"The criticism sits in sentence{'s' if len(negative) > 1 else ''} {where} "
        f"of {len(arc)}; read them below to see what the critic objects to."
    )
st.plotly_chart(arc_chart(arc), width="stretch")
st.caption(
    "One bar per sentence that carries an opinion, at the mean polarity of its tuples. "
    "Gaps are sentences with no opinion (plot summary, context, quotes)."
)
show_code(
    f"""
sentences = pd.read_csv("data/sentences.csv")
mine = opinions[opinions["review_id"] == {review_id}]
arc = mine.groupby("sentence_id")["polarity_score"].mean()
px.bar(arc)
"""
)

# --- Annotated text ---------------------------------------------------------
st.subheader("The annotated text")
st.markdown(
    "Underlined: aspect terms. Coloured: opinion terms, in their polarity colour "
    '(<span style="color:#B2182B">very negative</span>, '
    '<span style="color:#EF8A62">negative</span>, '
    '<span style="color:#67A9CF">positive</span>, '
    '<span style="color:#2166AC">very positive</span>). Grey: sentences with no opinion. '
    "Numbers match the bars above.",
    unsafe_allow_html=True,
)
only_opinions = st.toggle("Show only sentences with an opinion", value=False)

by_sentence = dict(tuple(tuples.groupby("sentence_id")))
paragraphs = []
for row in arc.itertuples():
    start = int(review_sentences.loc[review_sentences["sentence_id"] == row.sentence_id,
                                     "char_start"].iloc[0])
    number = f'<sup style="color:{GREY}">{row.number}</sup> '
    if row.sentence_id in by_sentence:
        spans = sentence_spans(by_sentence[row.sentence_id], start)
        paragraphs.append(number + highlight_sentence(row.text, spans))
    elif not only_opinions:
        paragraphs.append(f'{number}<span style="color:{GREY}">{html.escape(row.text)}</span>')
st.markdown(
    '<div style="line-height:1.9;font-size:1.05rem">' + " ".join(paragraphs) + "</div>",
    unsafe_allow_html=True,
)
st.caption(
    f"Text © {review['criticName']} / {review['publicatioName']}, quoted for teaching. "
    f"[Read it in the original]({review['reviewUrl']})."
)

with st.expander(f"See all {len(tuples)} opinion tuples in this review"):
    table = tuples.merge(arc[["sentence_id", "number"]], on="sentence_id")
    st.dataframe(
        table.sort_values(["number", "opinion_start"])[
            ["number", "aspect_term", "category", "opinion_term", "polarity", "confidence",
             "target"]
        ].rename(columns={"number": "sentence"}),
        hide_index=True,
        width="stretch",
    )
