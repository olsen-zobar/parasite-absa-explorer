"""Page 1: Home. What ABSA is, a worked example, headline numbers, the pipeline."""

import pandas as pd
import streamlit as st

from lib.charts import POLARITY_LABELS, highlight_sentence, polarity_bar, show_code, takeaway
from lib.data import ASSETS, POLARITY_ORDER, load_opinions, load_reviews, load_sentences
from lib.filters import apply_filters, is_filtered

# The worked example: Peter Bradshaw (Guardian), first sentence of his review
EXAMPLE_REVIEW, EXAMPLE_SENTENCE = 2588703, 0

reviews = load_reviews()
sentences = load_sentences()
opinions = load_opinions()

st.title("Parasite ABSA Explorer")
st.markdown(
    "How did film critics judge *Parasite* (기생충, 2019, dir. Bong Joon-ho)? This site breaks "
    "107 critic reviews down into hundreds of small, checkable opinions about the film's "
    "acting, direction, story, design and more."
)

# --- What ABSA is -----------------------------------------------------------
st.header("What is aspect-based sentiment analysis?")
st.markdown(
    "Ordinary sentiment analysis gives a whole review one score: good or bad. **Aspect-based "
    "sentiment analysis (ABSA)** asks a finer question: *what* is the critic praising or "
    "criticising, and *how strongly*? Each opinion becomes a **tuple**, a small record with:\n"
    "- **aspect term**: the words naming what is judged (\"suspense drama\"), or nothing when "
    "the subject is only implied;\n"
    "- **category**: which part of the film that is, as ENTITY#ATTRIBUTE "
    "(STORY#GENRE, ACTING#GENERAL, …);\n"
    "- **opinion term**: the critic's exact evaluative words (\"luxuriously watchable\");\n"
    "- **polarity**: very negative (−2), negative (−1), positive (+1) or very positive (+2)."
)

st.subheader("A worked example")
example = opinions[
    (opinions["review_id"] == EXAMPLE_REVIEW) & (opinions["sentence_id"] == EXAMPLE_SENTENCE)
]
sentence = example["sentence"].iloc[0]
offset = example["sentence_start"].iloc[0]
spans = []
for row in example.itertuples():
    spans.append((row.opinion_start - offset, row.opinion_end - offset, row.polarity))
    if pd.notna(row.aspect_start):
        spans.append((int(row.aspect_start) - offset, int(row.aspect_end) - offset, "aspect"))
spans = list(dict.fromkeys(spans))  # one aspect can carry several opinions

critic, outlet = example["criticName"].iloc[0], example["publicatioName"].iloc[0]
st.markdown(
    f'<blockquote style="font-size:1.15rem">{highlight_sentence(sentence, spans)}</blockquote>',
    unsafe_allow_html=True,
)
st.caption(
    f"{critic}, *{outlet}* · [read the original review]({example['reviewUrl'].iloc[0]}). "
    "Underlined: aspect term. Coloured: opinion terms, in their polarity colour."
)
st.markdown(
    "This one sentence yields **two tuples**. The same aspect, *suspense drama*, is judged "
    "twice: as a film overall, and as an example of its genre. *Luxuriously* intensifies "
    "*watchable*, so that opinion is very positive."
)
st.dataframe(
    example[["aspect_term", "category", "opinion_term", "polarity", "polarity_score"]],
    hide_index=True,
    width="stretch",
)

# --- Headline numbers -------------------------------------------------------
st.header("The corpus in numbers")
n_fresh = int((reviews["reviewState"] == "fresh").sum())
n_rotten = int((reviews["reviewState"] == "rotten").sum())
cols = st.columns(4)
cols[0].metric("Reviews", len(reviews))
cols[1].metric("Opinion tuples", len(opinions))
cols[2].metric("Categories used", opinions["category"].nunique())
cols[3].metric("Fresh / rotten", f"{n_fresh} / {n_rotten}")
cols = st.columns(4)
cols[0].metric("Publications", reviews["publicatioName"].nunique())
cols[1].metric("Top critics", int(reviews["isTopCritic"].sum()))
cols[2].metric("Sentences", len(sentences))
cols[3].metric("Sentences with an opinion", int(sentences["has_opinion"].sum()))
show_code(
    """
reviews = pd.read_csv("data/reviews.csv")
reviews = reviews[reviews["n_tuples"] > 0]   # drop the one non-Parasite review
opinions = pd.read_csv("data/opinions.csv")

len(reviews)                                 # reviews
len(opinions)                                # opinion tuples
opinions["category"].nunique()               # categories used
reviews["reviewState"].value_counts()        # fresh / rotten
"""
)

# --- Polarity distribution --------------------------------------------------
st.subheader("How positive is the corpus?")
shown = apply_filters(opinions)
counts = (
    shown["polarity"].value_counts().reindex(POLARITY_ORDER, fill_value=0).rename_axis("polarity")
)
counts = counts.reset_index(name="n")
positive_share = counts.loc[counts["polarity"].isin(["positive", "very_positive"]), "n"].sum()
share = positive_share / max(len(shown), 1)
takeaway(
    f"{share:.0%} of opinions are positive, because Rotten Tomatoes rated almost every review "
    "in this sample fresh: this describes the sample, not a verdict on the film."
)
if is_filtered():
    st.caption(f"Sidebar filters are on: showing {len(shown)} of {len(opinions)} tuples.")
st.plotly_chart(polarity_bar(counts), width="stretch")
st.caption(
    "Positive opinions have not yet been audited for intensity the way negatives were, so "
    "compare positive with negative, not very positive with very negative."
)
show_code(
    """
order = ["very_negative", "negative", "positive", "very_positive"]
counts = opinions["polarity"].value_counts().reindex(order)
px.bar(counts, orientation="h")
"""
)
with st.expander("See the sentences behind these bars"):
    pick = st.selectbox(
        "Polarity", POLARITY_ORDER, index=0, format_func=lambda p: POLARITY_LABELS[p]
    )
    evidence = shown[shown["polarity"] == pick]
    st.caption(f"{len(evidence)} tuples. Showing up to 20.")
    st.dataframe(
        evidence[["opinion_term", "category", "sentence", "criticName", "publicatioName"]].head(20),
        hide_index=True,
        width="stretch",
    )

# --- Pipeline ---------------------------------------------------------------
st.header("How the data was made")
st.markdown(
    "Reviews were collected from Rotten Tomatoes, their full text fetched from each original "
    "publication, and every opinion annotated against a 23-category schema, then checked by a "
    "deterministic validator. The Methods & Data page has the details and the limitations."
)
# pipeline.png is an SVG file despite its extension; st.image renders SVG markup
st.image((ASSETS / "pipeline.png").read_text(encoding="utf-8"), width="stretch")

# --- Credits ----------------------------------------------------------------
st.header("Credits and data source")
st.markdown(
    "- Built for *Coding for Film Studies*, Columbia University.\n"
    "- Reviews: critic reviews listed on **Rotten Tomatoes**, with full text from each "
    "**original publication**. Every review page in this site links back to the original "
    "article; the text remains the property of its authors and publishers.\n"
    "- Annotation: one LLM pass per review, followed by a deterministic validator that checks "
    "every span and label (see Methods & Data)."
)
