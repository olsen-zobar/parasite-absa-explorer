"""Page 7: Keyword in context. Search a word, see every sentence that uses it and its labels."""

import html

import streamlit as st

from lib.charts import POLARITY_LABELS, polarity_bar, show_code, takeaway
from lib.data import POLARITY_ORDER, load_opinions, load_reviews, load_sentences
from lib.filters import apply_filters, is_filtered
from lib.kwic import concordance, keyword_pattern

MAX_SENTENCES = 20
META = ["review_id", "criticName", "publicatioName", "isTopCritic", "reviewState", "reviewUrl"]


def mark_keyword(sentence: str, spans: list[tuple[int, int]]) -> str:
    """HTML for a sentence with each keyword match highlighted."""
    out, pos = [], 0
    for start, end in sorted(spans):
        out.append(html.escape(sentence[pos:start]))
        out.append(
            '<mark style="background:#FFE08A;padding:0 2px;border-radius:2px">'
            f"{html.escape(sentence[start:end])}</mark>"
        )
        pos = end
    out.append(html.escape(sentence[pos:]))
    return "".join(out)


opinions = load_opinions()
labels = apply_filters(opinions)  # sidebar filters, including confidence, apply to the labels
sentences = load_sentences().merge(load_reviews()[META], on="review_id")
sentences = apply_filters(sentences)  # verdict and critic filters apply to the sentences

st.title("Keyword in context")
st.markdown(
    "A *concordance* lists every place a word occurs with the words on either side, so you can "
    "see how critics actually use it. Search any word or phrase; each match comes with the "
    "opinion labels the annotators gave its sentence."
)

c1, c2, c3 = st.columns([3, 1, 1])
query = c1.text_input("Word or phrase", value="class", key="kwic_query")
whole_word = c2.checkbox(
    "Whole words only",
    value=True,
    key="kwic_whole",
    help="Off: 'class' also finds 'classic' and 'classes'.",
)
opinion_only = c3.checkbox("Only sentences with an opinion", value=False, key="kwic_opinion")

pattern = keyword_pattern(query, whole_word)
if pattern is None:
    st.info("Type a word or phrase to search the reviews.")
    st.stop()

pool = sentences[sentences["has_opinion"]] if opinion_only else sentences
hits = concordance(pool, pattern)
if hits.empty:
    st.warning(f"No sentence contains “{query.strip()}”. Try turning off *Whole words only*.")
    st.stop()

hit_keys = hits[["review_id", "sentence_id"]].drop_duplicates()
hit_sentences = pool.merge(hit_keys, on=["review_id", "sentence_id"])
hit_labels = labels.merge(hit_keys, on=["review_id", "sentence_id"])

m1, m2, m3, m4 = st.columns(4)
m1.metric("Matches", len(hits))
m2.metric("Sentences", len(hit_sentences))
m3.metric("Reviews", hit_sentences["review_id"].nunique())
m4.metric("Opinion labels in those sentences", len(hit_labels))
if is_filtered():
    st.caption("Sidebar filters are on: they narrow the reviews searched and the labels shown.")

# --- Polarity of the labels -------------------------------------------------
st.header("How the sentences that use it are labelled")
if hit_labels.empty:
    takeaway(f"None of the sentences using “{query.strip()}” carries an opinion label.")
else:
    counts = (
        hit_labels["polarity"]
        .value_counts()
        .reindex(POLARITY_ORDER, fill_value=0)
        .rename_axis("polarity")
        .reset_index(name="n")
    )
    positive = int(counts.loc[counts["polarity"].isin(POLARITY_ORDER[2:]), "n"].sum())
    share = 100 * positive / len(hit_labels)
    takeaway(
        f"{share:.0f}% of the {len(hit_labels)} opinions in sentences using “{query.strip()}” "
        f"are positive, against {100 * (labels['polarity_score'] > 0).mean():.0f}% across "
        + ("all opinions under the current filters." if is_filtered() else "the whole corpus.")
    )
    st.plotly_chart(polarity_bar(counts), width="stretch")
    st.caption(
        "These are all the opinions in the matching sentences. The keyword is not necessarily "
        "the aspect or the opinion term itself: check the sentences below."
    )
    show_code(
        """
query = "class"
# (?<!\\w) and (?!\\w) stop "class" matching inside "classic"
pattern = re.compile(rf"(?<!\\w){re.escape(query)}(?!\\w)", re.IGNORECASE)
hits = sentences[sentences["text"].str.contains(pattern)]
labels = opinions.merge(hits[["review_id", "sentence_id"]], on=["review_id", "sentence_id"])
labels["polarity"].value_counts()
"""
    )
    top = (
        hit_labels.groupby("category", observed=True)
        .size()
        .sort_values(ascending=False)
        .head(5)
        .rename("opinions")
        .reset_index()
    )
    st.markdown("**Most common categories in these sentences**")
    st.dataframe(top, hide_index=True)

# --- Concordance ------------------------------------------------------------
st.header("Concordance")
table = hits.merge(sentences[META[:3]].drop_duplicates(), on="review_id")
st.dataframe(
    table[["left", "keyword", "right", "criticName", "publicatioName"]],
    hide_index=True,
    width="stretch",
    column_config={
        "left": st.column_config.TextColumn("before", width="large"),
        "keyword": st.column_config.TextColumn("keyword", width="small"),
        "right": st.column_config.TextColumn("after", width="large"),
        "criticName": "critic",
        "publicatioName": "publication",
    },
)

# --- Sentences with labels --------------------------------------------------
st.header("Each sentence and its labels")
st.caption(
    f"{len(hit_sentences)} sentences. Showing the first {min(MAX_SENTENCES, len(hit_sentences))}."
    if len(hit_sentences) > MAX_SENTENCES
    else f"{len(hit_sentences)} sentences."
)
spans = hits.groupby(["review_id", "sentence_id"])[["start", "end"]].apply(
    lambda g: list(zip(g["start"], g["end"]))
)
for row in hit_sentences.sort_values(["review_id", "sentence_id"]).head(MAX_SENTENCES).itertuples():
    st.markdown(
        f"<blockquote>{mark_keyword(row.text, spans[(row.review_id, row.sentence_id)])}"
        "</blockquote>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"{row.criticName}, *{row.publicatioName}* · [read the original review]({row.reviewUrl})"
    )
    mine = hit_labels[
        (hit_labels["review_id"] == row.review_id) & (hit_labels["sentence_id"] == row.sentence_id)
    ]
    if mine.empty:
        st.caption("No opinion labelled in this sentence.")
    else:
        st.dataframe(
            mine.assign(polarity=mine["polarity"].map(POLARITY_LABELS))[
                ["aspect_term_norm", "category", "opinion_term", "polarity", "confidence"]
            ].rename(columns={"aspect_term_norm": "aspect"}),
            hide_index=True,
            width="stretch",
        )
