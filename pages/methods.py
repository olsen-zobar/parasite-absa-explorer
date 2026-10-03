"""Page 4: Methods & Data. Schema, pipeline, data dictionary, limitations, downloads."""

import pandas as pd
import streamlit as st

from lib.charts import GREY, show_code, takeaway
from lib.data import ASSETS, SMALL_N, load_opinions, load_schema, raw_csv_bytes
from lib.filters import apply_filters, is_filtered

opinions = load_opinions()
shown = apply_filters(opinions)

st.title("Methods & Data")
st.markdown(
    "How the corpus was built, what every column means, and what the numbers can and cannot "
    "tell you. Read the limitations before quoting any chart."
)

# --- Schema -----------------------------------------------------------------
st.header("The annotation schema")
st.markdown(
    "Every opinion is filed under one **ENTITY#ATTRIBUTE** category, the format used by the "
    "SemEval ABSA shared tasks. The schema defines 23 categories for *Parasite*; 22 appear in "
    "the data, because CHARACTER#DESIGN was retired during piloting (judgements of how a role "
    "is written now go under ACTING#GENERAL)."
)

table = (
    shown.groupby(["category", "entity", "attribute"], observed=True)
    .agg(tuples=("polarity_score", "size"), mean_polarity=("polarity_score", "mean"))
    .reset_index()
)
table["share_%"] = (100 * table["tuples"] / max(len(shown), 1)).round(1)
table["mean_polarity"] = table["mean_polarity"].round(2)
# Rank only categories with enough tuples; small ones go last, alphabetically, greyed out
big = table[table["tuples"] >= SMALL_N].sort_values("tuples", ascending=False)
small = table[table["tuples"] < SMALL_N].sort_values("category")
table = pd.concat([big, small], ignore_index=True)

schema = load_schema()
if schema is None:
    st.info(
        "Category definitions come from `data/parasite_absa_schema_v1.1.json`, which is not in "
        "the repository yet. Until it is added, this table lists the categories found in the "
        "data."
    )

takeaway(
    f"{big['category'].iloc[0]} alone holds {big['share_%'].iloc[0]:.0f}% of all opinions; "
    f"{len(small)} categories have fewer than {SMALL_N} tuples and are too small to rank."
)
if is_filtered():
    st.caption(f"Sidebar filters are on: counting {len(shown)} of {len(opinions)} tuples.")


def _grey_small(row: pd.Series) -> list[str]:
    style = f"color: {GREY}" if row["tuples"] < SMALL_N else ""
    return [style] * len(row)


st.dataframe(
    table.style.apply(_grey_small, axis=1).format({"mean_polarity": "{:+.2f}", "share_%": "{:.1f}"}),
    hide_index=True,
    width="stretch",
    column_config={
        "mean_polarity": st.column_config.NumberColumn(
            "mean polarity", help="Average score from −2 (very negative) to +2 (very positive)"
        ),
        "share_%": st.column_config.NumberColumn("share %"),
    },
)
st.caption(f"Grey rows have fewer than {SMALL_N} tuples: shown for completeness, never ranked.")
show_code(
    """
table = (
    opinions.groupby(["category", "entity", "attribute"])
    .agg(tuples=("polarity_score", "size"), mean_polarity=("polarity_score", "mean"))
    .reset_index()
)
table["share_%"] = 100 * table["tuples"] / len(opinions)
big = table[table["tuples"] >= 10].sort_values("tuples", ascending=False)
"""
)
with st.expander("See the sentences behind a category"):
    category = st.selectbox("Category", table["category"])
    evidence = shown[shown["category"] == category]
    st.caption(f"{len(evidence)} tuples. Showing up to 20.")
    st.dataframe(
        evidence[["aspect_term", "opinion_term", "polarity", "sentence", "publicatioName"]].head(20),
        hide_index=True,
        width="stretch",
    )

# --- Pipeline ---------------------------------------------------------------
st.header("The pipeline")
st.image((ASSETS / "pipeline.png").read_text(encoding="utf-8"), width="stretch")
st.markdown(
    """
1. **Collect.** A Rotten Tomatoes export of critic reviews: 954 rows, 477 unique reviews.
2. **Fetch full text.** Rotten Tomatoes only gives a one-line snippet, so each review's full text
   was fetched from its original publication. 108 English reviews could be recovered; the rest
   were paywalled, blocked or dead links.
3. **Guard.** A page is rejected unless it mentions *Parasite* or Bong Joon-ho and is over 400
   characters. This caught a review of a different 1982 film called *Parasite*. One more
   recovered page turned out to be a quotation about Cinemascope, not a review; it is excluded,
   leaving **107 reviews**.
4. **Annotate.** One LLM pass per review, with the whole review in view and sentences numbered.
   An opinion needs an explicit sentiment phrase of 1 to 6 words copied from the text. The aspect
   may be implicit (27% of tuples). *Very* positive or negative requires an intensifier, a
   superlative or an extreme word.
5. **Validate.** A deterministic validator re-reads every stored span from the original text
   (0 mismatches) and checks every label. A manual audit of all negative tuples corrected 25 rows.
6. **Check consistency.** 10 reviews were annotated twice independently: 98% of matched opinions
   got the same polarity, but only 77% of opinions were picked both times.
"""
)

# --- Data dictionary --------------------------------------------------------
st.header("Data dictionary")
DICTIONARY = {
    "opinions.csv: one row per opinion tuple": {
        "review_id, sentence_id": "Keys back to reviews.csv and sentences.csv",
        "aspect_term": "The critic's exact words for what is judged; empty when implicit",
        "aspect_start, aspect_end": "Character offsets of the aspect term in the review text",
        "aspect_explicit": "False when no noun phrase names the aspect",
        "aspect_term_norm": "Normalised aspect: use this for counting ('film', 'movie' → 'the film')",
        "category, entity, attribute": "ENTITY#ATTRIBUTE category and its two halves",
        "target": "The person judged (actor, character, crew), when there is one",
        "opinion_term": "The critic's exact sentiment words (1–6 words)",
        "opinion_start, opinion_end": "Character offsets of the opinion term in the review text",
        "polarity, polarity_score": "very_negative −2, negative −1, positive +1, very_positive +2",
        "mixed_within_aspect": "The same aspect is judged both ways in one sentence",
        "confidence": "Annotator confidence: low, medium or high",
        "note": "Any change the validator made to this row",
    },
    "reviews.csv: one row per review": {
        "reviewId": "Rotten Tomatoes review id (review_id in the other files)",
        "criticName, publicatioName": "Critic and publication (the column name's typo is original)",
        "isTopCritic": "Rotten Tomatoes 'top critic' status",
        "rt_added_date": "Date the review was added to Rotten Tomatoes, not its publication date",
        "reviewState": "fresh or rotten",
        "originalScore_recovered": "The critic's own rating where given (e.g. 4/5), reconstructed",
        "originalScore_was_date_artifact": "True where the rating had been mangled into a date",
        "reviewUrl": "Link to the original article",
        "n_sentences, n_tuples, …": "Per-review counts; mean_polarity_score is the review's average. "
        "This site recomputes them from opinions.csv so the filters apply",
    },
    "sentences.csv: one row per sentence": {
        "review_id, sentence_id": "Review and the sentence's position in it (from 0)",
        "char_start, char_end": "Character offsets of the sentence in the review text",
        "text": "The sentence",
        "has_opinion": "True when at least one tuple came from this sentence",
    },
}
for title, columns in DICTIONARY.items():
    with st.expander(title):
        st.table(pd.DataFrame(columns.items(), columns=["column", "meaning"]).set_index("column"))

# --- Limitations ------------------------------------------------------------
st.header("Limitations")
st.markdown(
    """
- **The sample is overwhelmingly positive.** 102 of 107 reviews are fresh, so about 92% of
  opinions are positive. Statements about "the sentiment of *Parasite*" are really statements about
  this Rotten Tomatoes sample.
- **The tuples are precise, not exhaustive.** Re-annotation agreed on polarity 98% of the time but
  picked the same opinions only 77% of the time. Absence of a tuple does not mean absence of an
  opinion.
- **Intensity is not symmetric yet.** Negative tuples were audited for intensity; positive ones were
  not. Compare positive against negative, not very positive against very negative.
- **Dates are Rotten Tomatoes dates.** `rt_added_date` is when the review was added to Rotten
  Tomatoes, and some 2019 reviews carry 2022 dates. Publication dates are not in the data.
- **Star ratings are reconstructed** for some reviews (see `originalScore_was_date_artifact`), and
  only 64 reviews have one.
- **Small categories are noisy.** Seven categories hold under 2% of tuples each; those with fewer
  than 10 are greyed out and never ranked on this site.
- **Some opinions may be about another film.** 143 tuples are low confidence, 101 of them because
  the critic may have been describing a film they compared *Parasite* to. Use the confidence filter
  in the sidebar to exclude them.
- **Only reviews whose full text could be fetched are included**, which favours publications
  without paywalls or bot-blocking.
"""
)

# --- Downloads --------------------------------------------------------------
st.header("Download the data")
st.markdown(
    "The full data files as stored, unfiltered. `dashboard.csv` has each tuple already joined to "
    "its sentence and review, which is the easiest place to start. The review text is "
    "copyrighted: use it for class work only."
)
cols = st.columns(4)
for col, name in zip(cols, ["dashboard.csv", "opinions.csv", "reviews.csv", "sentences.csv"]):
    col.download_button(
        name, raw_csv_bytes(name), file_name=name, mime="text/csv", width="stretch"
    )
