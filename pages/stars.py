"""Page 6: Stars vs. Words. Does a critic's star rating agree with the sentiment of their prose?"""

import plotly.graph_objects as go
import streamlit as st

from lib.charts import POLARITY_LABELS, highlight_sentence, show_code, takeaway
from lib.data import load_opinions, load_reviews
from lib.filters import apply_filters, is_filtered
from lib.stars import LETTERS, outliers, stars_vs_words

FRESH, ROTTEN = "#2166AC", "#B2182B"
N_OUTLIERS = 5
MIN_RATED = 10  # fewer points than this and a trend line means nothing

opinions = load_opinions()
shown = apply_filters(opinions)
rated = stars_vs_words(apply_filters(load_reviews()), shown)
flagged = outliers(rated, N_OUTLIERS)

st.title("Stars vs. Words")
st.markdown(
    "Most critics sum up a review with a grade: 4/5, 9/10, a B+. The annotation gives us a "
    "second, independent measure: the average polarity of every opinion the critic wrote. "
    "When the two disagree, the prose is saying something the stars are not."
)
n_total = (load_reviews()["originalScore_recovered"].notna()).sum()
st.caption(
    f"Only {n_total} of the {len(load_reviews())} reviews carry a rating. The ratings were "
    "recovered from a damaged export (fractions had been turned into dates), so treat them "
    "as provisional."
)

if len(rated) < MIN_RATED:
    st.warning(
        f"Only {len(rated)} rated reviews match the sidebar filters. That is too few to draw a "
        f"trend (this page needs at least {MIN_RATED}), so widen the filters."
    )
    st.stop()

# --- Scatter ----------------------------------------------------------------
st.header("Rating against mean polarity")
r = rated["rating"].corr(rated["mean_polarity"])
strength = "strongly" if abs(r) >= 0.6 else "moderately" if abs(r) >= 0.3 else "only weakly"
takeaway(
    f"Stars and words agree {strength} (Pearson r = {r:.2f} across {len(rated)} rated "
    f"reviews); the labelled points are the {len(flagged)} reviews that stray furthest from "
    "the trend."
)
if is_filtered():
    st.caption(
        f"Sidebar filters are on: {len(rated)} rated reviews, mean polarity from "
        f"{len(shown)} of {len(opinions)} tuples."
    )

fig = go.Figure()
for state, colour in [("fresh", FRESH), ("rotten", ROTTEN)]:
    part = rated[rated["reviewState"] == state]
    fig.add_trace(
        go.Scatter(
            x=part["rating"],
            y=part["mean_polarity"],
            mode="markers",
            name=state,
            marker=dict(color=colour, size=10, opacity=0.75, line=dict(width=1, color="white")),
            customdata=part[["criticName", "publicatioName", "originalScore_recovered", "tuples"]],
            hovertemplate="<b>%{customdata[0]}</b>, %{customdata[1]}<br>"
            "rating %{customdata[2]}<br>mean polarity %{y:+.2f} "
            "(%{customdata[3]} opinions)<extra></extra>",
        )
    )
line = rated.sort_values("rating")
fig.add_trace(
    go.Scatter(
        x=line["rating"],
        y=line["predicted"],
        mode="lines",
        name="trend",
        line=dict(color="#52606D", dash="dash"),
        hoverinfo="skip",
    )
)
for row in flagged.itertuples():
    fig.add_annotation(
        x=row.rating,
        y=row.mean_polarity,
        text=f"{row.criticName} ({row.originalScore_recovered})",
        showarrow=True,
        arrowhead=0,
        ax=0,
        ay=-28 if row.residual > 0 else 28,
        font=dict(size=11),
    )
fig.update_layout(
    height=480,
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis=dict(title="star rating, rescaled (0 = worst, 1 = best)", range=[-0.05, 1.08]),
    yaxis=dict(title="mean polarity of the review's opinions", range=[-2.1, 2.1], zeroline=True),
    legend=dict(orientation="h", y=1.02, x=0),
)
st.plotly_chart(fig, width="stretch")
show_code(
    """
rated = reviews.dropna(subset=["originalScore_recovered"]).copy()
rated["rating"] = rated["originalScore_recovered"].map(rating_to_unit)  # "4/5" -> 0.8
words = opinions.groupby("review_id")["polarity_score"].mean().rename("mean_polarity")
rated = rated.merge(words, on="review_id")

r = rated["rating"].corr(rated["mean_polarity"])           # Pearson r
slope, intercept = np.polyfit(rated["rating"], rated["mean_polarity"], 1)
rated["residual"] = rated["mean_polarity"] - (intercept + slope * rated["rating"])
outliers = rated.loc[rated["residual"].abs().nlargest(5).index]
"""
)
with st.expander("How ratings were put on one scale"):
    st.markdown(
        "- **Fractions** are divided out: 4/5 = 0.8, 9/10 = 0.9, 3.5/4 = 0.875.\n"
        f"- **Letter grades** sit on {len(LETTERS)} equal steps from F (0) to A+ (1): "
        "A = 0.92, A− = 0.83, B = 0.67.\n"
        "- One rating is a bare **5**, read as 5 out of 5.\n"
        "- Mean polarity runs from −2 (every opinion very negative) to +2."
    )

# --- Outliers ---------------------------------------------------------------
st.header("Where stars and words part ways")
st.markdown(
    "A point **below** the trend is a review whose prose is harsher than its grade; a point "
    "**above** it is warmer than its grade. Reviews with few opinions swing easily, so check "
    "the opinion count before reading much into one."
)
table = flagged[
    [
        "criticName",
        "publicatioName",
        "originalScore_recovered",
        "reviewState",
        "mean_polarity",
        "residual",
        "tuples",
    ]
]
st.dataframe(
    table.rename(
        columns={
            "criticName": "critic",
            "publicatioName": "publication",
            "originalScore_recovered": "rating",
            "reviewState": "verdict",
            "mean_polarity": "mean polarity",
            "residual": "words minus trend",
            "tuples": "opinions",
        }
    ).style.format({"mean polarity": "{:+.2f}", "words minus trend": "{:+.2f}"}),
    hide_index=True,
    width="stretch",
)

# --- Evidence ---------------------------------------------------------------
st.header("Read the opinions behind a point")
order = rated.reindex(rated["residual"].abs().sort_values(ascending=False).index)
labels = {
    row.review_id: f"{row.criticName}, {row.publicatioName} ({row.originalScore_recovered}, "
    f"mean {row.mean_polarity:+.2f})"
    for row in order.itertuples()
}
pick = st.selectbox(
    "Review (sorted from biggest to smallest disagreement)",
    list(labels),
    format_func=labels.get,
    key="stars_review",
)
review = rated[rated["review_id"] == pick].iloc[0]
st.markdown(f"[Read the original review]({review['reviewUrl']}) · *{review['publicatioName']}*")
evidence = shown[shown["review_id"] == pick].sort_values(["sentence_id", "opinion_start"])
for sentence_id, group in evidence.groupby("sentence_id", sort=True):
    offset = group["sentence_start"].iloc[0]
    spans = list(
        dict.fromkeys(
            (int(o.opinion_start) - offset, int(o.opinion_end) - offset, str(o.polarity))
            for o in group.itertuples()
        )
    )
    st.markdown(
        f"<p>{highlight_sentence(group['sentence'].iloc[0], spans)}</p>", unsafe_allow_html=True
    )
st.caption(
    "Opinion terms are coloured by polarity: "
    + ", ".join(POLARITY_LABELS.values())
    + ". Sentences without an opinion are left out."
)
