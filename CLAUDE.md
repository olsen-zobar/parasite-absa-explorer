# Parasite ABSA Explorer

Streamlit site for a *Coding for Film Studies* class (Columbia University). It presents an
aspect-based sentiment analysis (ABSA) of critic reviews of *Parasite* (2019, dir. Bong Joon-ho).
Audience: film studies students, not engineers. Clarity over cleverness.

## Data (read `data/SUMMARY.md` first; it lists every file and column)
- `data/reviews.csv`: one row per review (metadata, star rating, per-review counts).
- `data/sentences.csv`: one row per sentence (`review_id`, `sentence_id`, offsets, `text`, `has_opinion`).
- `data/opinions.csv`: one row per opinion tuple (aspect term, normalised term, category,
  target, opinion term, polarity, polarity_score, confidence).
- `data/parasite_absa_schema_v1.1.json`: category definitions (23 ENTITY#ATTRIBUTE categories).
- Polarity: very_negative (-2), negative (-1), positive (+1), very_positive (+2). No neutral.
- 108 reviews; exclude the one with no opinion tuples (not a Parasite review) from all stats.
- `creationDate` = date added to Rotten Tomatoes, not publication date. Label it that way.
- Categories with fewer than 10 tuples: show the count and grey them out; never rank them.

## Token rules (important)
- NEVER print or read whole data files. `sentences.csv` alone is ~160k tokens.
  Use pandas: `.shape`, `.head(3)`, `.columns`, `.value_counts()`.
- Test pages with `streamlit.testing.v1.AppTest` (tests/), not screenshots or browsers.
- Keep replies short: what changed, what to check. No long recaps.
- Commit after each working page.

## Stack and structure
- Python 3.11, Streamlit (multipage via `st.navigation`), pandas, Plotly.
- `app.py`: password gate + navigation. `pages/`: one file per page.
  `lib/`: `data.py` (cached loaders, `@st.cache_data`), `charts.py`, `filters.py`, `auth.py`.
- Shared sidebar filters (fresh/rotten, top critic, min confidence) via `st.session_state`.

## Access (the full review text is copyrighted)
- Password gate on every page via `lib/auth.py`, using `st.secrets["password"]` and `hmac.compare_digest`.
- Never commit `.streamlit/secrets.toml` (it is in .gitignore). The repo stays private.
- Every review page links to the original article.

## Design
- Polarity colours (colour-blind safe, diverging): very_negative #B2182B, negative #EF8A62,
  positive #67A9CF, very_positive #2166AC.
- Every chart: a one-sentence takeaway above it and a "Show the code" expander below it
  with the few lines of pandas that produced it (this is a coding class).
- Every number links back to its evidence: let students open the sentences behind a chart.
- Gauges: Plotly `go.Indicator` (gauge), range -2..+2, threshold marker at the corpus mean.

## Pages (build in this order)
1. Home: what ABSA is, one worked example sentence -> tuple, headline stats, pipeline figure
   (`assets/pipeline.png`), credits, data source (Rotten Tomatoes + original publications).
2. Review Explorer: pick a review; header card; gauges (mean polarity, % negative);
   review arc (polarity by sentence order); text with aspect/opinion terms highlighted.
3. Aspect Explorer: filters; diverging polarity bars per category; top terms; evidence sentences.
4. Methods & Data: schema table, pipeline, data dictionary, limitations, CSV downloads.
5. Cast & Crew: praise by person (target); photos from `assets/people/` with credits.
6. Stars vs. Words: star rating vs mean polarity (64 reviews have ratings); label outliers.
7. Keyword in context: search a word, list matching sentences with their labels.
8. Later: Annotator Challenge, reception timeline.
