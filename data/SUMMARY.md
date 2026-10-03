# Parasite (2019) ABSA corpus — project handoff for a Streamlit dashboard

Context document for whoever builds the dashboard. It covers where the data came from, how it was
annotated, what each column means, and the caveats that change how a chart should be read.

---

## 1. What the dataset is

Aspect-Based Sentiment Analysis (ABSA) annotations of English-language **critic** reviews of
*Parasite* (기생충, 2019, dir. Bong Joon-ho), for a methodology paper. Each row is one opinion
tuple: a reviewer's evaluation of one aspect of the film, in SemEval ENTITY#ATTRIBUTE form.

**2045 tuples** from **107 reviews** (96 publications, 22 top critics, 102 fresh / 5 rotten),
covering **4026 sentences**, of which 1254 produced at least one tuple.

## 2. How the corpus was built

1. Source: a Rotten Tomatoes critic-review export (`parasite_2019_revs.xlsx`), 954 rows = 477
   unique reviews after deduplicating on `reviewId`.
2. RT supplies only a one-line snippet, so the full text was fetched from each review's own URL
   (requests + trafilatura, disk-cached per review). The 344 candidate English reviews sit on 323
   distinct hostnames, each needing its own network approval; measured yield was ~0.39 usable
   reviews per hostname granted. **108 reviews** were recovered this way; the rest are
   403 bot-blocks, paywalls and dead links. No browser User-Agent spoofing was used.
3. Content guards: an extraction is rejected unless it mentions /parasite|bong joon/i and is over
   400 characters. This caught two poisoned pages — a review of the *different* 1982 film called
   *Parasite*, and a site that had lapsed into gambling spam.
4. One recovered review (2843303) is excluded from the annotated set: its scraped text is a Robert
   Eggers quotation about Cinemascope, not a review of this film.

## 3. The annotation schema

`parasite_absa_schema_v1.1.json` is the documentation of record (23 ENTITY#ATTRIBUTE categories,
definitions, include/exclude rules, confusion rules, named-entity roster, prior-work mapping to
Zhuang 2006 / Thet 2010 / Parkhe & Biswas 2014). `digest_v3.md` is the derived annotator-facing
codebook actually used at annotation time. **22 categories appear in the data.**

Amendments made during piloting, all deliberate:
- **CHARACTER#DESIGN retired** — under explicit-sentiment rules, reviewers discuss how characters
  are written without a compact sentiment term, so it collapsed to ~1 tuple per 5 reviews.
  Written-role judgements now fold into ACTING#GENERAL (target = character name for the writing,
  actor name for the performance).
- **No OTHER, no gap list** — if nothing fits exactly, the closest category is used.
- **Emotion/Plutchik annotation dropped** — the 8 basic emotions rarely fit critics judging craft,
  leaving ~85% of rows at "none".
- **Neutral polarity not used** — film reviews are rarely neutral and this is opinion mining.

## 4. How the annotation was produced

One LLM pass per review (whole review in context, sentences numbered), then a **deterministic
validator** that is the real guarantor of quality. Annotation policy:

- A tuple needs an **explicit sentiment term**: verbatim, contiguous, 1-6 words. It may be a
  situational description whose sign is obvious ("nothing much here"), per the SentEMO guidelines.
- The **aspect may be implicit** (pronoun or "there is" subject), flagged `aspect_explicit=false`.
  27% of rows are implicit. An aspect term is always copied text or NULL — never a name the
  annotator inferred.
- Aspect and sentiment spans must be **disjoint** and **inside the same sentence**.
- **One row per distinct sentiment term**: "an enjoyable, elegant, scabrous movie" is three rows.
  Concessive pairs ("flawed, yet deeply human") are not split — opposite polarity on either side.
- **very_positive/very_negative require a reinforcing token** in the sentiment term — an
  intensifier, a superlative, or an extreme word — checked against an editable lexicon
  (`reinforcers.json`) with stem-tolerant matching. The validator enforces this on every row.
- Skipped: plot summary, boilerplate, other people's opinions, and anything judging another film.

Every stored span is re-read as `full_text[start:end]` and must match exactly: **0 offset
mismatches** across the corpus. All label values are in the allowed sets.

**Self-consistency** (10 reviews re-annotated independently): sentence-selection agreement
0.817 (Jaccard), 77% of (sentence, category) keys matched, and **98% exact polarity
agreement** on matched keys. The instability is in which sentences get picked, not in how they are
scored — relevant if the dashboard implies the tuple set is exhaustive. It is not; it is precise.

A full audit of all 168 negative tuples corrected 25 rows: 20 intensity upgrades and 5 sign flips
where a reinforcing word was used approvingly ("a searing social commentary"). See
`negatives_audit.csv`.

## 5. The file the dashboard needs

**`dashboard.csv` — 2045 rows, one per tuple, pre-joined. No merges needed.**

| column | meaning |
|---|---|
| `review_id`, `sentence_id` | keys back to `sentences.csv` / `reviews.csv` |
| `aspect_term` | the critic's exact wording, or empty when the aspect is implicit |
| `aspect_explicit` | false = no noun phrase named the aspect |
| `aspect_term_norm` | **use this for counting and charts** (see caveats) |
| `category` | one of 22 ENTITY#ATTRIBUTE ids |
| `target` | roster person (actor, character, crew) or NULL |
| `opinion_term` | the verbatim sentiment span |
| `polarity`, `polarity_score` | very_positive +2, positive +1, negative −1, very_negative −2 |
| `mixed_within_aspect` | same aspect judged both ways in one sentence |
| `confidence`, `note` | annotator confidence; `note` records every validator intervention |
| `sentence` | the full sentence the tuple came from |
| `criticName`, `publicatioName`, `isTopCritic`, `reviewState`, `rt_added_date`, `originalScore_recovered`, `reviewUrl` | review metadata |

Supporting files: `sentences.csv` (all 4026 sentences, including those with no tuple — needed only
for coverage views), `opinions.csv` (same tuples **with character offsets**, for span highlighting
or training a span tagger), `reviews.csv` (per-review metadata and counts), `negatives_audit.csv`
and `validation_log.csv` (QA records, not dashboard inputs).

## 6. Caveats that change how a chart should be read

1. **Count `aspect_term_norm`, not `aspect_term`.** Critics call the film "Parasite", "film",
   "movie", "the film", "picture" — one concept split across five bars. The normalised column maps
   them all to "the film" and normalises person names to the roster spelling. Implicit aspects
   group as "(implicit)" (555 rows); show or filter them deliberately, don't let them hide.
2. **`rt_added_date` is when Rotten Tomatoes ingested the review, not when it was published.**
   Several 2019–2020 reviews carry 2022 dates. Label any timeline "added to Rotten Tomatoes", or
   leave the time axis out. Publication dates are not in this corpus.
3. **`originalScore_recovered` is reconstructed.** The source export had parsed fractions as dates
   ("2026-05-04" meant 4/5). `originalScore_was_date_artifact` in `reviews.csv` marks repaired
   rows. Treat as provisional; don't stratify on it without checking.
4. **The corpus is 92% positive-side**, because it is 102 fresh reviews against 5 rotten.
   Any "sentiment of Parasite" headline is really a statement about Rotten Tomatoes' sample.
   Mean polarity does separate the verdicts: fresh +1.06 (n=1961) vs rotten -0.39 (n=84).
5. **Intensity is not symmetric yet.** Negatives were audited and upgraded; positives were not.
   348 of 1,534 positive rows contain a reinforcing token and would become very_positive under the
   same rule. Until that call is made, compare counts of positive-vs-negative, not of
   very_positive-vs-very_negative.
6. **Seven categories fall below the schema's 2% merge floor** — STORY#PACING, SCREENPLAY#DIALOGUE,
   MUSIC, CULTURAL_CONTEXT, MOVIE#REWATCHABILITY, EDITING, and TRANSLATION#SUBTITLES (1 tuple).
   Small-n categories will look noisy; consider a minimum-count filter on category charts.
7. **143 rows are `confidence = low`**, including 101 flagged as possible comparison-film
   attribution (a pronoun subject in a passage discussing Snowpiercer, Shoplifters, etc.). Offer a
   toggle to exclude low-confidence rows rather than silently including them.

## 7. Current distribution (for sanity-checking your charts)

| category | n | share % |
|---|---|---|
| MOVIE#GENERAL | 421 | 20.6 |
| ACTING#GENERAL | 216 | 10.6 |
| THEME#SOCIAL_COMMENTARY | 204 | 10.0 |
| STORY#PLOT | 197 | 9.6 |
| STORY#GENRE | 192 | 9.4 |
| DIRECTION#GENERAL | 161 | 7.9 |
| PRODUCTION_DESIGN#GENERAL | 98 | 4.8 |
| CINEMATOGRAPHY#GENERAL | 81 | 4.0 |
| TONE#TENSION | 71 | 3.5 |
| TONE#HUMOR | 64 | 3.1 |
| SCREENPLAY#STRUCTURE | 62 | 3.0 |
| THEME#SYMBOLISM | 56 | 2.7 |
| RECEPTION#AWARDS | 51 | 2.5 |
| STORY#ENDING | 44 | 2.2 |
| STORY#TWIST | 41 | 2.0 |
| STORY#PACING | 29 | 1.4 |
| SCREENPLAY#DIALOGUE | 16 | 0.8 |
| MUSIC#GENERAL | 15 | 0.7 |
| CULTURAL_CONTEXT#GENERAL | 11 | 0.5 |
| MOVIE#REWATCHABILITY | 8 | 0.4 |
| EDITING#GENERAL | 6 | 0.3 |
| TRANSLATION#SUBTITLES | 1 | 0.0 |

Polarity: very_positive 348, positive 1534, negative 142, very_negative 21.

| top normalised aspect | n |
|---|---|
| (implicit) | 555 |
| the film | 368 |
| Bong Joon-ho | 120 |
| Hong Kyung-pyo | 48 |
| story | 32 |
| Lee Ha-jun | 23 |
| performances | 21 |
| Song Kang-ho | 19 |
| characters | 18 |
| cast | 16 |

## 8. Views worth building

- Category × polarity heatmap, with a low-confidence toggle and a minimum-count filter.
- Per-review drill-down: pick a review, see its tuples in sentence order with the sentiment span
  highlighted (use `opinions.csv` offsets for highlighting).
- Aspect leaderboard on `aspect_term_norm`, split by polarity.
- Publication / top-critic comparison on mean `polarity_score`, with n shown per group — several
  publications contribute a single review.
- Target view: the roster people (Bong Joon-ho, Song Kang-ho, Hong Kyung-pyo, Lee Ha-jun…) and how
  critics score them.
- Coverage panel from `sentences.csv`: share of sentences carrying an opinion, per review.
