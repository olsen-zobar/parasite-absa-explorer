"""Keyword in context: find a word in the sentences and cut the text around it."""

import re

import pandas as pd


def keyword_pattern(query: str, whole_word: bool = True) -> re.Pattern | None:
    """A case-insensitive regex for the query, treated as plain text (not a regex)."""
    query = query.strip()
    if not query:
        return None
    body = re.escape(query)
    if whole_word:
        body = rf"(?<!\w){body}(?!\w)"
    return re.compile(body, re.IGNORECASE)


def concordance(sentences: pd.DataFrame, pattern: re.Pattern, width: int = 60) -> pd.DataFrame:
    """One row per match (a sentence can match more than once): the left context, the
    keyword as written, the right context, and the match position within the sentence."""
    rows = []
    for row in sentences.itertuples(index=False):
        for m in pattern.finditer(row.text):
            left = row.text[max(0, m.start() - width) : m.start()]
            right = row.text[m.end() : m.end() + width]
            rows.append(
                {
                    "review_id": row.review_id,
                    "sentence_id": row.sentence_id,
                    "left": ("…" if m.start() > width else "") + left,
                    "keyword": m.group(0),
                    "right": right + ("…" if m.end() + width < len(row.text) else ""),
                    "start": m.start(),
                    "end": m.end(),
                }
            )
    columns = ["review_id", "sentence_id", "left", "keyword", "right", "start", "end"]
    return pd.DataFrame(rows, columns=columns)
