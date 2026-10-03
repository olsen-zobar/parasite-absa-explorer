"""Parasite ABSA annotation pipeline (schema v1.1, 5-level polarity override).

Reusable across the pilot (stage 1) and the full run (stage 2).
One review per call; every span is verified against the source text before it is kept.
"""
import json, os, re, unicodedata

PROMPT_VERSION = "parasite-absa-prompt-v1"

CATEGORIES = None  # filled by load_schema
POLARITIES = ["very_positive", "positive", "neutral", "negative", "very_negative"]
POLARITY_SCORE = {"very_positive": 2, "positive": 1, "neutral": 0, "negative": -1, "very_negative": -2}
EMOTIONS = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation", "none"]
INTENSITIES = ["low", "medium", "high"]
PLUTCHIK = {
    "joy": {"low": "serenity", "medium": "joy", "high": "ecstasy"},
    "trust": {"low": "acceptance", "medium": "trust", "high": "admiration"},
    "fear": {"low": "apprehension", "medium": "fear", "high": "terror"},
    "surprise": {"low": "distraction", "medium": "surprise", "high": "amazement"},
    "sadness": {"low": "pensiveness", "medium": "sadness", "high": "grief"},
    "disgust": {"low": "boredom", "medium": "disgust", "high": "loathing"},
    "anger": {"low": "annoyance", "medium": "anger", "high": "rage"},
    "anticipation": {"low": "interest", "medium": "anticipation", "high": "vigilance"},
}
EMOTION_ALIASES = {}  # wheel-named forms -> (base emotion, intensity)
for _b, _d in PLUTCHIK.items():
    for _i, _n in _d.items():
        EMOTION_ALIASES[_n] = (_b, _i)

DYADS = ["love", "submission", "awe", "disapproval", "remorse", "contempt", "aggressiveness", "optimism"]
CONFIDENCES = ["low", "medium", "high"]


def load_schema(path):
    global CATEGORIES
    with open(path, encoding="utf-8") as fh:
        schema = json.load(fh)
    CATEGORIES = [c["id"] for c in schema["categories"]]
    return schema


# ---------------------------------------------------------------- sentences
def split_sentences(text):
    """pysbd segmentation with char offsets, further split on hard newlines."""
    import pysbd
    seg = pysbd.Segmenter(language="en", clean=False, char_span=True)
    out = []
    for sp in seg.segment(text):
        s, e = sp.start, sp.end
        # trim whitespace into exact offsets
        raw = text[s:e]
        lead = len(raw) - len(raw.lstrip())
        trail = len(raw) - len(raw.rstrip())
        s, e = s + lead, e - trail
        if e <= s:
            continue
        # pysbd can keep a paragraph break inside one span; split on newlines
        chunk = text[s:e]
        pos = 0
        for piece in re.split(r"\n+", chunk):
            idx = chunk.index(piece, pos)
            pos = idx + len(piece)
            ps, pe = s + idx, s + idx + len(piece)
            ps += len(piece) - len(piece.lstrip())
            pe -= len(piece) - len(piece.rstrip())
            if pe > ps:
                out.append((ps, pe))
    return out


# ---------------------------------------------------------------- prompt
def schema_digest(schema):
    lines = []
    for c in schema["categories"]:
        lines.append(f"### {c['id']}  ({c['name']})")
        lines.append(f"DEF: {c['definition']}")
        lines.append("INCLUDE: " + " | ".join(c["include"]))
        lines.append("EXCLUDE: " + " | ".join(c["exclude"]))
        lines.append("CONFUSIONS: " + " | ".join(c["confusions"]))
        lines.append("")
    tr = schema["target_rules"]
    lines.append("## TARGET RULES")
    lines.append(tr["principle"])
    lines.append(tr["target_value"])
    for k, v in tr["actor_vs_character"].items():
        lines.append(f"- {k}: {v}")
    lines.append("## ROSTER (use these canonical spellings in `target`)")
    lines.append(json.dumps(tr["roster"], ensure_ascii=False))
    lines.append("## HARD CASES")
    for k, v in schema["hard_cases"].items():
        rule = v.get("rule", "")
        lines.append(f"- {k}: {rule}")
    return "\n".join(lines)


SYSTEM = """You are an expert annotator for an Aspect-Based Sentiment Analysis research corpus of
English-language critic reviews of the film Parasite (2019, Bong Joon-ho).
Work ONLY from the review text given to you. Never use outside knowledge about the film to infer
an opinion the reviewer did not express. Accuracy and consistency matter more than coverage.
You output JSON only."""


def build_user_prompt(digest, meta, text, sents):
    numbered = "\n".join(f"[S{i}] {text[s:e]}" for i, (s, e) in enumerate(sents))
    return f"""# TASK
Annotate one review. Emit one JSON object per OPINION TUPLE: an evaluation of one aspect of the
film by the REVIEWER. Read the whole review first; irony, pronouns and implicit aspects depend on
context.

# WHAT TO ANNOTATE / SKIP
- Annotate only the REVIEWER's own stance. Skip opinions quoted from other people, audience
  reactions, and characters' feelings inside plot summary.
- Skip boilerplate: credits, cast lists, runtimes, release dates, cinema listings, subscription or
  newsletter text, author bios, "Advertisement", "Share this".
- Plot/premise recounting WITHOUT evaluation IS annotated: relevant STORY#* (or CHARACTER#DESIGN)
  category, polarity neutral, is_summary=true.
- Comparisons to other films (Snowpiercer, Memories of Murder, Us, The Housemaid, Shoplifters,
  Okja, The Host...) are annotated only for what they say about PARASITE. Never make a comparison
  film the target, never record sentiment about it.
- A sentence may yield zero, one or several tuples. Decompose mixed sentiment: different aspects ->
  separate tuples; the SAME aspect praised and criticised -> one tuple per opinion term, each with
  its own polarity, and mixed_within_aspect=true on both.
- The category inventory is FIXED. If a genuine evaluation of the film fits NO category, use
  category "OTHER" and also list it under "gaps".

# FIELDS PER TUPLE
- sid: integer sentence id, e.g. 7 for [S7].
- aspect_term: the EXACT verbatim substring of that sentence naming what is evaluated
  ("the score", "Song Kang-ho", "the staircases"). Must be copyable character-for-character from
  the sentence. If the aspect is implicit (no naming phrase), use null.
- category: one id from the list below (ENTITY#ATTRIBUTE), or "OTHER".
- target: canonical roster name for a named person; else the aspect phrase actually used
  ("the score", "the houses", "the cast", "Oscars"); else "NULL".
- opinion_term: the EXACT verbatim substring of that sentence carrying the evaluation
  ("devastating", "never slackens", "laid on with a trowel"). CONTIGUOUS verbatim substring -
  never join separated words with "...", never paraphrase. If the evaluation is spread over a
  clause, quote the single most evaluative contiguous stretch. null if the evaluation is implied
  with no evaluative words.
- polarity: very_positive | positive | neutral | negative | very_negative
  very_positive = unreserved praise or explicit intensity (superlatives, "masterpiece",
  "flawless", "best of the year", strong intensifiers).
  positive = clearly favourable but moderate or hedged.
  neutral = aspect discussed in an evaluative context but no stance expressed.
  negative = clearly unfavourable but moderate or hedged.
  very_negative = strong dismissal or explicit intensity ("unbearable", "a total failure").
  Critics write in an elevated register: reserve very_* for EXPLICIT intensity markers, not merely
  eloquent phrasing. Hedges ("rather", "somewhat", "a little") lower intensity one step. Negation
  flips polarity. Sarcasm is labelled by intended meaning, and only with textual evidence
  (scare quotes, overt exaggeration, a contradicting adjacent clause).
- emotion: the emotion the REVIEWER expresses or reports experiencing toward this aspect,
  independent of polarity. Use EXACTLY one of these 9 strings and no other word (not
  "admiration", not "disappointment"): joy | trust | fear | surprise | sadness | disgust |
  anger | anticipation | none. Use "none" when no emotion is expressed; do not force one.
- emotion_intensity: low | medium | high (omit meaning: use "low" only when hedged). If
  emotion is "none", use null.
- dyad: optional, only when two basic emotions clearly co-occur: love (joy+trust),
  submission (trust+fear), awe (fear+surprise), disapproval (surprise+sadness),
  remorse (sadness+disgust), contempt (disgust+anger), aggressiveness (anger+anticipation),
  optimism (anticipation+joy). Else null.
- is_summary: true only for non-evaluative plot/premise description.
- mixed_within_aspect: true only in the same-aspect-both-ways case described above.
- confidence: low | medium | high. note: short free text; REQUIRED when confidence is "low".

# OUTPUT FORMAT (JSON only, no prose, no code fence)
{{"tuples": [{{"sid": 0, "aspect_term": "...", "category": "...", "target": "...",
"opinion_term": "...", "polarity": "...", "emotion": "...", "emotion_intensity": "...",
"dyad": null, "is_summary": false, "mixed_within_aspect": false, "confidence": "high",
"note": ""}}],
"gaps": [{{"sid": 0, "span": "...", "why": "..."}}]}}

# SCHEMA (categories, definitions, include/exclude, confusion rules)
{digest}

# REVIEW METADATA (context only; do not annotate it)
publication: {meta['publicatioName']} | critic: {meta['criticName']} | RT state: {meta['reviewState']}

# REVIEW, SENTENCE BY SENTENCE
{numbered}

Return the JSON object now.
"""


# ---------------------------------------------------------------- validation
def _find_span(text, span, s, e):
    """Locate span inside sentence [s,e); fall back to whole text. Returns (start,end) or None.

    Non-contiguous spans joined with an ellipsis are repaired to their longest
    contiguous segment, so the stored term is always verbatim source text.
    """
    if span is None:
        return None
    span = span.strip()
    if not span:
        return None
    if re.search(r"\.\.\.|\u2026", span):
        segs = sorted((x.strip() for x in re.split(r"\.\.\.|\u2026", span)), key=len, reverse=True)
        for seg in segs:
            if len(seg) >= 3:
                hit = _find_span(text, seg, s, e)
                if hit:
                    return hit
        return None
    sent = text[s:e]
    # pairs are annotated within one sentence: never search outside [s, e)
    for hay, off in ((sent, s),):
        i = hay.find(span)
        if i >= 0:
            return (i + off, i + off + len(span))
    # tolerate whitespace / curly-quote variation
    pat = re.escape(span)
    pat = re.sub(r"\\\s+", r"\\s+", pat)
    pat = pat.replace("'", "['\u2019\u2018]").replace('"', '["\u201c\u201d]')
    for hay, off in ((sent, s),):
        m = re.search(pat, hay)
        if m:
            return (m.start() + off, m.end() + off)
    # case-insensitive, then determiner-stripped, then head-noun - all inside the sentence only,
    # so the stored span is still a verbatim slice of the source text
    m = re.search(re.escape(span), sent, re.I)
    if m:
        return (m.start() + s, m.end() + s)
    stripped = re.sub(r"^(?:the|a|an|its|his|her|their|this|that|these|those)\s+", "", span, flags=re.I).strip()
    if stripped and stripped != span:
        m = re.search(re.escape(stripped), sent, re.I)
        if m:
            return (m.start() + s, m.end() + s)
        span = stripped
    toks = span.split()
    if len(toks) > 1:
        for n in (2, 1):
            if len(toks) > n:
                tail = " ".join(toks[-n:])
                m = re.search(r"\b" + re.escape(tail), sent, re.I)
                if m:
                    return (m.start() + s, m.end() + s)
    return None


def normalise_target(t, roster_map):
    if t is None:
        return "NULL"
    t = re.sub(r"\s+", " ", str(t)).strip().strip('"\u201c\u201d')
    if not t or t.lower() in {"null", "none", "n/a"}:
        return "NULL"
    key = t.lower().strip(".,;:'\u2019s ")
    return roster_map.get(key, t)


def build_roster_map(schema):
    r = schema["target_rules"]["roster"]
    m = {}
    names = list(r["actor_to_character"].keys()) + list(r["actor_to_character"].values()) \
        + list(r["crew"].keys()) + r["director"] + r["writers"]
    for n in names:
        m[n.lower()] = n
        m[n.lower().replace("-", " ")] = n
        m[unicodedata.normalize("NFKD", n.lower())] = n
    # unambiguous short forms only
    short = {"song": "Song Kang-ho", "bong": "Bong Joon-ho", "bong joon ho": "Bong Joon-ho",
             "joon-ho bong": "Bong Joon-ho", "kang-ho song": "Song Kang-ho",
             "choi": "Choi Woo-shik", "cho": "Cho Yeo-jeong", "jang": "Jang Hye-jin",
             "hong": "Hong Kyung-pyo", "yang": "Yang Jin-mo", "paquet": "Darcy Paquet",
             "jung jae il": "Jung Jae-il", "han": "Han Jin-won"}
    m.update(short)
    return m


def validate_tuples(raw, text, sents, roster_map, review_id):
    """Coerce, validate and offset-verify model output. Returns (rows, problems)."""
    rows, probs = [], []
    for k, t in enumerate(raw):
        try:
            sid = int(t["sid"])
        except Exception:
            probs.append({"review_id": review_id, "i": k, "problem": "bad sid", "detail": str(t)[:200]})
            continue
        if not (0 <= sid < len(sents)):
            probs.append({"review_id": review_id, "i": k, "problem": "sid out of range", "detail": str(sid)})
            continue
        s, e = sents[sid]
        cat = str(t.get("category", "")).strip().upper().replace(" ", "")
        if cat not in CATEGORIES and cat != "OTHER":
            probs.append({"review_id": review_id, "i": k, "problem": "bad category", "detail": cat})
            continue
        pol = str(t.get("polarity", "")).strip().lower()
        if pol not in POLARITIES:
            probs.append({"review_id": review_id, "i": k, "problem": "bad polarity", "detail": pol})
            continue
        emo = str(t.get("emotion") or "none").strip().lower()
        if emo in EMOTION_ALIASES:
            emo, _alias_int = EMOTION_ALIASES[emo]
            if not t.get("emotion_intensity"):
                t = dict(t); t["emotion_intensity"] = _alias_int
        if emo not in EMOTIONS:
            probs.append({"review_id": review_id, "i": k, "problem": "bad emotion", "detail": emo})
            emo = "none"
        inten = t.get("emotion_intensity")
        inten = str(inten).strip().lower() if inten else None
        if emo == "none":
            inten, plut = None, "none"
        else:
            if inten not in INTENSITIES:
                inten = "medium"
                probs.append({"review_id": review_id, "i": k, "problem": "bad emotion_intensity->medium",
                              "detail": str(t.get("emotion_intensity"))[:40]})
            plut = PLUTCHIK[emo][inten]
        dyad = t.get("dyad")
        dyad = str(dyad).strip().lower() if dyad else None
        if dyad not in DYADS:
            dyad = None
        conf = str(t.get("confidence", "medium")).strip().lower()
        if conf not in CONFIDENCES:
            conf = "medium"
        note = re.sub(r"\s+", " ", str(t.get("note") or "")).strip()

        asp_sp = _find_span(text, t.get("aspect_term"), s, e)
        op_sp = _find_span(text, t.get("opinion_term"), s, e)
        if t.get("aspect_term") and asp_sp is None:
            probs.append({"review_id": review_id, "i": k, "problem": "aspect_term not found -> NULL",
                          "detail": str(t.get("aspect_term"))[:120]})
            note = (note + " | unverifiable aspect span dropped").strip(" |")
            conf = "low"
        if t.get("opinion_term") and op_sp is None:
            probs.append({"review_id": review_id, "i": k, "problem": "opinion_term not found -> NULL",
                          "detail": str(t.get("opinion_term"))[:120]})
            note = (note + " | unverifiable opinion span dropped").strip(" |")
            conf = "low"
        if conf == "low" and not note:
            note = "low confidence, no reason given by annotator"
        entity, attribute = (cat.split("#", 1) + [""])[:2] if "#" in cat else (cat, "")
        rows.append({
            "review_id": review_id, "sentence_id": sid,
            "aspect_term": text[asp_sp[0]:asp_sp[1]] if asp_sp else None,
            "aspect_start": asp_sp[0] if asp_sp else None,
            "aspect_end": asp_sp[1] if asp_sp else None,
            "aspect_explicit": bool(asp_sp),
            "category": cat, "entity": entity, "attribute": attribute,
            "target": normalise_target(t.get("target"), roster_map),
            "opinion_term": text[op_sp[0]:op_sp[1]] if op_sp else None,
            "opinion_start": op_sp[0] if op_sp else None,
            "opinion_end": op_sp[1] if op_sp else None,
            "opinion_explicit": bool(op_sp),
            "polarity": pol, "polarity_score": POLARITY_SCORE[pol],
            "emotion": emo, "emotion_intensity": inten, "plutchik_label": plut, "dyad": dyad,
            "is_summary": bool(t.get("is_summary", False)),
            "mixed_within_aspect": bool(t.get("mixed_within_aspect", False)),
            "confidence": conf, "note": note,
        })
    return rows, probs


def extract_json(s):
    s = s.strip()
    s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.M).strip()
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j < 0:
        raise ValueError("no JSON object in model output")
    return json.loads(s[i:j + 1])


# ================================ v2: explicit-only, concise, token-lean ===============
PROMPT_VERSION_V2 = "parasite-absa-prompt-v2-explicit"

REINFORCERS = r"""(?ix)\b(
 very|so|too|utterly|absolutely|completely|totally|entirely|thoroughly|deeply|profoundly|
 incredibly|extraordinarily|exceptionally|remarkably|astonishingly|staggeringly|stunningly|
 breathtakingly|dazzlingly|wildly|insanely|crazy|crazily|super|hugely|immensely|enormously|
 ferociously|savagely|devastatingly|wickedly|gloriously|magnificently|sublimely|perfectly|
 unbearably|appallingly|laughably|hopelessly|woefully|painfully|excruciatingly|
 masterpiece|masterful|masterly|flawless|perfection|triumph|sensational|phenomenal|
 unmissable|essential|unforgettable|never|nothing|no\s+one|pitch-perfect|
 best|greatest|finest|worst|most|least|\w+est
)\b"""

SYSTEM_V2 = SYSTEM


def build_user_prompt_v2(digest, meta, text, sents):
    numbered = "\n".join(f"[S{i}] {text[s:e]}" for i, (s, e) in enumerate(sents))
    return f"""# TASK
Annotate ONE film review for aspect-based sentiment. Emit ONLY clear, EXPLICIT
aspect-sentiment pairs. Read the whole review first, then work sentence by sentence.

# WHAT COUNTS AS A PAIR
A pair needs BOTH, inside the SAME sentence:
  - an EXPLICIT aspect term: a noun phrase naming what is judged ("the score", "Song Kang-ho",
    "the staircases", "Bong's direction");
  - an EXPLICIT sentiment term: the words carrying the judgement ("devastating", "never
    slackens", "laid on with a trowel").
Both must be copied VERBATIM and CONTIGUOUS from that sentence. Never paraphrase, never join
separated words with "...".

# SKIP (emit nothing) WHEN
- the aspect is implicit (pronoun, "there is", or no naming phrase at all) - no NULL aspects;
- the sentiment is implicit, vague, hedged into ambiguity, or merely descriptive;
- the sentence is plot summary, narrative recap, cast/credit listing, runtime, release info,
  or publication boilerplate;
- the wording is long-winded and you would have to quote a whole clause to capture it;
- the opinion belongs to someone else (quoted critic, audience, the director, "some have said");
- the pair is out of scope: it judges another film (Snowpiercer, Memories of Murder, Us,
  Shoplifters, Okja...), a festival, or a person's career rather than this film;
- you are unsure, or two categories are equally defensible.
Skipping is free and correct. Precision beats coverage: a skipped sentence costs nothing,
a doubtful tuple costs the dataset.

# CONCISION (hard requirement)
- aspect term: the smallest substantive noun phrase, 1-4 content words, no articles.
  GOOD: "the score", "Hong Kyung-pyo's camera", "the bunker set".
  BAD: "the film's focus on poverty, desperation and the phenomenon of those in debt".
- sentiment term: 1-6 words, the evaluative core only.
  GOOD: "savagely funny", "never slackens", "a trowel". BAD: a whole clause.

# POLARITY (integer)
  +1 positive   -1 negative   -> the normal case. "great", "wonderful", "terrific",
                                 "horrible", "dull" are +1 / -1.
  +2 / -2 ONLY when the sentiment term itself contains a REINFORCING token: an intensifier
     ("very", "utterly", "crazy good", "super", "savagely"), a superlative ("the best of the
     year", "finest"), or an absolute ("masterpiece", "flawless", "unbearable").
  0 is NOT used in this run. If you would label something neutral, skip it instead.
  Negation flips sign. Hedges ("rather", "somewhat", "a bit") keep the magnitude at 1.
  Sarcasm is labelled by intended meaning, only with textual evidence in the same or the
  adjacent sentence.

# EMOTION (Plutchik; the reviewer's own feeling, independent of polarity)
  one of: joy | trust | fear | surprise | sadness | disgust | anger | anticipation | none
  "none" is the DEFAULT and should be frequent: a judgement with no reported feeling is "none".
  Use disgust only for revulsion or boredom, anger only for irritation or indignation,
  fear/surprise only where the reviewer reports being frightened or taken aback.
  intensity: l | m | h. Use "" for both intensity and dyad when emotion is "none".
  dyad (optional, usually ""): love, submission, awe, disapproval, remorse, contempt,
  aggressiveness, optimism.

# OUTPUT - JSON only, no prose, no code fence. One array per tuple, in this exact order:
[sid, aspect_term, category, target, sentiment_term, polarity, emotion, intensity, dyad,
 confidence, note]
 sid = integer; polarity = -2|-1|1|2; confidence = "h"|"m"|"l"; note = "" unless confidence
 is "l" (then one short clause). target = roster name, or "" if no roster person is named.
Shape: {{"t": [[3, "the score", "MUSIC#GENERAL", "Jung Jae-il", "extraordinary", 1, "trust",
"m", "", "h", ""]], "gaps": [[sid, "span", "why no category fits"]]}}
Use "gaps" only for a clear evaluation of the film that fits NO category in the list.

{digest}

# REVIEW ({meta['publicatioName']}, {meta['criticName']}, RT: {meta['reviewState']})
{numbered}

JSON now.
"""


def _final_target(cat, tgt, roster_map, schema_roster):
    t = normalise_target(tgt, roster_map)
    if t == "NULL" or schema_roster is None:
        return "NULL"
    t = type_target(cat, t, schema_roster)
    return t if known_person(t, schema_roster) else "NULL"


COMPARISON_FILMS = {"shoplifters", "snowpiercer", "memories of murder", "the host", "okja", "mother",
    "burning", "the housemaid", "us", "pain and glory", "dolor y gloria", "roma", "the handmaiden",
    "joker", "1917", "once upon a time in hollywood", "the irishman", "knives out", "barking dogs never bite"}

PRONOUN_ASPECTS = {"this","that","it","they","them","these","those","he","she","there","one","its","his","her"}


def type_target(cat, tgt, schema_roster):
    """ACTING targets must be actors; CHARACTER#DESIGN targets must be characters."""
    a2c = schema_roster["actor_to_character"]
    c2a = {v: k for k, v in a2c.items()}
    if cat == "ACTING#GENERAL" and tgt in c2a:
        return c2a[tgt]
    if cat == "CHARACTER#DESIGN" and tgt in a2c:
        return a2c[tgt]
    return tgt


def known_person(tgt, schema_roster):
    names = set(schema_roster["actor_to_character"]) | set(schema_roster["actor_to_character"].values()) \
        | set(schema_roster["crew"]) | set(schema_roster["director"]) | set(schema_roster["writers"]) \
        | {"Kim family", "Park family", "the cast", "the ensemble", "supporting cast"}
    return tgt in names


def validate_tuples_v2(raw, text, sents, roster_map, review_id, max_aspect_words=6, max_op_words=9,
                       schema_roster=None):
    """Validate v2 positional rows. Enforces explicit spans, no neutral, reinforced +/-2."""
    rows, probs = [], []
    pol_map = {2: "very_positive", 1: "positive", -1: "negative", -2: "very_negative", 0: "neutral"}
    conf_map = {"h": "high", "m": "medium", "l": "low", "high": "high", "medium": "medium", "low": "low"}
    int_map = {"l": "low", "m": "medium", "h": "high", "low": "low", "medium": "medium", "high": "high"}
    for k, t in enumerate(raw):
        if isinstance(t, dict):  # tolerate key-value form
            t = [t.get(x, "") for x in ("sid", "aspect_term", "category", "target", "sentiment_term",
                                        "polarity", "emotion", "intensity", "dyad", "confidence", "note")]
        if not isinstance(t, (list, tuple)) or len(t) < 10:
            probs.append({"review_id": review_id, "i": k, "problem": "malformed row", "detail": str(t)[:160]}); continue
        sid, asp, cat, tgt, opi, pol, emo, inten, dyad, conf = t[:10]
        note = re.sub(r"\s+", " ", str(t[10] if len(t) > 10 else "")).strip()
        try:
            sid = int(sid)
        except Exception:
            probs.append({"review_id": review_id, "i": k, "problem": "bad sid", "detail": str(sid)[:60]}); continue
        if not (0 <= sid < len(sents)):
            probs.append({"review_id": review_id, "i": k, "problem": "sid out of range", "detail": str(sid)}); continue
        s, e = sents[sid]
        cat = str(cat).strip().upper().replace(" ", "")
        if cat not in CATEGORIES and cat != "OTHER":
            probs.append({"review_id": review_id, "i": k, "problem": "bad category", "detail": cat}); continue
        try:
            pol_i = int(pol)
        except Exception:
            probs.append({"review_id": review_id, "i": k, "problem": "bad polarity", "detail": str(pol)[:40]}); continue
        if pol_i == 0:
            probs.append({"review_id": review_id, "i": k, "problem": "neutral dropped (not used in v2)",
                          "detail": str(asp)[:80]}); continue
        if re.sub(r"[^a-z]", "", str(asp).lower()) in PRONOUN_ASPECTS:
            probs.append({"review_id": review_id, "i": k, "problem": "tuple dropped: pronoun aspect (implicit)",
                          "detail": str(asp)[:60]}); continue
        asp_sp, op_sp = _find_span(text, asp, s, e), _find_span(text, opi, s, e)
        if asp_sp is None or op_sp is None:
            probs.append({"review_id": review_id, "i": k,
                          "problem": "tuple dropped: span not verbatim in sentence",
                          "detail": f"aspect={str(asp)[:60]!r} opinion={str(opi)[:60]!r}"}); continue
        asp_txt, op_txt = text[asp_sp[0]:asp_sp[1]], text[op_sp[0]:op_sp[1]]
        # reinforcement gate on +/-2
        if abs(pol_i) == 2 and not re.search(REINFORCERS, op_txt):
            probs.append({"review_id": review_id, "i": k, "problem": "downgraded: no reinforcing token",
                          "detail": op_txt[:60]})
            pol_i = 1 if pol_i > 0 else -1
            note = (note + " | intensity downgraded, no reinforcing token").strip(" |")
        # concision gate (flag, do not alter the span)
        long_flags = []
        if len(asp_txt.split()) > max_aspect_words: long_flags.append("aspect span long")
        if len(op_txt.split()) > max_op_words: long_flags.append("sentiment span long")
        if long_flags:
            probs.append({"review_id": review_id, "i": k, "problem": "; ".join(long_flags),
                          "detail": f"{asp_txt[:50]!r} / {op_txt[:50]!r}"})
            note = (note + " | " + "; ".join(long_flags)).strip(" |")
        emo = str(emo or "none").strip().lower()
        if emo in EMOTION_ALIASES:
            emo, _ai = EMOTION_ALIASES[emo]
            inten = inten or _ai
        if emo not in EMOTIONS:
            probs.append({"review_id": review_id, "i": k, "problem": "bad emotion -> none", "detail": emo}); emo = "none"
        if emo == "none":
            inten, plut = None, "none"
        else:
            inten = int_map.get(str(inten).strip().lower(), "medium")
            plut = PLUTCHIK[emo][inten]
        dyad = str(dyad or "").strip().lower() or None
        if dyad not in DYADS: dyad = None
        conf = conf_map.get(str(conf).strip().lower(), "medium")
        if conf == "low" and not note: note = "low confidence, no reason given by annotator"
        entity, attribute = (cat.split("#", 1) + [""])[:2] if "#" in cat else (cat, "")
        rows.append({"review_id": review_id, "sentence_id": sid,
                     "aspect_term": asp_txt, "aspect_start": asp_sp[0], "aspect_end": asp_sp[1],
                     "category": cat, "entity": entity, "attribute": attribute,
                     "target": _final_target(cat, tgt, roster_map, schema_roster),
                         "opinion_term": op_txt, "opinion_start": op_sp[0], "opinion_end": op_sp[1],
                     "polarity": pol_map[pol_i], "polarity_score": pol_i,
                     "emotion": emo, "emotion_intensity": inten, "plutchik_label": plut, "dyad": dyad,
                     "mixed_within_aspect": False, "confidence": conf, "note": note})
    # mixed_within_aspect derived, not asserted: same sentence+category+target, opposite signs
    import collections
    g = collections.defaultdict(list)
    for i, r in enumerate(rows):
        g[(r["sentence_id"], r["category"], r["target"])].append(i)
    for idxs in g.values():
        signs = {rows[i]["polarity_score"] > 0 for i in idxs}
        if len(signs) > 1:
            for i in idxs: rows[i]["mixed_within_aspect"] = True
    return rows, probs


# ================================ v3: explicit sentiment, optional implicit aspect, no emotion =========
PROMPT_VERSION_V3 = "parasite-absa-prompt-v3"


def load_reinforcers(path="reinforcers.json"):
    L = json.load(open(path, encoding="utf-8"))
    words = L["intensifier_adverbs"] + L["extreme_adjectives_nouns"]
    pat = r"(?i)\b(?:" + "|".join(re.escape(w) for w in words) + r")\b"
    blocked = set(w.lower() for w in L["downtoners"] + L.get("non_intensifying_adverbs", []))
    blocked |= {_stem(w) for w in blocked}
    stems = {_stem(w.lower()) for w in words if " " not in w}
    return (re.compile(pat), [re.compile(p, re.I) for p in L["superlative_patterns"]], blocked, stems)


def _stem(w):
    for suf in ("ingly", "edly", "ly", "ing", "ed", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def is_reinforced(span, rx_words, rx_sups, blocked=frozenset(), stems=frozenset()):
    """A degree/manner intensifier, an extreme word, a superlative, or any -ly adverb that is
    not a downtoner or a domain adverb."""
    if rx_words.search(span) or any(r.search(span) for r in rx_sups):
        return True
    # stem match, so devastating / devastated / devastatingly all count as one entry
    for w in re.findall(r"[A-Za-z][\w'\u2019-]*", span.lower()):
        if w in blocked or _stem(w) in blocked:
            continue
        if _stem(w) in stems:
            return True
    for w in re.findall(r"\b\w{4,}ly\b", span.lower()):
        if w not in blocked:
            return True
    return False


def build_user_prompt_v3(digest, meta, text, sents):
    numbered = "\n".join(f"[S{i}] {text[s:e]}" for i, (s, e) in enumerate(sents))
    return f"""# TASK
Annotate ONE film review for aspect-based sentiment. Emit one row per (aspect, sentiment) pair.
Read the whole review first, then work sentence by sentence.

# A PAIR
Each pair needs a SENTIMENT TERM: an exact, contiguous, verbatim stretch of the sentence that
carries the judgement. It is usually evaluative wording ("devastating", "never slackens", "laid on
with a trowel"), but it may also be a DESCRIPTION of the situation whose polarity is obvious to
common sense ("nothing much here", "I checked my watch twice"). If common sense cannot settle the
sign, skip it.

Each pair should also have an ASPECT TERM: the smallest noun phrase naming what is judged ("the
score", "Song Kang-ho", "the staircases", "Bong's direction"). Leave the aspect "" ONLY when the
sentence truly contains no such noun phrase (a bare pronoun, a "there is" construction) AND the
category is still unambiguous from the surrounding sentences. Aim for an explicit aspect wherever
one exists; "" should be the minority of rows. The film's own title ("Parasite") and plain nouns
for it ("the movie", "the film") ARE explicit aspects - use them rather than leaving the aspect
empty, including when the sentiment adjectives sit in front of the noun ("an enjoyable, elegant,
scabrous movie" -> aspect "movie", three rows).

ATTRIBUTIVE ADJECTIVES: when the evaluative word sits in front of the noun it judges
("an insubstantial farce", "a toothless screwball comedy", "the sleek house"), the ASPECT is the
head noun ("farce", "comedy", "house") and the SENTIMENT is the modifier ("insubstantial",
"toothless", "sleek"). Do not quote the whole noun phrase as the sentiment term and then leave the
aspect empty - that throws the aspect away.

PRONOUN SUBJECTS: when a sentence opens with "It"/"This" standing for the film, use a noun for the
film ONLY if one actually appears in that sentence ("Parasite", "the movie", "the film", "suspense
drama"). The aspect term is always a span you copy out of the sentence - NEVER write a name that
is not there. If the sentence has only the pronoun, leave the aspect empty.

The aspect term and the sentiment term MUST NOT OVERLAP. If your aspect span contains your
sentiment span, shorten the aspect to the naming noun phrase, or leave it "".

ONE ROW PER DISTINCT SENTIMENT TERM. "The movie is enjoyable, elegant, scabrous" is THREE rows on
the same aspect, one per adjective - not one row quoting all three. Different aspects in one
sentence are of course separate rows too.

# SKIP (emit nothing) WHEN
- the sentence is plot summary, narrative recap, cast or credit listing, runtime, release info, or
  publication boilerplate (including a repeated headline or standfirst);
- no sentiment is expressed, or the stance is genuinely undecidable;
- the opinion belongs to someone else (a quoted critic, the audience, the director, "some say");
- the pair judges another film (Snowpiercer, Memories of Murder, Us, Shoplifters, Okja...),
  a festival, or a person's career rather than this film;
- you would have to quote a whole clause to capture the sentiment.
Skipping is free and correct: precision beats coverage.

# CONCISION (hard requirement)
- aspect term: 1-4 content words, no articles. GOOD "the score", "Hong Kyung-pyo's camera",
  "the bunker set". BAD "the film's focus on poverty, desperation and those in debt".
- sentiment term: 1-6 words, the evaluative core only.

# CATEGORY
Exactly one id from the list below. There is no OTHER: if nothing fits exactly, pick the CLOSEST.

# POLARITY (integer)
  +1 positive  /  -1 negative  - the normal case. "great", "wonderful", "terrific", "horrible",
     "dull", "thin" are +1 / -1.
  +2 / -2 ONLY when the sentiment term itself carries a REINFORCING token: a degree or manner
     intensifier ("very", "utterly", "crazy good", "savagely", "sumptuously", "spectacularly"),
     a superlative ("the best of the year", "finest"), or an extreme word ("masterpiece",
     "flawless", "searing", "unbearable").
  0 is NOT used. If you would call something neutral, skip it.
  Negation flips the sign. Downtoners ("rather", "somewhat", "a bit", "mildly") keep magnitude 1.
  Sarcasm is labelled by intended meaning, and only with textual evidence nearby.

# OUTPUT - JSON only, no prose, no code fence. One array per pair, in this exact order:
[sid, aspect_term, category, target, sentiment_term, polarity, confidence, note]
 sid = integer; polarity = -2|-1|1|2; confidence = "h"|"m"|"l"; note = "" unless confidence is "l".
 target = a roster name when a person is named, otherwise "".
Shape: {{"t": [[3, "the score", "MUSIC#GENERAL", "Jung Jae-il", "extraordinary", 1, "h", ""]]}}

{digest}

# REVIEW ({meta['publicatioName']}, {meta['criticName']}, RT: {meta['reviewState']})
{numbered}

JSON now.
"""


CONCESSIVE = re.compile(r"\b(?:yet|but|though|although|while|whereas|if not)\b", re.I)
COORD_ITEM = r"(?:[A-Za-z][\w'\u2019-]*ly\s+)?[A-Za-z][\w'\u2019-]*"


def split_coordination(text, start, end):
    """Split a sentiment span that is a plain coordinated list of adjectives into its items.

    Returns [(start, end), ...] - the original span unchanged when it is not a plain
    coordination. Concessive lists ('flawed, yet deeply human') are NOT split: the items
    usually carry opposite polarity and the annotator must assign each one.
    """
    span = text[start:end]
    if CONCESSIVE.search(span) or len(span.split()) > 8:
        return [(start, end)]
    if not re.fullmatch(rf"{COORD_ITEM}(?:\s*,\s*{COORD_ITEM})*(?:\s*,?\s*(?:and|or)\s+{COORD_ITEM})?",
                        span.strip()):
        return [(start, end)]
    out, pos = [], 0
    for m in re.finditer(COORD_ITEM, span):
        if m.group(0).lower() in {"and", "or"}:
            continue
        if m.start() < pos:
            continue
        pos = m.end()
        out.append((start + m.start(), start + m.end()))
    return out if len(out) > 1 else [(start, end)]


def validate_tuples_v3(raw, text, sents, roster_map, review_id, rx=None,
                       max_aspect_words=6, max_op_words=9, schema_roster=None, drop_categories=()):
    rows, probs = [], []
    rx_words, rx_sups, blocked = rx if rx else load_reinforcers()
    pol_map = {2: "very_positive", 1: "positive", -1: "negative", -2: "very_negative"}
    conf_map = {"h": "high", "m": "medium", "l": "low", "high": "high", "medium": "medium", "low": "low"}
    for k, t in enumerate(raw):
        if not isinstance(t, (list, tuple)) or len(t) < 7:
            probs.append({"review_id": review_id, "i": k, "problem": "malformed row", "detail": str(t)[:160]}); continue
        sid, asp, cat, tgt, opi, pol, conf = t[:7]
        note = re.sub(r"\s+", " ", str(t[7] if len(t) > 7 else "")).strip()
        try:
            sid = int(sid); pol_i = int(pol)
        except Exception:
            probs.append({"review_id": review_id, "i": k, "problem": "bad sid/polarity", "detail": str(t)[:80]}); continue
        if not (0 <= sid < len(sents)):
            probs.append({"review_id": review_id, "i": k, "problem": "sid out of range", "detail": str(sid)}); continue
        cat = str(cat).strip().upper().replace(" ", "")
        if cat in drop_categories:
            probs.append({"review_id": review_id, "i": k, "problem": "retired category used",
                          "detail": f"{cat}: {str(opi)[:50]}"}); continue
        if cat not in CATEGORIES:
            probs.append({"review_id": review_id, "i": k, "problem": "bad category", "detail": cat}); continue
        if pol_i not in pol_map:
            probs.append({"review_id": review_id, "i": k, "problem": "polarity not in {-2,-1,1,2}",
                          "detail": str(pol)}); continue
        if str(asp).strip().lower().strip(".,'\u2019s ") in COMPARISON_FILMS:
            probs.append({"review_id": review_id, "i": k, "problem": "dropped: out of scope (other film)",
                          "detail": f"{str(asp)[:30]} / {str(opi)[:50]}"}); continue
        s, e = sents[sid]
        op_sp = _find_span(text, opi, s, e)
        if op_sp is None:
            probs.append({"review_id": review_id, "i": k, "problem": "dropped: sentiment span not verbatim",
                          "detail": str(opi)[:70]}); continue
        asp_sp = _find_span(text, asp, s, e) if str(asp).strip() else None
        if str(asp).strip() and asp_sp is None:
            probs.append({"review_id": review_id, "i": k, "problem": "aspect span not verbatim -> implicit",
                          "detail": str(asp)[:70]})
            note = (note + " | aspect span unverifiable, recorded as implicit").strip(" |")
        if asp_sp and not (asp_sp[1] <= op_sp[0] or op_sp[1] <= asp_sp[0]):
            probs.append({"review_id": review_id, "i": k, "problem": "overlapping spans -> aspect implicit",
                          "detail": f"{text[asp_sp[0]:asp_sp[1]][:40]!r} / {text[op_sp[0]:op_sp[1]][:40]!r}"})
            note = (note + " | aspect overlapped the sentiment term and was dropped").strip(" |")
            asp_sp = None
        op_txt = text[op_sp[0]:op_sp[1]]
        asp_txt = text[asp_sp[0]:asp_sp[1]] if asp_sp else None
        if abs(pol_i) == 2 and not is_reinforced(op_txt, rx_words, rx_sups, blocked):
            probs.append({"review_id": review_id, "i": k, "problem": "downgraded: no reinforcing token",
                          "detail": op_txt[:60]})
            pol_i = 1 if pol_i > 0 else -1
            note = (note + " | intensity downgraded, no reinforcing token").strip(" |")
        long_flags = []
        if asp_txt and len(asp_txt.split()) > max_aspect_words: long_flags.append("aspect span long")
        if len(op_txt.split()) > max_op_words: long_flags.append("sentiment span long")
        if long_flags:
            probs.append({"review_id": review_id, "i": k, "problem": "; ".join(long_flags),
                          "detail": f"{str(asp_txt)[:40]!r} / {op_txt[:40]!r}"})
            note = (note + " | " + "; ".join(long_flags)).strip(" |")
        conf = conf_map.get(str(conf).strip().lower(), "medium")
        if conf == "low" and not note: note = "low confidence, no reason given by annotator"
        entity, attribute = (cat.split("#", 1) + [""])[:2]
        pieces = split_coordination(text, op_sp[0], op_sp[1])
        if len(pieces) > 1:
            probs.append({"review_id": review_id, "i": k, "problem": f"coordination split into {len(pieces)}",
                          "detail": op_txt[:60]})
        if CONCESSIVE.search(op_txt) and re.search(r",", op_txt):
            probs.append({"review_id": review_id, "i": k,
                          "problem": "concessive coordination left intact - check polarity by hand",
                          "detail": op_txt[:60]})
        for ps, pe in pieces:
            op_sp, op_txt = (ps, pe), text[ps:pe]
            pol_j = pol_i
            if abs(pol_j) == 2 and not is_reinforced(op_txt, rx_words, rx_sups, blocked):
                pol_j = 1 if pol_j > 0 else -1
            rows.append({"review_id": review_id, "sentence_id": sid,
                         "aspect_term": asp_txt, "aspect_start": asp_sp[0] if asp_sp else None,
                         "aspect_end": asp_sp[1] if asp_sp else None, "aspect_explicit": bool(asp_sp),
                         "category": cat, "entity": entity, "attribute": attribute,
                         "target": _final_target(cat, tgt, roster_map, schema_roster),
                         "opinion_term": op_txt, "opinion_start": op_sp[0], "opinion_end": op_sp[1],
                         "polarity": pol_map[pol_j], "polarity_score": pol_j,
                         "mixed_within_aspect": False, "confidence": conf, "note": note})
    # pronoun-subject rows sitting inside a comparison passage are the main misattribution risk:
    # "- but it manages at least to be humane" can be about the film being compared, not Parasite.
    film_rx = re.compile(r"\b(" + "|".join(re.escape(f) for f in sorted(COMPARISON_FILMS, key=len, reverse=True)) + r")\b", re.I)
    par_rx = re.compile(r"\bParasite\b|\bthe (?:film|movie|picture)\b", re.I)
    for r in rows:
        if r["aspect_explicit"]:
            continue
        sid = r["sentence_id"]
        window = " ".join(text[sents[j][0]:sents[j][1]] for j in range(max(0, sid - 2), sid + 1))
        here = text[sents[sid][0]:sents[sid][1]]
        if film_rx.search(window) and not par_rx.search(here):
            r["confidence"] = "low"
            r["note"] = (r["note"] + " | pronoun subject inside a comparison passage - verify the film referred to").strip(" |")
            probs.append({"review_id": review_id, "i": -1,
                          "problem": "flagged: possible comparison-film attribution", "detail": r["opinion_term"][:60]})

    import collections
    g = collections.defaultdict(list)
    for i, r in enumerate(rows):
        g[(r["sentence_id"], r["category"], r["target"], r["aspect_term"])].append(i)
    for idxs in g.values():
        if len({rows[i]["polarity_score"] > 0 for i in idxs}) > 1:
            for i in idxs: rows[i]["mixed_within_aspect"] = True
    return rows, probs
