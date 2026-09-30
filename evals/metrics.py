import re

HIT = 0.8   # share of the gold paragraph's text that must appear to count as a hit
EVID = 0.4  # OR: share of the gold answer's content words that must appear (for gold contexts that are not verbatim)

# "not stated", "does not mention", "no information" ... (used on gold answers and on app answers)
_ABSENT = re.compile(
    r"\b(?:not|n't)\s+(?:state[ds]?|contain(?:s|ed)?|mention(?:s|ed)?|provide[ds]?|specif(?:y|ies|ied)|"
    r"includ(?:e|es|ed)|present|disclosed?|found|available)\b|\bno\s+(?:information|mention|details?|pricing)\b",
    re.I,
)
_STOP = {"that", "this", "with", "from", "were", "have", "which", "their", "there", "when", "what", "would",
         "into", "been", "they", "about", "also", "after", "while", "because", "them", "then", "than",
         "some", "other", "very", "just", "only", "over", "such", "more", "each", "will", "does"}


def tokens(s):
    return re.findall(r"\w+", s.lower())


def _squash(s):
    """lowercase, keep letters/digits/underscore only: immune to whitespace, punctuation and quote differences"""
    return re.sub(r"\W+", "", s.lower())


def recall(gold, texts, n=12):
    """Share of the gold paragraph's 12-character shingles found in any of the texts (0..1).
    Character based on purpose: gold code often has its newlines stripped, which merges words."""
    g = _squash(gold)
    if not g:
        return 0.0
    n = min(n, len(g))
    want = {g[i:i + n] for i in range(len(g) - n + 1)}
    have = set()
    for t in texts:
        s = _squash(t)
        have |= {s[i:i + n] for i in range(len(s) - n + 1)}
    return len(want & have) / len(want)


def _content(s):
    return {t for t in tokens(s) if len(t) > 3 and t not in _STOP}


def answer_recall(gold_answer, candidate):
    """Share of the gold answer's content words that appear in the candidate text (0..1).
    Used on the app's answer, and as 'evidence' on the retrieved passages."""
    g = _content(gold_answer)
    return len(g & _content(candidate)) / len(g) if g else 0.0


def is_unanswerable(row):
    """Optional 'answerable' column (yes/no) wins; otherwise detect 'the article does not state...' in the gold answer."""
    v = row.get("answerable", "").strip().lower()
    if v:
        return v in ("no", "false", "0", "n")
    return bool(_ABSENT.search(row["answer"][:200]))


def says_not_found(text):
    return bool(_ABSENT.search(text[:300]))


def source_text(chunk):
    return "\n".join([chunk.text] + list(chunk.code))


def hit(recall_, evidence):
    """The gold paragraph was found verbatim, or the facts of the gold answer are in the text."""
    return recall_ >= HIT or evidence >= EVID