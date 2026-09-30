"""Rank passages by embedding cosine + keyword overlap."""

import re

import numpy as np


K = 4

STOP = {"the", "how", "what", "give", "for", "with", "and", "that", "this", "from", "use",
        "using", "can", "does", "code", "show", "get", "you", "are", "your"}


def _embed_text(c):
    # the embedder has a short window (~128 tokens): heading path + start of the section
    return (c.path + ". " + c.text)[:500]


def embed_chunks(chunks, llm):
    return np.array(llm.embed([_embed_text(c) for c in chunks]), dtype=float)


def _cos(m, v):
    m = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
    return m @ (v / (np.linalg.norm(v) + 1e-9))


def _kw(question, c):
    # keyword overlap over the FULL section (incl. code), covers what the short embedding misses
    toks = {t for t in re.findall(r"\w+", question.lower()) if len(t) > 2 and t not in STOP}
    if not toks:
        return 0.0
    hay = (c.path + " " + c.text + " " + " ".join(c.code)).lower()
    return sum(t in hay for t in toks) / len(toks)


def score(question, chunks, embs, llm):
    q = np.array(llm.embed([question])[0], dtype=float)
    return _cos(embs, q) + 0.3 * np.array([_kw(question, c) for c in chunks])


def rank(question, chunks, embs, llm, k=K):
    scores = score(question, chunks, embs, llm)
    return [chunks[i] for i in np.argsort(-scores)[:k]]