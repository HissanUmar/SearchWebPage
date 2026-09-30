"""Ingest: turn a URL into a Page (fetch → split → card)."""

import re

import requests
from bs4 import BeautifulSoup, Tag, Comment

from .models import Card, Chunk


# --- fetch ---

def fetch(url):
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (page-qa)"}, timeout=30)
    r.raise_for_status()
    return r.text


# --- split ---

HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4}
BLOCK = {"p", "li", "ul", "ol", "pre", "blockquote", "table", "div", "section", "dl",
         "h1", "h2", "h3", "h4", "h5", "h6"}
TEXT = ["p", "li", "blockquote", "td", "th", "dd"]
MAX = 1000  # max chars per passage
NOISE = ["script", "style", "svg", "nav", "footer", "form", "button", "noscript"]


def _clean(s):
    return " ".join(re.sub(r"[\u200b-\u200d\ufeff]", "", s).split())


def _own_text(el):
    """Text of el without its nested block children (those are visited on their own)."""
    parts = []
    for n in el.children:
        if isinstance(n, Comment):
            continue
        if isinstance(n, Tag):
            if n.name not in BLOCK:
                parts.append(n.get_text())
        else:
            parts.append(str(n))
    return _clean("".join(parts))


def _pieces(t):
    """Break an over-long paragraph at sentence boundaries."""
    if len(t) <= MAX:
        return [t]
    out, cur = [], ""
    for sent in re.split(r"(?<=[.!?])\s+", t):
        if cur and len(cur) + len(sent) + 1 > MAX:
            out.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    return out + [cur]


def _anchor(h):
    if h.get("id"):
        return h["id"]
    c = h.find(id=True)
    if c:
        return c["id"]
    a = h.find("a", href=True)
    if a and a["href"].startswith("#") and len(a["href"]) > 1:
        return a["href"][1:]
    p = h.find_parent("section", id=True)
    return p["id"] if p else ""


def split(html):
    """html -> (page title, list[Chunk]); one chunk per heading section."""
    soup = BeautifulSoup(html, "html.parser")
    title = _clean(soup.title.get_text()) if soup.title else ""
    root = soup.find("main") or soup.find("article") or soup.body or soup
    for t in root.find_all(NOISE):
        t.decompose()

    chunks, stack, items = [], [], []
    path, anchor = title or "Intro", ""

    def flush():
        """Turn the section's items into passages of <= MAX chars; a code block ends a passage."""
        buf, code = [], []

        def emit():
            nonlocal buf, code
            if buf or code:
                chunks.append(Chunk(len(chunks), path, anchor, "\n".join(buf), code))
            buf, code = [], []

        for kind, val in items:
            if kind == "c":
                code.append(val)
                emit()
            else:
                for piece in _pieces(val):
                    if buf and len("\n".join(buf)) + len(piece) > MAX:
                        emit()
                    buf.append(piece)
        emit()
        items.clear()

    for el in root.find_all(list(HEADINGS) + TEXT + ["pre"]):
        if el.name in HEADINGS:
            label = _clean(el.get_text())
            if not label:
                continue
            flush()
            lvl = HEADINGS[el.name]
            while stack and stack[-1][0] >= lvl:
                stack.pop()
            stack.append((lvl, label))
            path = " > ".join(t for _, t in stack)
            anchor = _anchor(el)
        elif el.name == "pre":
            c = el.get_text().strip("\n")
            if c.strip():
                items.append(("c", c))
        else:
            t = _own_text(el)
            if t:
                items.append(("t", t))
    flush()
    return title, chunks


# --- card ---

def card_parts(title, chunks):
    h1 = chunks[0].path.split(" > ")[0]
    intro = next((c.text for c in chunks if c.text), "")[:300]
    outline = list(dict.fromkeys(c.path for c in chunks))[:30]
    return h1, intro, outline


def page_card(title, chunks, llm):
    """Small summary of what the page is for, built from its own structure (one cheap LLM call)."""
    h1, intro, outline = card_parts(title, chunks)
    prompt = (
        "In 1-2 sentences, say what this web page is for. Use only the info below.\n\n"
        f"Title: {title}\nMain heading: {h1}\nIntro: {intro}\nSections:\n" + "\n".join(outline)
    )
    purpose = llm.generate(prompt).strip()[:400]
    return Card(title, h1, intro, outline, purpose)