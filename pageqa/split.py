import re
from bs4 import BeautifulSoup, Tag, Comment
from .models import Chunk

HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4}
BLOCK = {"p", "li", "ul", "ol", "pre", "blockquote", "table", "div", "section", "dl",
         "h1", "h2", "h3", "h4", "h5", "h6"}
TEXT = ["p", "li", "blockquote", "td", "th", "dd"]
MAX = 1000  # max chars per passage
NOISE = ["script", "style", "svg", "nav", "footer", "form", "button", "noscript"]


def _clean(s):
    return " ".join(s.split())


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