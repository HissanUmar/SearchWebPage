from .answer import ask, attach
from .ingest import fetch, page_card, split
from .llm import LLM
from .models import Answer, Page
from .retrieve import embed_chunks, rank


# --- cache (in-memory, per-URL, cleared on process restart) ---

_store = {}


def _cache_get(url):
    return _store.get(url)


def _cache_put(url, page):
    _store[url] = page


# --- pipeline ---

def _load(url, llm):
    page = _cache_get(url)
    if page:
        return page
    title, chunks = split(fetch(url))
    if not chunks:
        raise ValueError("No readable content found (page may be JavaScript-rendered).")
    page = Page(url, title, chunks)
    page.card = page_card(title, chunks, llm)
    page.embs = embed_chunks(chunks, llm)
    _cache_put(url, page)
    return page


def run(url, question, colab_url):
    llm = LLM(colab_url)
    page = _load(url, llm)
    top = rank(question, page.chunks, page.embs, llm)
    ids, explanation = ask(page.card, top, question, llm)
    sources = attach(ids, page.chunks, url)
    return Answer(page.card.purpose, explanation if sources else "", sources, bool(sources), top)