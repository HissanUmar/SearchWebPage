from . import cache
from .ask import ask
from .attach import attach
from .card import page_card
from .fetch import fetch
from .llm import LLM
from .models import Answer, Page
from .rank import embed_chunks, rank
from .split import split


def _load(url, llm):
    page = cache.get(url)
    if page:
        return page
    title, chunks = split(fetch(url))
    if not chunks:
        raise ValueError("No readable content found (page may be JavaScript-rendered).")
    page = Page(url, title, chunks)
    page.card = page_card(title, chunks, llm)
    page.embs = embed_chunks(chunks, llm)
    cache.put(url, page)
    return page


def run(url, question, colab_url):
    llm = LLM(colab_url)
    page = _load(url, llm)
    top = rank(question, page.chunks, page.embs, llm)
    ids, explanation = ask(page.card, top, question, llm)
    sources = attach(ids, page.chunks, url)
    return Answer(page.card.purpose, explanation if sources else "", sources, bool(sources), top)
