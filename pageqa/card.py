from .models import Card


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