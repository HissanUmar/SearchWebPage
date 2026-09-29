from urllib.parse import quote

from .models import Source


def _link(c, base):
    if c.anchor:
        return f"{base}#{c.anchor}"
    words = c.text.split()[:8]  # no heading id: scroll-to-text link (supported by current browsers)
    if not words:
        return base
    return f"{base}#:~:text=" + quote(" ".join(words), safe="").replace("-", "%2D")


def attach(ids, chunks, url):
    """Pull verbatim text/code and the deep link from the scraped page."""
    by_id = {c.id: c for c in chunks}
    base = url.split("#")[0]
    return [Source(by_id[i], _link(by_id[i], base)) for i in ids if i in by_id]