_store = {}


def get(url):
    return _store.get(url)


def put(url, page):
    _store[url] = page
