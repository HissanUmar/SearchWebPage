import requests


def fetch(url):
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (page-qa)"}, timeout=30)
    r.raise_for_status()
    return r.text
