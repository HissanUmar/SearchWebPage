import requests


class LLM:
    """Only file that knows about the Colab/ngrok server."""

    def __init__(self, base_url):
        self.url = base_url.rstrip("/")
        self.headers = {"ngrok-skip-browser-warning": "1"}

    def _post(self, path, body, timeout):
        return requests.post(self.url + path, json=body, headers=self.headers, timeout=timeout)

    def generate(self, prompt):
        r = self._post("/generate", {"text": prompt}, 180)
        r.raise_for_status()
        return r.json()["response"]

    def embed(self, texts):
        r = self._post("/embed_batch", {"texts": texts}, 300)
        if r.status_code == 404:  # batch endpoint missing: fall back to one call per text
            out = []
            for t in texts:
                x = self._post("/embed", {"text": t}, 60)
                x.raise_for_status()
                out.append(x.json()["embedding"])
            return out
        r.raise_for_status()
        return r.json()["embeddings"]
