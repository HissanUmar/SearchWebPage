# Page Q&A

Give it a page URL and a question. It answers using only that page and shows where the answer lives (section, verbatim text/code, `url#anchor` link).

## How it works

```
fetch -> split by headings -> page card (purpose) -> rank sections (embeddings + keywords)
      -> LLM picks section IDs + explains -> code attaches verbatim text, code, link
```

The LLM never writes code or quotes. It only returns section IDs and a short explanation, so code and locations always come straight from the scraped page.

## Setup

**1. Colab (LLM server).** Use your Ollama + FastAPI + ngrok notebook with three changes:

```python
# uncomment the embedder
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# larger context window
llm = OllamaLLM(model="qwen3:4b-instruct-2507-q4_K_M", temperature=0.1, num_ctx=8192)

# add next to /embed, before starting the server
class Batch(BaseModel):
    texts: list[str]

@app.post("/embed_batch")
def embed_batch(b: Batch):
    return {"embeddings": embedder.encode(b.texts).tolist()}
```

Run the notebook and copy the printed ngrok URL. It changes every restart.

**2. App.**

```
pip install -r requirements.txt
streamlit run app.py
```

Paste the ngrok URL in the sidebar, enter a page URL and a question, click Ask.

## Structure

```
app.py              Streamlit UI (only calls pipeline.run)
pageqa/
  models.py         Chunk, Card, Page, Source, Answer
  llm.py            Colab client (/generate, /embed_batch, /embed fallback)
  fetch.py          download HTML
  split.py          HTML -> passages (<=1000 chars, with heading path, anchor, text, code)
  card.py           page purpose from title, h1, intro, outline
  rank.py           embeddings + keyword overlap -> top-k sections
  ask.py            prompt + parse model reply (section IDs, explanation)
  attach.py         verbatim text/code + deep link for chosen sections
  cache.py          per-URL cache (in memory)
  pipeline.py       run(url, question, colab_url)
```

## Limits

- Long sections and pages without headings are cut into ~1000-char passages. Without a heading id the link uses a scroll-to-text fragment (`#:~:text=`), which needs a current browser.
- Single page only. Links to other pages are not followed.
- JavaScript-rendered pages return "no readable content" (no browser fallback).
- Code is read from `<pre>` tags; sites that render code differently may show none.
- Cache is in memory and clears when Streamlit restarts.
- Small 4B model: if answers look off, check the prompt in `ask.py` first.