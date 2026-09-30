# CLAUDE.md

Page Q&A: user gives a page URL and a question. The app answers using only that page and shows where the answer lives (section, verbatim text/code, link). Full reasoning is in `PLAN.md`. How pages are split into passages is in `CHUNKING.md`. How quality is scored is in `EVAL.md`.

## Commands
```
pip install -r requirements.txt
streamlit run app.py                     # paste the Colab ngrok URL in the sidebar
python check.py <page-url>               # how the page is chunked (also: --full --ids 8,15 --find WORD --dump FILE)
python check.py <url> --q "..." --colab <ngrok-url>   # ranking scores, exact prompt, raw model reply
```
The LLM runs on Colab (Ollama + FastAPI + ngrok). The app cannot answer without that server up.

## Layout
```
app.py             Streamlit UI, only calls pipeline.run
pageqa/
  models.py        dataclasses: Chunk, Card, Page, Source, Answer
  llm.py           Colab client (/generate, /embed_batch, /embed fallback)
  fetch.py         download HTML
  split.py         HTML -> passages (<=1000 chars, heading path, anchor, text, code)
  card.py          page purpose (title, h1, intro, outline + one LLM call)
  rank.py          score() + rank(): embeddings + keyword overlap -> top-K (K=4)
  ask.py           build_prompt, parse (`IDS:` / `EXPLANATION:`, defensive), ask
  attach.py        verbatim text/code + link for chosen passages
  cache.py         in-memory per-URL cache
  pipeline.py      run(url, question, colab_url)
evals/             metrics.py (lexical scores), judge.py (LLM judge), run.py (CLI, JSON report), questions.csv
```

## Rules
- Keep it a lean MVP. No new files, dependencies or abstractions unless something actually needs them.
- `app.py` imports only `pipeline.run`. Steps depend on `models.py` and `llm.py`, never on each other. Only `pipeline.py` wires steps.
- Only `llm.py` talks to Colab.
- The model returns passage IDs and a short explanation. It must never produce the code or quotes shown to the user; `attach.py` takes them from the scraped page.
- "Not found on this page" is a valid answer. Do not add fallbacks that guess.
- Before changing behavior, check `PLAN.md` for the reason it was built that way, and update it when a decision changes.

## Gotchas
- Small model (Qwen3 4B): context is limited. Raising `MAX` in `split.py` or `MAX_TEXT` in `ask.py` can overflow it. Colab must set `num_ctx=8192`.
- The embedder reads about 128 tokens, so `rank.py` also scores keyword overlap on the full passage.
- The ngrok URL changes on every Colab restart.
- Pages with no headings depend on passage splitting. If a query wrongly returns "not found", or an answer looks off, run `check.py` first (see `CHUNKING.md`).
- Text not inside `p`/`li`/`td`/`th`/`dd`/`blockquote` (and `h5`/`h6`, `summary`, `figcaption`) is dropped by `split.py`.
- JS-rendered pages return no content. Code is read only from `<pre>` tags.

## Testing
The `evals/` pipeline scores answer quality on a labelled question set (`EVAL.md`). No unit tests yet. Checked so far with a mock Colab server (`/generate`, `/embed_batch`, `/embed`) and local HTML pages. Add tests for `split.py` and `attach.py` first.
