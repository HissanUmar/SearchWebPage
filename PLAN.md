# PLAN

Decisions made while building Page Q&A, and why. Read this before changing behavior.

## 1. Business decisions

### Problem
A person has a specific web page (e.g. a docs page) and a question. They want the answer from **that page only**, and they want to know **where on the page it is**.

### Users and I/O
- Users: developers reading docs, or anyone facing a long page.
- In: page URL + natural-language question.
- Out: short explanation, verbatim text/code from the page, and a link to the exact section.

### Scope
- Single page only. Links to other pages are not followed.
- MVP: lean, end to end, Streamlit UI, minimal files and dependencies.
- Non-goals: multi-page crawling, JavaScript-rendered pages, login-gated pages, persistence, multi-user, auth.

### Why this instead of the alternatives
| Alternative | Gap this closes |
|---|---|
| Ctrl+F | Finds by intent ("persist state"), not exact keyword |
| General chatbot | Answers only from the chosen page, no training-data mixing, says "not found" instead of guessing |
| Standard RAG | No corpus, ingestion or index. Paste a URL and ask |
| Chatbot with browsing | Location of the answer is verified against the page, not claimed by the model |

### Honest assessment
Differentiation is thin. A chatbot with browsing does a rough version of this. The real edge is **trust** (verified location, verbatim code, "not on this page") plus **zero setup**. It is a strong utility, not a moat. It is worse than RAG when an answer spans several pages.

### Working mode
Requirements first (tech lead questions, PM answers), then build. Keep it an MVP: nothing that isn't used.

## 2. Technical decisions

### Stack
- **UI:** Streamlit (`app.py`). Fastest end-to-end web UI, no frontend code.
- **LLM:** Ollama on Colab, model `qwen3:4b-instruct-2507-q4_K_M`, served by FastAPI and exposed through ngrok. Endpoints: `/generate {text}`, `/embed {text}`, `/embed_batch {texts}`.
- **Embedder (Colab side):** `paraphrase-multilingual-MiniLM-L12-v2`.
- **Python deps:** streamlit, requests, beautifulsoup4, numpy.

### Pipeline
```
fetch -> split into passages -> page card -> rank -> ask (IDs + explanation) -> attach (text, code, link)
```

### Decision log

| # | Decision | Why | Trade-off |
|---|---|---|---|
| 1 | **Model returns section IDs, code pulls the text/code** | 4B model is unreliable at copying code and strict JSON. Location and code are guaranteed real | Model cannot synthesize across passages beyond a short explanation |
| 2 | **Retrieval is required** (no "send whole page") | 4B model with a few-thousand-token context truncates silently | Adds an embedding step |
| 3 | **Structure-aware splitting** at h1-h4 under `main`/`article`/`body`; nav, footer, forms, scripts removed | Keeps heading path and anchor with each chunk, keeps code whole | Quality depends on the page's markup |
| 4 | **Passages of at most 1000 chars**; long paragraphs split at sentence boundaries; a code block ends a passage | Flat pages (no headings) became one giant chunk, and the model only saw its first characters. Found from a real failed query | More chunks to embed |
| 5 | **Page card, not a full-page summary** | Full summary needs map-reduce over a small model: slow, hallucination risk. Title, h1, intro and outline already state the page's purpose. One cheap LLM call writes a 1-2 sentence purpose | Weak on pages with no clear structure |
| 6 | **Embeddings plus keyword overlap** (cosine + 0.3 x keyword score), top-4 | Embedder only reads about 128 tokens, so it misses text later in a passage and code. Keyword overlap over the full passage covers that | Weights are untuned |
| 7 | **Line format `IDS:` / `EXPLANATION:`, not JSON** | Small models break JSON more often. Parser is defensive: invalid IDs dropped, `NONE` means not found | Model must follow the format |
| 8 | **"Not found" is a valid outcome** | Trust is the product. No guessing | False "not found" is possible if ranking misses |
| 9 | **Link = `url#anchor`**, else scroll-to-text fragment `#:~:text=` | Many pages have no heading ids | Text fragments need a current browser and can miss if page text differs |
| 10 | **`llm.py` is the only file that knows about Colab** | Swapping models or hosts touches one file | None |
| 11 | **`/embed_batch` with fallback to `/embed`** | One call per passage is slow on the first question | Notebook needs the extra endpoint for speed |
| 12 | **`ngrok-skip-browser-warning` header on every call** | Free ngrok can return an interstitial page to API calls | None |
| 13 | **In-memory cache per URL** (chunks, card, embeddings) | Repeat questions skip scrape and embedding | Lost on restart, never invalidated while running |
| 14 | **Package split into one concern per file** | Requested, so the project can grow without a single large file | About 11 small files for a small MVP |
| 15 | **Ngrok URL entered in the sidebar** | It changes on every Colab restart | Manual paste |

### Module rules
- `app.py` imports only `pipeline.run`.
- Steps depend on `models.py` and `llm.py`, never on each other. Only `pipeline.py` connects steps.
- The model never writes code or quotes shown to the user.

## 3. Known gaps
- Not run against the real Qwen model on Colab. The `IDS:`/`EXPLANATION:` prompt may need tuning.
- Code extraction assumes code sits in `<pre>`. Sites with live code editors may show none. Untested on react.dev.
- JavaScript-rendered pages return "no readable content". No browser fallback.
- Sites with bot protection may block the plain HTTP fetch.
- No relevance threshold: always sends top-4 passages and relies on the model to say `NONE`.
- No automated tests. Everything was checked against a mock Colab server and local HTML.
- Cache is in memory only.

## 4. Possible next steps (not committed)
1. Tests for `split.py` and `attach.py` first, since they must not break.
2. Headless-browser fallback for JS pages.
3. Tune keyword weight and top-k on real pages.
4. Relevance threshold before calling the model.
5. Persistent cache.

## 5. Diagnostics
`python check.py <url> [keyword]` prints passage count and lengths, and which passage contains the keyword. Use it first when a query wrongly returns "not found".
