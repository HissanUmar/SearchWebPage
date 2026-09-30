# EVAL

Scores the app against a labelled question set and writes a ranked JSON report. The judge is the same Colab model the app uses.

## Run

From the repo root, with the Colab server up:

```
python -m evals.run --colab https://xxxx.ngrok-free.app
python -m evals.run --colab URL --limit 3            # quick smoke test
python -m evals.run --colab URL --no-judge           # lexical scores only, much faster
python -m evals.run --colab URL --csv other.csv --out report.json
```

Defaults: `--csv evals/questions.csv`, `--out eval_results.json`. Cost is about 5 model calls per question (1 embed, 1 answer, 3 judge calls), plus one page load per article.

## Dataset format

CSV columns: `article link`, `question`, `answer`, `context`. Optional `answerable` (yes/no).
- `article link` may be `[url](url)` or a plain URL.
- `context` should be copied verbatim from the page.
- A row is treated as **unanswerable** (correct behavior = "not found") if `answerable` says no, or if the gold answer starts with wording like "does not state / does not contain / not mentioned". With the current file this marks rows 10 and 20.

## What is scored (per question)

| Score | How | Range |
|---|---|---|
| `context_recall` | Share of the gold paragraph's 12-character shingles found in the passages the app returned. Ignores case, punctuation and whitespace (so code with stripped newlines still matches) | 0-1 |
| `topk_recall` | Same, against the top-4 passages sent to the model (before it chose) | 0-1 |
| `source_evidence`, `topk_evidence` | Share of the gold answer's content words found in the returned / top-4 passages. Works when the gold context is paraphrased rather than verbatim | 0-1 |
| `context_match` | max(`context_recall`, `source_evidence`); the context part of the overall score | 0-1 |
| `answer_recall` | Share of the gold answer's content words found in the app's explanation | 0-1 |
| `correctness` (judge) | Does the app's explanation state the same key facts as the gold answer | 0-5, stored /5 |
| `grounded` (judge) | Is every claim in the explanation supported by the returned passages | 0-5, stored /5 |
| `context_judge` (judge) | Do the returned passages contain the information in the gold paragraph | 0-5, stored /5 |
| `abstained` | Unanswerable rows only: the app returned "not found" or its explanation says the info is not there | 0 or 1 |

## Overall score and rank

- Answerable: weighted mean of correctness 0.30, grounded 0.10, context_judge 0.20, context_match 0.25, answer_recall 0.15. Missing parts (for example with `--no-judge`) are dropped and the weights renormalized. Weights are in `WEIGHTS` in `evals/run.py`.
- Unanswerable: 1.0 if the app said "not found", otherwise the judge's correctness (0 if none).
- Rows are ranked by overall score, highest first. A page that fails to load scores 0 with `stage: "error"`.

## Stage (where it failed first)

| Stage | Meaning |
|---|---|
| `retrieval_miss` | Gold paragraph was not in the top-4 passages (fix chunking or ranking) |
| `selection_miss` | It was sent to the model, but the model did not pick it or said NONE (fix prompt or model) |
| `answer_weak` | Right passage returned, but the explanation is wrong or thin |
| `ok` | Passed |
| `correct_abstain` / `false_answer` | Unanswerable row: app said "not found" / app answered anyway |
| `error` | Fetch or server failure |

A passage counts as a hit if verbatim recall >= 0.8 (`HIT`) or evidence >= 0.4 (`EVID`, both in `evals/metrics.py`). `EVID` is a guess, not calibrated: `python -m evals.diagnose` prints `ev` (best passage) next to `next` (runner-up) so you can check the gap.

## JSON layout

```
meta      created, csv, questions, top_k, judge, weights, hit_threshold, notes
summary   mean_overall, stages, answerable{hit rates, mean scores}, unanswerable{abstain_rate}, by_article
results   ranked list; each item has: rank, article, question, expected_not_found,
          gold_answer, gold_context, app_answer, found, abstained, page_purpose,
          retrieved[{passage_id, path, link, text}], top_k[{passage_id, path, preview}],
          scores{...}, judge{name: {score, reason}}, overall, stage, seconds, input_order
```

## How to read it

1. `summary.stages` first. It says which step is failing across the whole set.
2. `retrieval_hit_rate` vs `selection_hit_rate`: a big gap means ranking is fine and the model is choosing badly.
3. For a bad row, compare `gold_context` with `retrieved[].text` and `top_k`, then use `python check.py <url> --q "<question>" --colab <url>` to see the exact prompt and raw reply.

## Caveats

- The judge is the same 4B model that produced the answer. It can grade itself generously or fail the format (a failed judge call gives `score: null` and is left out of the mean). Trust the lexical scores and the stage counts more than the judge scores.
- Verbatim recall assumes the gold `context` matches the page text. If it was paraphrased, verbatim recall drops even when the app is right; the evidence score covers that, but the cleaner fix is to paste the real paragraph. `python -m evals.diagnose` marks such rows with `~`.
- 20 questions is a small sample: one row moves the mean by about 5 points.
- Unanswerable detection for the current file is wording-based. Add an `answerable` column to be explicit.

## Diagnosing misses

```
python -m evals.diagnose --colab https://xxxx
```

Re-fetches each page and labels every answerable row: `not_on_page`, `split_across_passages`, `ranked_low` (with the rank of the best passage), `model_skipped_it`, `selected_ok`. A `~` suffix means the gold context is not verbatim on the page.