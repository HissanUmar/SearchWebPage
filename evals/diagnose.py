"""Find out WHY answerable questions missed: is the text on the page, split across passages, or ranked low?

  python -m evals.diagnose                          # reads eval_results.json
  python -m evals.diagnose --colab https://xxxx     # also shows the rank of the best passage

Re-fetches each page and re-splits it, so it reflects the current split.py.
ctx  = verbatim match of the gold context in the best single passage (0-1)
ev   = share of the gold answer's content words in that passage; next = same for the runner-up passage
"""
import argparse
import json
import sys
from collections import Counter

import numpy as np

from pageqa.fetch import fetch
from pageqa.llm import LLM
from pageqa.rank import K, embed_chunks, score
from pageqa.split import split

from . import metrics

LEGEND = {
    "not_on_page": "neither the gold text nor the facts of the gold answer are in what the splitter kept "
                   "(dropped tags, JS content, or the page changed since the dataset was written)",
    "split_across_passages": "the facts are on the page but no single passage holds them (add overlap or bigger passages)",
    "ranked_low": "a passage holding the answer exists but ranking did not put it in the top-K",
    "model_skipped_it": "it was sent to the model, the model chose other passages or NONE",
    "selected_ok": "the best passage was returned to the user",
    "~": "suffix: the gold context is NOT verbatim on the page (paraphrased/edited); fix the dataset row",
}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="eval_results.json")
    ap.add_argument("--colab", help="optional: compute the rank of the best passage")
    a = ap.parse_args()

    data = json.load(open(a.results, encoding="utf-8"))
    rows = sorted((r for r in data["results"] if not r["expected_not_found"] and r["stage"] != "error"),
                  key=lambda r: r["input_order"])
    llm = LLM(a.colab) if a.colab else None
    pages, embs, counts = {}, {}, Counter()

    print(f"{'#':>2} {'eval stage':<15} {'ctx':>4} {'ev':>4} {'next':>4} {'psg':>4} {'topK':>4} {'rank':>4}  verdict / question")
    for r in rows:
        url = r["article"]
        if url not in pages:
            pages[url] = split(fetch(url))[1]
        ps = pages[url]
        texts = [metrics.source_text(c) for c in ps]
        gold, ans = r["gold_context"], r["gold_answer"]
        page_ctx = metrics.recall(gold, texts)
        page_ev = metrics.answer_recall(ans, "\n".join(texts))
        ctxs = [metrics.recall(gold, [t]) for t in texts]
        evs = [metrics.answer_recall(ans, t) for t in texts]
        b = max(range(len(ps)), key=lambda i: ctxs[i] + evs[i])
        nxt = sorted(evs, reverse=True)[1] if len(evs) > 1 else 0.0
        in_top = b in {t["passage_id"] for t in r.get("top_k", [])}
        in_sel = b in {t["passage_id"] for t in r.get("retrieved", [])}
        rank = "-"
        if llm:
            if url not in embs:
                embs[url] = embed_chunks(ps, llm)
            rank = list(np.argsort(-score(r["question"], ps, embs[url], llm))).index(b) + 1

        if page_ctx < 0.6 and page_ev < metrics.EVID:
            verdict = "not_on_page"
        elif not metrics.hit(ctxs[b], evs[b]):
            verdict = "split_across_passages"
        elif in_sel:
            verdict = "selected_ok"
        elif in_top:
            verdict = "model_skipped_it"
        else:
            verdict = "ranked_low"
        if page_ctx < 0.6:
            verdict += "~"
        counts[verdict] += 1
        print(f"{r['input_order']:>2} {r['stage']:<15} {ctxs[b]:>4.2f} {evs[b]:>4.2f} {nxt:>4.2f} {b:>4} {'Y' if in_top else 'N':>4} {rank!s:>4}  "
              f"{verdict} | {r['question'][:45]}")
        if verdict.startswith("not_on_page"):
            print(f"     gold : {' '.join(gold.split())[:110]}")
            print(f"     close: [{ps[b].path[:40]}] {' '.join(ps[b].text.split())[:110]}")

    print(f"\nverdicts: {dict(counts)}   (K={K}; psg/rank = best passage and its rank)")
    for v in LEGEND:
        if v == "~" or any(k.startswith(v) for k in counts):
            print(f"  {v}: {LEGEND[v]}")


if __name__ == "__main__":
    main()