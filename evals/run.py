"""Score the app against a labelled question set and write a ranked JSON report.

  python -m evals.run --colab https://xxxx.ngrok-free.app
  python -m evals.run --colab URL --csv other.csv --out report.json --limit 5 --no-judge

CSV columns: article link, question, answer, context   (optional: answerable = yes/no)
"""
import argparse
import csv
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone

# evals/run.py
from pageqa.llm import LLM
from pageqa.pipeline import run
from pageqa.retrieve import K     # was: from pageqa.rank import K

from . import judge, metrics

# Weights for the overall score of an answerable question. Missing parts (e.g. --no-judge) are dropped and the rest renormalized.
WEIGHTS = {"correctness": 0.30, "grounded": 0.10, "context_judge": 0.20, "context_match": 0.25, "answer_recall": 0.15}


def load(path):
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            r = {k.strip().lower(): (v or "").strip() for k, v in r.items()}
            m = re.search(r"\((https?://[^)]+)\)", r["article link"])  # links come as [url](url)
            r["url"] = m.group(1) if m else r["article link"].strip("[]")
            rows.append(r)
    return rows


def overall(rec):
    sc = rec["scores"]
    if rec["expected_not_found"]:  # correct behaviour is to say "not found"
        if sc.get("abstained"):
            return 1.0
        return sc.get("correctness") or 0.0
    parts = {k: sc[k] for k in WEIGHTS if sc.get(k) is not None}
    return sum(WEIGHTS[k] * v for k, v in parts.items()) / sum(WEIGHTS[k] for k in parts)


def stage(rec):
    """Where the pipeline failed first: ranking, passage selection, or the answer itself."""
    sc = rec["scores"]
    if rec["expected_not_found"]:
        return "correct_abstain" if sc["abstained"] else "false_answer"
    if not metrics.hit(sc["topk_recall"], sc["topk_evidence"]):
        return "retrieval_miss"  # gold paragraph / its facts were not among the passages sent to the model
    if not rec["found"] or not metrics.hit(sc["context_recall"], sc["source_evidence"]):
        return "selection_miss"  # it was sent, but the model did not pick it (or said NONE)
    c = sc.get("correctness")
    return "answer_weak" if (c is not None and c <= 0.4) or sc["answer_recall"] < 0.3 else "ok"


def evaluate(row, colab, llm, use_judge):
    url, q, unans = row["url"], row["question"], metrics.is_unanswerable(row)
    rec = {"article": url, "question": q, "expected_not_found": unans,
           "gold_answer": row["answer"], "gold_context": row["context"]}
    t0 = time.time()
    try:
        a = run(url, q, colab)
    except Exception as e:
        rec.update(error=f"{type(e).__name__}: {e}", scores={}, judge={}, overall=0.0, stage="error")
        return rec

    src = [metrics.source_text(s.chunk) for s in a.sources]
    top = [metrics.source_text(c) for c in a.top]
    abstained = (not a.found) or metrics.says_not_found(a.explanation)
    rec.update(
        found=a.found, abstained=abstained, app_answer=a.explanation, page_purpose=a.purpose,
        retrieved=[{"passage_id": s.chunk.id, "path": s.chunk.path, "link": s.link, "text": t}
                   for s, t in zip(a.sources, src)],
        top_k=[{"passage_id": c.id, "path": c.path, "preview": " ".join(t.split())[:120]}
               for c, t in zip(a.top, top)],
    )

    scores, jd = {}, {}
    joined = "\n\n".join(src)
    if unans:
        scores["abstained"] = 1.0 if abstained else 0.0
        if use_judge and not abstained:
            jd["correctness"] = judge.correctness(llm, q, row["answer"], a.explanation)
    else:
        scores["context_recall"] = metrics.recall(row["context"], src)
        scores["topk_recall"] = metrics.recall(row["context"], top)
        scores["source_evidence"] = metrics.answer_recall(row["answer"], joined)
        scores["topk_evidence"] = metrics.answer_recall(row["answer"], "\n".join(top))
        scores["context_match"] = max(scores["context_recall"], scores["source_evidence"])
        scores["answer_recall"] = metrics.answer_recall(row["answer"], a.explanation)
        if use_judge and a.found:
            jd["correctness"] = judge.correctness(llm, q, row["answer"], a.explanation)
            jd["grounded"] = judge.grounded(llm, q, a.explanation, joined)
            jd["context_judge"] = judge.context(llm, q, row["context"], joined)
        elif use_judge:
            jd["correctness"] = jd["context_judge"] = {"score": 0, "reason": "app returned not found"}
    for k, j in jd.items():
        scores[k] = None if j["score"] is None else j["score"] / 5
    rec["scores"] = {k: None if v is None else round(v, 3) for k, v in scores.items()}
    rec["judge"] = jd
    rec["seconds"] = round(time.time() - t0, 1)
    rec["overall"] = round(overall(rec), 3)
    rec["stage"] = stage(rec)
    return rec


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def summarize(recs):
    ok = [r for r in recs if r["stage"] != "error"]
    ans = [r for r in ok if not r["expected_not_found"]]
    unans = [r for r in ok if r["expected_not_found"]]

    def avg(rs, k):
        return _mean([r["scores"].get(k) for r in rs])

    def hit(rs, a, b):
        return _mean([1.0 if metrics.hit(r["scores"][a], r["scores"][b]) else 0.0 for r in rs])

    by_article = {}
    for u in dict.fromkeys(r["article"] for r in recs):
        rs = [r for r in recs if r["article"] == u]
        by_article[u] = {"n": len(rs), "mean_overall": _mean([r["overall"] for r in rs]),
                         "stages": dict(Counter(r["stage"] for r in rs))}
    return {
        "questions": len(recs),
        "errors": len(recs) - len(ok),
        "mean_overall": _mean([r["overall"] for r in recs]),
        "stages": dict(Counter(r["stage"] for r in recs)),
        "answerable": {
            "n": len(ans),
            "retrieval_hit_rate": hit(ans, "topk_recall", "topk_evidence") if ans else None,
            "selection_hit_rate": hit(ans, "context_recall", "source_evidence") if ans else None,
            "mean_context_recall": avg(ans, "context_recall"),
            "mean_source_evidence": avg(ans, "source_evidence"),
            "mean_answer_recall": avg(ans, "answer_recall"),
            "mean_judge_correctness": avg(ans, "correctness"),
            "mean_judge_grounded": avg(ans, "grounded"),
            "mean_judge_context": avg(ans, "context_judge"),
        },
        "unanswerable": {"n": len(unans), "abstain_rate": avg(unans, "abstained")},
        "by_article": by_article,
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--colab", required=True, help="ngrok URL of the Colab server")
    ap.add_argument("--csv", default="evals/questions.csv")
    ap.add_argument("--out", default="eval_results.json")
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions")
    ap.add_argument("--no-judge", action="store_true", help="skip LLM judging (lexical scores only, much faster)")
    a = ap.parse_args()

    rows = load(a.csv)
    if a.limit:
        rows = rows[: a.limit]
    llm = LLM(a.colab)
    recs = []
    try:
        for i, row in enumerate(rows, 1):
            rec = evaluate(row, a.colab, llm, not a.no_judge)
            rec["input_order"] = i
            recs.append(rec)
            print(f"[{i}/{len(rows)}] {rec['overall']:.2f} {rec['stage']:<16} {rec['question'][:70]}", flush=True)
    except KeyboardInterrupt:
        print("interrupted: writing partial results")

    ranked = [{"rank": n, **r} for n, r in enumerate(sorted(recs, key=lambda r: (-r["overall"], r["input_order"])), 1)]
    report = {
        "meta": {
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "csv": a.csv, "questions": len(recs), "top_k": K, "judge": not a.no_judge,
            "weights": WEIGHTS, "hit_threshold": metrics.HIT, "evidence_threshold": metrics.EVID,
            "notes": [
                "overall (answerable) = weighted mean of judge correctness/grounded/context and lexical context_match/answer_recall; context_match = max(verbatim recall of gold context, share of gold-answer words found in the returned passages)",
                "overall (unanswerable) = 1.0 if the app said not found, else judge correctness",
                "stage = first failing step: retrieval_miss, selection_miss, answer_weak; ok otherwise",
                "judge is the same Colab model as the app: treat judge scores as noisy, lexical scores as reliable",
            ],
        },
        "summary": summarize(recs),
        "results": ranked,
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\nRANK SCORE STAGE            QUESTION")
    for r in ranked:
        print(f"{r['rank']:>4} {r['overall']:.2f}  {r['stage']:<16} {r['question'][:70]}")
    s = report["summary"]
    print(f"\nmean overall {s['mean_overall']} | stages {s['stages']} | answerable {s['answerable']['n']} "
          f"(retrieval hit {s['answerable']['retrieval_hit_rate']}, selection hit {s['answerable']['selection_hit_rate']}) "
          f"| unanswerable {s['unanswerable']['n']} (abstain {s['unanswerable']['abstain_rate']})")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()