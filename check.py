"""Show how a page is chunked, and optionally how a question is ranked and prompted.

  python check.py URL                      overview table of every passage
  python check.py URL --full               full text of every passage
  python check.py URL --ids 8,15           full text of chosen passages only
  python check.py URL --find WORD          passages containing WORD
  python check.py URL --dump out.txt       write the whole formatted document to a file
  python check.py URL --q "question" --colab https://xxxx.ngrok-free.app
                                           ranking scores, exact prompt, raw model reply
"""
import argparse
import sys

import numpy as np

from pageqa.card import card_parts
from pageqa.fetch import fetch
from pageqa.split import split


def body(c):
    out = c.text
    for k in c.code:
        out += f"\n[code]\n{k}\n[/code]"
    return out


def full(c):
    head = f"--- passage {c.id} | {len(c.text)} chars | {len(c.code)} code | anchor: {c.anchor or '-'}"
    return f"{head}\npath: {c.path}\n{body(c)}\n"


def row(c):
    prev = " ".join(c.text.split())[:90] or "(code only)"
    return f"{c.id:>3} {len(c.text):>5} {len(c.code):>4}  {(c.anchor or '-')[:20]:<20} {c.path[:55]}\n      > {prev}"


sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("url")
ap.add_argument("--full", action="store_true")
ap.add_argument("--ids")
ap.add_argument("--find")
ap.add_argument("--dump")
ap.add_argument("--q")
ap.add_argument("--colab")
a = ap.parse_args()

title, ch = split(fetch(a.url))
if not ch:
    raise SystemExit("0 passages: no readable content (JS-rendered page?)")

if a.dump:
    with open(a.dump, "w", encoding="utf-8") as f:
        f.write("\n".join(full(c) for c in ch))
    print(f"wrote {len(ch)} passages to {a.dump}")

sizes = [len(c.text) for c in ch]
print(f"PAGE: {title}")
print(f"passages: {len(ch)} | chars min/avg/max: {min(sizes)}/{sum(sizes) // len(sizes)}/{max(sizes)}"
      f" | with anchor: {sum(bool(c.anchor) for c in ch)} | with code: {sum(bool(c.code) for c in ch)}"
      f" | distinct sections: {len({c.path for c in ch})}")
h1, intro, outline = card_parts(title, ch)
print(f"card inputs -> h1: {h1!r} | intro: {intro[:80]!r} | outline entries: {len(outline)}\n")

if a.ids:
    for i in [int(x) for x in a.ids.split(",")]:
        print(full(ch[i]))
elif a.find:
    hits = [c for c in ch if a.find.lower() in body(c).lower()]
    print(f"{len(hits)} passage(s) contain {a.find!r}\n")
    for c in hits:
        print(full(c))
elif a.full:
    print("\n".join(full(c) for c in ch))
elif not a.q:
    print("ID  CHARS CODE  ANCHOR               PATH")
    print("\n".join(row(c) for c in ch))

if a.q:
    if not a.colab:
        ap.error("--q needs --colab")
    from pageqa.ask import build_prompt, parse
    from pageqa.card import page_card
    from pageqa.llm import LLM
    from pageqa.rank import K, embed_chunks, score

    llm = LLM(a.colab)
    card = page_card(title, ch, llm)
    sc = score(a.q, ch, embed_chunks(ch, llm), llm)
    order = list(np.argsort(-sc))
    print(f"PURPOSE (from card): {card.purpose}\n")
    print(f"RANKING (* = sent to the model, top {K})")
    for n, i in enumerate(order[:8]):
        print(f"{'*' if n < K else ' '} passage {ch[i].id:>3} score {sc[i]:.3f} | {ch[i].path[:50]} | {' '.join(ch[i].text.split())[:60]}")
    top = [ch[i] for i in order[:K]]
    prompt = build_prompt(card, top, a.q)
    print(f"\n===== EXACT PROMPT ({len(prompt)} chars) =====\n{prompt}")
    raw = llm.generate(prompt)
    ids, expl = parse(raw, top)
    print(f"\n===== RAW MODEL REPLY =====\n{raw}\n\n===== PARSED =====\nids: {ids}\nexplanation: {expl}")