"""LLM judge. Uses the same Colab model as the app. One criterion per call, 0-5 score + one-line reason."""
import re


def _score(llm, task, body, scale):
    prompt = (
        f"{task}\n\n{body}\n\nScale: {scale}\n\n"
        "Reply in exactly this format and nothing else:\n"
        "SCORE: <integer 0-5>\nREASON: <one short sentence>"
    )
    out = llm.generate(prompt)
    m = re.search(r"SCORE:\s*([0-5])\b", out)
    r = re.search(r"REASON:\s*(.*)", out, re.S)
    return {"score": int(m.group(1)) if m else None, "reason": (r.group(1) if r else out).strip()[:300]}


def correctness(llm, question, reference, candidate):
    return _score(
        llm,
        "You are grading an answer. Does the CANDIDATE answer state the same key facts as the REFERENCE answer? "
        "Judge facts only, not wording or length.",
        f"QUESTION: {question}\n\nREFERENCE ANSWER: {reference[:800]}\n\nCANDIDATE ANSWER: {candidate[:800]}",
        "5 = same key facts; 4 = same facts, minor omission; 3 = some key facts missing; "
        "2 = mostly different or vague; 1 = wrong or unrelated; 0 = empty",
    )


def grounded(llm, question, candidate, source):
    return _score(
        llm,
        "You are checking for made-up content. Is every claim in the CANDIDATE answer supported by the SOURCE TEXT?",
        f"QUESTION: {question}\n\nSOURCE TEXT: {source[:3500]}\n\nCANDIDATE ANSWER: {candidate[:800]}",
        "5 = every claim supported; 3 = some claims unsupported; 1 = mostly unsupported; 0 = empty",
    )


def context(llm, question, reference_paragraph, retrieved):
    return _score(
        llm,
        "Does the RETRIEVED TEXT contain the information from the REFERENCE PARAGRAPH that is needed to answer the question?",
        f"QUESTION: {question}\n\nREFERENCE PARAGRAPH: {reference_paragraph[:1200]}\n\nRETRIEVED TEXT: {retrieved[:3500]}",
        "5 = contains all the needed information; 3 = contains some of it; 1 = unrelated; 0 = empty",
    )
