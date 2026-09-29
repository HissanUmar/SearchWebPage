import re

MAX_TEXT, MAX_CODE = 1100, 400


def _fmt(c):
    s = f"[{c.id}] {c.path}\n{c.text[:MAX_TEXT]}"
    if c.code:
        s += f"\nCode:\n{c.code[0][:MAX_CODE]}"
    return s


def build_prompt(card, top, question):
    return (
        "Answer the question using ONLY the sections below, taken from one web page.\n"
        f"Page: {card.title}\nPurpose: {card.purpose}\n\n"
        + "\n\n".join(_fmt(c) for c in top)
        + f"\n\nQuestion: {question}\n\n"
        "Reply in exactly this format and nothing else:\n"
        "IDS: <section numbers that contain the answer, comma-separated, or NONE if they do not answer it>\n"
        "EXPLANATION: <1-3 sentences answering the question and noting anything important in those sections>"
    )


def parse(out, top):
    valid = {c.id for c in top}
    m = re.search(r"IDS:\s*(.*)", out)
    ids = [int(x) for x in re.findall(r"\d+", m.group(1)) if int(x) in valid] if m else []
    e = re.search(r"EXPLANATION:\s*(.*)", out, re.S)
    explanation = (e.group(1) if e else re.sub(r"IDS:.*", "", out)).strip()
    return list(dict.fromkeys(ids)), explanation


def ask(card, top, question, llm):
    """-> (chunk ids that answer the question, short explanation). Model never writes code/quotes."""
    return parse(llm.generate(build_prompt(card, top, question)), top)