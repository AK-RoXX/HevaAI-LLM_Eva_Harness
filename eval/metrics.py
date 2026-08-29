import math, re
from collections import Counter

STOP = set(
    "the a an is are was were to of in on for and or with from did does do what when where who how which it its this that company document provide provided information about approximately current".split()
)

def tokens(s):
    return [
        x
        for x in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?%?", s.lower())
        if x not in STOP
    ]

def answer_correct(row, result):
    if row["answerable"] != (not result.get("abstained", False)):
        return False
    if not row["answerable"]:
        return True
    expected = tokens(row["expected_answer"])
    actual = tokens(result.get("answer", ""))
    if not expected:
        return True
    hits = sum(1 for x in expected if x in actual)
    return hits / len(expected) >= 0.6


def evidence_support(row, result):
    """Local grounding signal: semantic TF-IDF is computed by runner; this function checks exact facts/numbers/entities from verified evidence keywords."""
    text = (
        result.get("answer", "")
        + " "
        + " ".join(c.get("text", "") for c in result.get("citations", []))
    ).lower()
    keys = row.get("evidence_keywords", [])
    if not keys:
        return 1.0
    return sum(1 for k in keys if k.lower() in text) / len(keys)


def ece(rows):
    bins = [[] for _ in range(10)]
    for r in rows:
        b = min(9, int(float(r.get("confidence", 0)) * 10))
        bins[b].append(r)
    total = len(rows)
    score = 0.0
    for b, items in enumerate(bins):
        if not items:
            continue
        acc = sum(x["correct"] for x in items) / len(items)
        conf = sum(x["confidence"] for x in items) / len(items)
        score += len(items) / total * abs(acc - conf)
    return score


def brier(rows):
    if not rows:
        return 0.0
    return sum((r["confidence"] - float(r["correct"])) ** 2 for r in rows) / len(rows)


def summarize(rows):
    n = len(rows)
    correct = sum(r["correct"] for r in rows)
    return {
        "n": n,
        "accuracy": correct / n if n else 0,
        "abstention_rate": sum(r["abstained"] for r in rows) / n if n else 0,
        "hallucination_rate": sum(r["hallucination"] for r in rows) / n if n else 0,
        "ece": ece(rows),
        "brier": brier(rows),
    }
