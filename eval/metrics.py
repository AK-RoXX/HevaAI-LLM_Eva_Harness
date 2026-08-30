import re
import string
from collections import Counter

import sacrebleu
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


_PUNCT_TRANSLATION = str.maketrans({
    "–": "-",
    "—": "-",
    "‑": "-",
    "’": "'",
    "“": '"',
    "”": '"',
})

_NUMBER_PATTERN = re.compile(r"(?<![a-z])\$?\d+(?:[.,]\d+)*(?:%|[mk])?(?![a-z])", re.I)
_NEGATIVE_PATTERN = re.compile(
    r"\b(?:no|not|never|does not|do not|did not|cannot|can't|insufficient|"
    r"unknown|unstated|not stated|not mentioned|not provided|not identified|"
    r"doesn't|don't|didn't)\b",
    re.I,
)
_FACT_QUALIFIERS = {"approximately", "approx", "about", "around"}


def _canonical_numeric(token):
    value = token.lower().replace(",", "")
    suffix = ""
    if value.endswith("%"):
        suffix, value = "%", value[:-1]
    elif value.endswith(("m", "k")):
        suffix, value = value[-1], value[:-1]
    if value.startswith("$"):
        value = value[1:]
    try:
        number = float(value)
    except ValueError:
        return token.lower()
    return number, suffix


def _numeric_values(value):
    text = re.sub(r"(\$?\d+(?:[.,]\d+)*)\s*(million|m)\b", r"\1m", str(value or ""), flags=re.I)
    text = re.sub(r"(\$?\d+(?:[.,]\d+)*)\s*(thousand|k)\b", r"\1k", text, flags=re.I)
    return [_canonical_numeric(match) for match in _NUMBER_PATTERN.findall(text)]


def normalize_text(value):
    """Normalize factual QA text without discarding meaningful numbers."""
    text = str(value or "").lower().translate(_PUNCT_TRANSLATION)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def tokens(value):
    """Safely tokenize words, decimals, comma-separated numbers, and percentages."""
    text = str(value or "").lower().translate(_PUNCT_TRANSLATION)
    text = re.sub(r"(\$?\d+(?:[.,]\d+)*)\s*million\b", r"\1m", text)
    text = re.sub(r"(\$?\d+(?:[.,]\d+)*)\s*thousand\b", r"\1k", text)
    return re.findall(r"\d+(?:[.,]\d+)*%?|[a-z]+(?:'[a-z]+)?", text)


def exact_match(reference, prediction):
    return float(normalize_text(reference) == normalize_text(prediction))


def token_precision(reference, prediction):
    expected = Counter(tokens(reference))
    actual = Counter(tokens(prediction))
    predicted_count = sum(actual.values())
    overlap = sum((expected & actual).values())
    return overlap / predicted_count if predicted_count else float(not expected)


def token_recall(reference, prediction):
    expected = Counter(tokens(reference))
    actual = Counter(tokens(prediction))
    expected_count = sum(expected.values())
    overlap = sum((expected & actual).values())
    return overlap / expected_count if expected_count else float(not actual)


def token_f1(reference, prediction):
    precision = token_precision(reference, prediction)
    recall = token_recall(reference, prediction)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def bleu(reference, prediction):
    """Return normalized SacreBLEU, with safe handling for short/empty answers."""
    reference_tokens = tokens(reference)
    prediction_tokens = tokens(prediction)
    if not reference_tokens and not prediction_tokens:
        return 1.0
    if not reference_tokens or not prediction_tokens:
        return 0.0
    score = sacrebleu.sentence_bleu(
        " ".join(prediction_tokens),
        [" ".join(reference_tokens)],
        smooth_method="exp",
        tokenize="none",
    )
    return min(1.0, max(0.0, float(score.score / 100.0)))


def rouge_scores(reference, prediction):
    """Return ROUGE F1 scores for factual answer comparison."""
    if normalize_text(reference) == normalize_text(prediction):
        return {"rouge1": 1.0, "rouge2": 1.0, "rougeL": 1.0}
    if not tokens(reference) and not tokens(prediction):
        return {"rouge1": 1.0, "rouge2": 1.0, "rougeL": 1.0}
    if not tokens(reference) or not tokens(prediction):
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=False
    )
    scores = scorer.score(normalize_text(reference), normalize_text(prediction))
    return {name: float(scores[name].fmeasure) for name in ("rouge1", "rouge2", "rougeL")}


def semantic_similarity(reference, prediction):
    """Deterministic lexical semantic similarity using TF-IDF cosine similarity."""
    reference_text = normalize_text(reference)
    prediction_text = normalize_text(prediction)
    if not reference_text and not prediction_text:
        return 1.0
    if not reference_text or not prediction_text:
        return 0.0
    try:
        matrix = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(
            [reference_text, prediction_text]
        )
        return min(1.0, max(0.0, float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])))
    except ValueError:
        return 0.0


def answer_metrics(reference, prediction):
    rouge = rouge_scores(reference, prediction)
    return {
        "exact_match": exact_match(reference, prediction),
        "precision": token_precision(reference, prediction),
        "recall": token_recall(reference, prediction),
        "f1": token_f1(reference, prediction),
        "bleu": bleu(reference, prediction),
        **rouge,
        "semantic_similarity": semantic_similarity(reference, prediction),
    }


def fact_aware_match(reference, prediction):
    """Match facts using normalized tokens and numeric values, not raw substrings.

    A prediction may contain a supported explanatory frame around a short answer,
    but it must contain every reference token (with multiplicity) and every
    reference number. Numeric values may differ only within a small rounding
    tolerance; materially different years, amounts, or percentages do not match.
    """
    reference_tokens = Counter(token for token in tokens(reference) if token not in _FACT_QUALIFIERS)
    prediction_tokens = Counter(token for token in tokens(prediction) if token not in _FACT_QUALIFIERS)
    if not reference_tokens:
        return not prediction_tokens
    if any(
        prediction_tokens[token] < count
        for token, count in reference_tokens.items()
        if not any(character.isdigit() for character in token)
    ):
        return False

    reference_numbers = _numeric_values(reference)
    prediction_numbers = _numeric_values(prediction)
    if len(prediction_numbers) < len(reference_numbers):
        return False
    for expected in reference_numbers:
        if isinstance(expected, tuple):
            expected_value, expected_suffix = expected
            candidates = [
                actual_value
                for actual in prediction_numbers
                if isinstance(actual, tuple)
                and actual[1] == expected_suffix
                and abs(actual[0] - expected_value)
                <= (
                    0.0
                    if expected_value.is_integer() and not expected_suffix
                    else max(0.01, abs(expected_value) * 0.001)
                )
                for actual_value in [actual[0]]
            ]
            if not candidates:
                return False
        elif expected not in prediction_numbers:
            return False
    return True


def fact_aware_correct(row, result):
    """Correctness that allows supported explanatory wording for answerable QA."""
    if not row.get("answerable"):
        return bool(result.get("abstained", False)) or is_grounded_negative(row, result)
    if result.get("abstained", False):
        return False
    if result.get("hallucination", False):
        return False
    return fact_aware_match(row.get("expected_answer", ""), result.get("answer", ""))


def is_grounded_negative(row, result):
    """Identify a negative answer grounded in citations and verified keywords."""
    if row.get("answerable") or result.get("abstained", False):
        return False
    answer = str(result.get("answer", ""))
    citations = result.get("citations", [])
    combined = (answer + " " + " ".join(c.get("text", "") for c in citations)).lower()
    keywords = row.get("evidence_keywords", [])
    return bool(citations) and bool(_NEGATIVE_PATTERN.search(answer)) and all(
        str(keyword).lower() in combined for keyword in keywords
    )


def evaluation_state(row, result):
    """Return one of answered, abstained, grounded_negative, or unsupported."""
    if result.get("abstained", False):
        return "abstained"
    if not row.get("answerable"):
        return "grounded_negative" if is_grounded_negative(row, result) else "unsupported"
    if result.get("hallucination", False):
        return "unsupported"
    return "answered"

def answer_correct(row, result):
    if row["answerable"] != (not result.get("abstained", False)):
        return False
    if not row["answerable"]:
        return True
    return bool(answer_metrics(row["expected_answer"], result.get("answer", ""))["exact_match"])


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


def _trace_chunks(trace):
    """Normalize an evaluation trace; malformed traces are treated as empty."""
    if not isinstance(trace, dict):
        return []
    chunks = trace.get("retrieved_chunks", [])
    if not isinstance(chunks, list):
        return []
    normalized = []
    for item in chunks:
        if not isinstance(item, dict):
            continue
        try:
            normalized.append({
                "rank": int(item.get("rank", len(normalized) + 1)),
                "score": float(item.get("score", 0.0)),
                "text": str(item.get("text", "")),
                "chunk_id": str(item.get("chunk_id", "")),
                "document_id": str(item.get("document_id", "")),
            })
        except (TypeError, ValueError):
            continue
    return sorted(normalized, key=lambda item: item["rank"])


def retrieval_metrics(row, trace):
    """Keyword-based retrieval metrics, not gold chunk annotations."""
    keywords = [str(k).lower() for k in row.get("evidence_keywords", []) if str(k).strip()]
    chunks = _trace_chunks(trace)
    if not keywords:
        return {
            "hit_at_1": None, "hit_at_3": None, "hit_at_5": None, "mrr": None,
            "context_precision": None, "context_recall": None,
            "top1_score": chunks[0]["score"] if chunks else None,
            "top5_score": (sum(c["score"] for c in chunks[:5]) / min(5, len(chunks))) if chunks else None,
            "retrieval_abstained": bool(trace.get("retrieval_abstained", False)) if isinstance(trace, dict) else False,
        }

    def covered(items):
        text = " ".join(item["text"] for item in items).lower()
        return {keyword for keyword in keywords if keyword in text}

    first_full_rank = None
    for index in range(1, min(5, len(chunks)) + 1):
        if len(covered(chunks[:index])) == len(set(keywords)):
            first_full_rank = index
            break
    all_covered = covered(chunks)
    precision_hits = sum(bool(covered([chunk])) for chunk in chunks[:5])
    return {
        "hit_at_1": float(first_full_rank is not None and first_full_rank <= 1),
        "hit_at_3": float(first_full_rank is not None and first_full_rank <= 3),
        "hit_at_5": float(first_full_rank is not None and first_full_rank <= 5),
        "mrr": (1.0 / first_full_rank) if first_full_rank else 0.0,
        "context_precision": precision_hits / min(5, len(chunks)) if chunks else 0.0,
        "context_recall": len(all_covered) / len(set(keywords)),
        "top1_score": chunks[0]["score"] if chunks else None,
        "top5_score": (sum(c["score"] for c in chunks[:5]) / min(5, len(chunks))) if chunks else None,
        "retrieval_abstained": bool(trace.get("retrieval_abstained", False)) if isinstance(trace, dict) else False,
    }


def deterministic_grounding_score(row, result, hallucinated=False, semantic_support_value=None):
    """Transparent support heuristic; this is not claim-level faithfulness."""
    answer = str(result.get("answer", ""))
    citations = result.get("citations", []) or []
    citation_text = " ".join(str(c.get("text", "")) for c in citations if isinstance(c, dict))
    support = float(semantic_support_value if semantic_support_value is not None else 0.0)
    keywords = [str(k).lower() for k in row.get("evidence_keywords", []) if str(k).strip()]
    combined = (answer + " " + citation_text).lower()
    keyword_coverage = (sum(keyword in combined for keyword in keywords) / len(keywords)) if keywords else None
    answer_numbers = set(re.findall(r"\$?\d+(?:[.,]\d+)?%?", answer))
    citation_numbers = set(re.findall(r"\$?\d+(?:[.,]\d+)?%?", citation_text))
    unsupported_numeric = len(answer_numbers - citation_numbers)
    numeric_support = 1.0 if not unsupported_numeric else 0.0
    if hallucinated:
        return 0.0
    coverage = 1.0 if keyword_coverage is None else keyword_coverage
    return max(0.0, min(1.0, 0.5 * support + 0.3 * coverage + 0.2 * numeric_support))


def ece(rows):
    bins = [[] for _ in range(10)]
    for r in rows:
        b = min(9, int(float(r.get("heva_confidence", r.get("confidence", 0)) or 0) * 10))
        bins[b].append(r)
    total = len(rows)
    score = 0.0
    for b, items in enumerate(bins):
        if not items:
            continue
        acc = sum(bool(x.get("fact_aware_correct", x.get("correct", False))) for x in items) / len(items)
        conf = sum(float(x.get("heva_confidence", x.get("confidence", 0)) or 0) for x in items) / len(items)
        score += len(items) / total * abs(acc - conf)
    return score


def brier(rows):
    if not rows:
        return 0.0
    return sum(
        (float(r.get("heva_confidence", r.get("confidence", 0)) or 0)
         - float(bool(r.get("fact_aware_correct", r.get("correct", False))))) ** 2
        for r in rows
    ) / len(rows)


def summarize(rows):
    n = len(rows)
    strict_correct = sum(bool(r.get("strict_correct", r.get("correct", False))) for r in rows)
    fact_correct = sum(bool(r.get("fact_aware_correct", r.get("correct", False))) for r in rows)
    unanswerable = [r for r in rows if not r.get("answerable")]
    abstained = [r for r in rows if r.get("evaluation_state") == "abstained" or r.get("abstained")]
    expected_abstentions = len(unanswerable)
    true_abstentions = sum(r in abstained for r in unanswerable)
    abstention_precision = true_abstentions / len(abstained) if abstained else (1.0 if not expected_abstentions else 0.0)
    abstention_recall = true_abstentions / expected_abstentions if expected_abstentions else 1.0
    abstention_f1 = (
        2 * abstention_precision * abstention_recall / (abstention_precision + abstention_recall)
        if abstention_precision + abstention_recall else 0.0
    )
    return {
        "n": n,
        "accuracy": strict_correct / n if n else 0,
        "strict_accuracy": strict_correct / n if n else 0,
        "fact_aware_accuracy": fact_correct / n if n else 0,
        "abstention_rate": len(abstained) / n if n else 0,
        "abstention_accuracy": sum(r.get("answerable") == (not r.get("abstained", False)) for r in rows) / n if n else 0,
        "abstention_precision": abstention_precision,
        "abstention_recall": abstention_recall,
        "abstention_f1": abstention_f1,
        "false_answer_rate": sum(r.get("evaluation_state") == "unsupported" for r in unanswerable) / expected_abstentions if expected_abstentions else 0.0,
        "grounded_negative_accuracy": sum(r.get("evaluation_state") == "grounded_negative" for r in unanswerable) / expected_abstentions if expected_abstentions else 0.0,
        "hallucination_rate": sum(bool(r.get("hallucination")) for r in rows) / n if n else 0,
        "ece": ece(rows),
        "brier": brier(rows),
    }
