import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from eval.metrics import (
    answer_correct,
    answer_metrics,
    deterministic_grounding_score,
    evaluation_state,
    fact_aware_correct,
    retrieval_metrics,
)
from app.config import settings

ROOT = Path(__file__).resolve().parents[1]


def select_cases(rows, cases=None, start=0, limit=None):
    if cases:
        by_id = {row["id"]: row for row in rows}
        invalid = [case_id for case_id in cases if case_id not in by_id]
        if invalid:
            raise ValueError(
                "Invalid case ID(s): " + ", ".join(invalid)
                + ". Available IDs include: " + ", ".join(by_id)
            )
        return [by_id[case_id] for case_id in cases]
    if start < 0:
        raise ValueError("--start must be zero or greater")
    return rows[start:start + limit if limit is not None else None]


def model_slug(model):
    return re.sub(r"[^A-Za-z0-9]+", "_", model).strip("_").lower()


def result_path(path, provider, model):
    return ROOT / "eval" / "results" / f"{Path(path).stem}_{provider}_{model_slug(model)}.jsonl"


def load_jsonl(path):
    return [
        json.loads(x)
        for x in Path(path).read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def semantic_support(answer, citations):
    if not answer or not citations:
        return 0.0

    docs = [c.get("text", "") for c in citations]

    vec = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        stop_words="english",
    )

    m = vec.fit_transform([answer] + docs)

    return float(cosine_similarity(m[0:1], m[1:]).max())


def hallucination(row, result):
    if result.get("abstained"):
        return False

    sim = semantic_support(
        result.get("answer", ""),
        result.get("citations", []),
    )

    keys = row.get("evidence_keywords", [])

    ans = result.get("answer", "").lower()

    citation_text = " ".join(
        c.get("text", "")
        for c in result.get("citations", [])
    ).lower()

    missing = [
        k
        for k in keys
        if k.lower() not in (ans + " " + citation_text)
    ]

    nums = set(
        re.findall(
            r"\$?\d+(?:\.\d+)?%?",
            ans,
        )
    )

    source_nums = set(
        re.findall(
            r"\$?\d+(?:\.\d+)?%?",
            citation_text,
        )
    )

    invented_nums = nums - source_nums

    # Lightweight named-entity proxy: capitalized names/organizations in the
    # answer should also occur in the cited evidence. This is intentionally
    # heuristic and complements, rather than replaces, the keyword checks.
    entity_pattern = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b")
    answer_entities = {
        entity.lower()
        for entity in entity_pattern.findall(str(result.get("answer", "")))
        if entity.lower() not in {"the", "this", "when", "what", "where", "which", "yes", "no"}
    }
    source_entities = {
        entity.lower() for entity in entity_pattern.findall(" ".join(c.get("text", "") for c in result.get("citations", [])))
    }
    unsupported_entities = answer_entities - source_entities

    return bool(sim < 0.18 or invented_nums or unsupported_entities) and bool(
        missing or invented_nums or unsupported_entities
    )


def heva_confidence(result, support, hallucinated):
    """Conservative deterministic confidence from observable runtime signals.

    Formula: model confidence × retrieval factor × lexical-support factor ×
    grounding factor. Retrieval and support are capped observable signals;
    hallucination sets grounding to zero. Abstentions have zero answer
    confidence. This is intentionally separate from the model's confidence.
    """
    if result.get("abstained", False):
        return 0.0, ["abstention"]
    citations = result.get("citations", [])
    top_relevance = max((float(c.get("relevance", 0)) for c in citations), default=0.0)
    retrieval_signal = min(1.0, top_relevance / 0.30)
    support_signal = min(1.0, max(0.0, float(support)))
    grounding_signal = 0.0 if hallucinated else 1.0
    model_signal = min(1.0, max(0.0, float(result.get("confidence", 0))))
    score = model_signal * (0.4 + 0.6 * retrieval_signal) * (0.4 + 0.6 * support_signal) * grounding_signal
    signals = ["model_confidence", "retrieval_relevance", "lexical_support", "local_grounding"]
    if not citations:
        signals.remove("retrieval_relevance")
    return min(1.0, max(0.0, score)), signals

def upload(client, base, path):
    with open(path, "rb") as f:
        r = client.post(
            base + "/documents",
            files={
                "file": (
                    Path(path).name,
                    f,
                    "text/markdown",
                )
            },
        )

    r.raise_for_status()
    return r.json()


def is_quota_error(response):
    """
    Detect model quota/resource exhaustion even when the
    application incorrectly exposes it as HTTP 500.
    """

    body = response.text.lower()

    quota_indicators = [
        "quota",
        "resource exhausted",
        "resource_exhausted",
        "rate limit",
        "rate_limit",
        "too many requests",
        "exceeded",
        "429",
        "billing",
        "limit reached",
        "model quota",
    ]

    return any(indicator in body for indicator in quota_indicators)


def run(path=None, base="http://127.0.0.1:8000", limit=None, upload_ref=True,
        use_judge=False, provider="ollama", model=None, cases=None,
        start=0, adversarial=False):
    if path is None:
        path = ROOT / "dataset" / ("adversarial.jsonl" if adversarial else "ground_truth.jsonl")
    provider = (provider or "ollama").lower()
    if provider not in {"ollama", "gemini"}:
        raise ValueError(f"Unsupported provider: {provider}. Choose ollama or gemini.")
    model = model or (settings.ollama_model if provider == "ollama" else settings.gemini_model)
    rows = load_jsonl(path)
    rows = select_cases(rows, cases=cases, start=start, limit=limit)
    timestamp = datetime.now(timezone.utc).isoformat()

    results = []
    judge = None
    quota_exhausted = False

    if use_judge:
        from eval.judge import GeminiGroundingJudge

        judge = GeminiGroundingJudge()

    with httpx.Client(timeout=120) as client:

        if upload_ref:
            upload(
                client,
                base,
                ROOT / "data" / "eval_reference.md",
            )

        for i, row in enumerate(rows, 1):

            try:
                started = time.perf_counter()
                rr = client.post(
                    base + "/qa/eval",
                    json={"question": row["question"], "provider": provider, "model": model},
                )
                latency_ms = (time.perf_counter() - started) * 1000

                # Explicit model quota detection

                if rr.status_code >= 400:

                    if is_quota_error(rr):

                        error_message = (
                            f"MODEL_QUOTA_EXHAUSTED: "
                            f"HTTP {rr.status_code}: {rr.text}"
                        )

                        results.append(
                            {
                                **row,
                                "provider": provider,
                                "model": model,
                                "timestamp": timestamp,
                                "status": "error",
                                "error_type": "model_quota_exhausted",
                                "actual_answer": "",
                                "confidence": None,
                                "model_confidence": None,
                                "heva_confidence": None,
                                "confidence_signals": [],
                                "abstained": None,
                                "citations": [],
                                "correct": None,
                                "hallucination": None,
                                "semantic_support": None,
                                "metrics": {
                                    "exact_match": None,
                                    "precision": None,
                                    "recall": None,
                                    "f1": None,
                                    "bleu": None,
                                    "rouge1": None,
                                    "rouge2": None,
                                    "rougeL": None,
                                    "semantic_similarity": None,
                                    "latency_ms": latency_ms,
                                },
                                "latency_ms": latency_ms,
                                "error": error_message,
                            }
                        )

                        print(
                            f"\n[{i:03d}/{len(rows)}] {row['id']} "
                            "[WARN] MODEL QUOTA EXHAUSTED",
                            flush=True,
                        )

                        print(
                            "Evaluation stopped because the model quota "
                            "or resource limit has been exhausted.",
                            flush=True,
                        )

                        quota_exhausted = True
                        break

                    # Non-quota HTTP error
                    rr.raise_for_status()

                payload = rr.json()
                result = payload.get("response", payload)
                trace = payload.get("retrieval_trace", {})
                retrieval = retrieval_metrics(row, trace)

                metric_values = (
                    answer_metrics(
                        row["expected_answer"],
                        result.get("answer", ""),
                    )
                    if row["answerable"]
                    else {
                        "exact_match": None,
                        "precision": None,
                        "recall": None,
                        "f1": None,
                        "bleu": None,
                        "rouge1": None,
                        "rouge2": None,
                        "rougeL": None,
                        "semantic_similarity": None,
                    }
                )

                h = hallucination(row, result)

                judgment = None

                if judge and not result.get("abstained"):

                    try:
                        judgment = judge.judge(
                            row["question"],
                            result.get("answer", ""),
                            result.get("citations", []),
                        )

                        h = not judgment["supported"]

                    except Exception as je:
                        judgment = {
                            "error": str(je)
                        }

                support = semantic_support(
                    result.get("answer", ""),
                    result.get("citations", []),
                )
                observed = {
                    **result,
                    "abstained": bool(result.get("abstained", False)),
                    "hallucination": bool(h),
                }
                state = evaluation_state(row, observed)
                strict_correct = answer_correct(row, observed)
                fact_correct = fact_aware_correct(row, observed)
                heva_score, confidence_signals = heva_confidence(
                    observed,
                    support,
                    bool(h),
                )
                model_confidence = float(result.get("confidence", 0))
                grounding_score = deterministic_grounding_score(
                    row, result, bool(h), support
                )
                judge_available = bool(judgment and "error" not in judgment)

                results.append(
                    {
                        **row,
                        "provider": provider,
                        "model": model,
                        "timestamp": timestamp,
                        "status": "ok",
                        "actual_answer": result.get(
                            "answer",
                            "",
                        ),
                        "confidence": model_confidence,
                        "model_confidence": model_confidence,
                        "heva_confidence": heva_score,
                        "confidence_signals": confidence_signals,
                        "abstained": bool(
                            result.get(
                                "abstained",
                                False,
                            )
                        ),
                        "citations": result.get(
                            "citations",
                            [],
                        ),
                        "correct": bool(fact_correct),
                        "strict_correct": bool(strict_correct),
                        "fact_aware_correct": bool(fact_correct),
                        "evaluation_state": state,
                        "abstention_correct": row["answerable"] == (not result.get("abstained", False)),
                        "hallucination": bool(h),
                        "semantic_support": support,
                        "metrics": {
                            **metric_values,
                            "latency_ms": latency_ms,
                        },
                        "latency_ms": latency_ms,
                        "grounding_judgment": judgment,
                        "judge_available": judge_available,
                        "judge_grounded": judgment.get("supported") if judge_available else None,
                        "judge_score": judgment.get("score") if judge_available else None,
                        "retrieval_trace": trace,
                        "retrieval_metrics": retrieval,
                        "deterministic_grounding_score": grounding_score,
                    }
                )

                print(
                    f"[{i:03d}/{len(rows)}] "
                    f"{row['id']} "
                    f"{'PASS' if fact_correct else 'FAIL'}",
                    flush=True,
                )

            except httpx.HTTPStatusError as e:

                error_message = f"{e}; response={e.response.text}"

                results.append(
                    {
                        **row,
                        "provider": provider,
                        "model": model,
                        "timestamp": timestamp,
                        "status": "error",
                        "error_type": "http_error",
                        "actual_answer": "",
                        "confidence": None,
                        "model_confidence": None,
                        "heva_confidence": None,
                        "confidence_signals": [],
                        "abstained": None,
                        "citations": [],
                        "correct": None,
                        "hallucination": None,
                        "semantic_support": None,
                        "metrics": {
                            "exact_match": None,
                            "precision": None,
                            "recall": None,
                            "f1": None,
                            "bleu": None,
                            "rouge1": None,
                            "rouge2": None,
                            "rougeL": None,
                            "semantic_similarity": None,
                            "latency_ms": None,
                        },
                        "latency_ms": None,
                        "error": error_message,
                    }
                )

                print(
                    f"[{i:03d}/{len(rows)}] {row['id']} "
                    f"[WARN] HTTP ERROR: {error_message}",
                    flush=True,
                )

            except Exception as e:

                error_message = str(e)

                results.append(
                    {
                        **row,
                        "status": "error",
                        "error_type": "evaluation_error",
                        "actual_answer": "",
                        "confidence": None,
                        "model_confidence": None,
                        "heva_confidence": None,
                        "confidence_signals": [],
                        "abstained": None,
                        "citations": [],
                        "correct": None,
                        "hallucination": None,
                        "semantic_support": None,
                        "metrics": {
                            "exact_match": None,
                            "precision": None,
                            "recall": None,
                            "f1": None,
                            "bleu": None,
                            "rouge1": None,
                            "rouge2": None,
                            "rougeL": None,
                            "semantic_similarity": None,
                            "latency_ms": None,
                        },
                        "latency_ms": None,
                        "error": error_message,
                    }
                )

                print(
                    f"[{i:03d}/{len(rows)}] {row['id']} "
                    f"[WARN] EVALUATION ERROR: {error_message}",
                    flush=True,
                )

    # Save results

    outdir = ROOT / "eval" / "results"
    outdir.mkdir(exist_ok=True)

    out = result_path(path, provider, model)

    out.write_text(
        "\n".join(
            json.dumps(
                x,
                ensure_ascii=False,
            )
            for x in results
        )
        + "\n",
        encoding="utf-8",
    )

    # Evaluation summary

    evaluated = [
        r
        for r in results
        if r.get("status") == "ok"
    ]

    errors = [
        r
        for r in results
        if r.get("status") == "error"
    ]

    print()
    print("=" * 60)

    if quota_exhausted:
        print("EVALUATION INCOMPLETE")
        print("[WARN] MODEL QUOTA EXHAUSTED")
    else:
        print("EVALUATION COMPLETE")

    print("=" * 60)

    print(f"Requested cases : {len(rows)}")
    print(f"Evaluated cases : {len(evaluated)}")
    print(f"Error cases     : {len(errors)}")

    if quota_exhausted:
        remaining = len(rows) - len(results)
        print(f"Skipped cases   : {remaining}")
        print(
            "Reason          : MODEL QUOTA EXHAUSTED"
        )

    print("=" * 60)

    return results, out


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument(
        "dataset",
        default=None,
        nargs="?",
    )

    p.add_argument(
        "--base",
        default="http://127.0.0.1:8000",
    )

    p.add_argument(
        "--limit",
        type=int,
    )

    p.add_argument("--start", type=int, default=0)
    p.add_argument("--cases", help="Comma-separated case IDs; takes precedence over --limit")
    p.add_argument("--adversarial", action="store_true")
    p.add_argument("--provider", choices=["ollama", "gemini"], default="ollama")
    p.add_argument("--model")

    p.add_argument(
        "--no-upload",
        action="store_true",
    )

    p.add_argument(
        "--judge",
        action="store_true",
    )

    a = p.parse_args()

    if a.dataset is None:
        a.dataset = str(ROOT / "dataset" / ("adversarial.jsonl" if a.adversarial else "ground_truth.jsonl"))
    case_ids = [x.strip() for x in a.cases.split(",") if x.strip()] if a.cases else None
    try:
        results, out = run(
            a.dataset, a.base, a.limit, not a.no_upload, a.judge,
            a.provider, a.model, case_ids, a.start, a.adversarial,
        )
    except ValueError as exc:
        p.error(str(exc))

    print()
    print(f"Saved {out}")
