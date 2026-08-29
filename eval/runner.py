import argparse
import json
import re
from pathlib import Path

import httpx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]


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

    return bool(sim < 0.18 or invented_nums) and bool(
        missing or invented_nums
    )


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


def run(path, base, limit=None, upload_ref=True, use_judge=False):
    rows = load_jsonl(path)
    rows = rows[:limit] if limit else rows

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
                rr = client.post(
                    base + "/qa",
                    json={"question": row["question"]},
                )

                # --------------------------------------------------
                # Explicit model quota detection
                # --------------------------------------------------

                if rr.status_code >= 400:

                    if is_quota_error(rr):

                        error_message = (
                            f"MODEL_QUOTA_EXHAUSTED: "
                            f"HTTP {rr.status_code}: {rr.text}"
                        )

                        results.append(
                            {
                                **row,
                                "status": "error",
                                "error_type": "model_quota_exhausted",
                                "actual_answer": "",
                                "confidence": None,
                                "abstained": None,
                                "citations": [],
                                "correct": None,
                                "hallucination": None,
                                "semantic_support": None,
                                "error": error_message,
                            }
                        )

                        print(
                            f"\n[{i:03d}/{len(rows)}] {row['id']} "
                            "⚠ MODEL QUOTA EXHAUSTED",
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

                result = rr.json()

                correct = __import__(
                    "eval.metrics",
                    fromlist=["answer_correct"],
                ).answer_correct(row, result)

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

                results.append(
                    {
                        **row,
                        "status": "ok",
                        "actual_answer": result.get(
                            "answer",
                            "",
                        ),
                        "confidence": float(
                            result.get(
                                "confidence",
                                0,
                            )
                        ),
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
                        "correct": bool(correct),
                        "hallucination": bool(h),
                        "semantic_support": semantic_support(
                            result.get(
                                "answer",
                                "",
                            ),
                            result.get(
                                "citations",
                                [],
                            ),
                        ),
                        "grounding_judgment": judgment,
                    }
                )

                print(
                    f"[{i:03d}/{len(rows)}] "
                    f"{row['id']} "
                    f"{'PASS' if correct else 'FAIL'}",
                    flush=True,
                )

            except httpx.HTTPStatusError as e:

                error_message = str(e)

                results.append(
                    {
                        **row,
                        "status": "error",
                        "error_type": "http_error",
                        "actual_answer": "",
                        "confidence": None,
                        "abstained": None,
                        "citations": [],
                        "correct": None,
                        "hallucination": None,
                        "semantic_support": None,
                        "error": error_message,
                    }
                )

                print(
                    f"[{i:03d}/{len(rows)}] {row['id']} "
                    f"⚠ HTTP ERROR: {error_message}",
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
                        "abstained": None,
                        "citations": [],
                        "correct": None,
                        "hallucination": None,
                        "semantic_support": None,
                        "error": error_message,
                    }
                )

                print(
                    f"[{i:03d}/{len(rows)}] {row['id']} "
                    f"⚠ EVALUATION ERROR: {error_message}",
                    flush=True,
                )

    # --------------------------------------------------------------
    # Save results
    # --------------------------------------------------------------

    outdir = ROOT / "eval" / "results"
    outdir.mkdir(exist_ok=True)

    out = outdir / (
        Path(path).stem + "_results.jsonl"
    )

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

    # --------------------------------------------------------------
    # Evaluation summary
    # --------------------------------------------------------------

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
        print("⚠ MODEL QUOTA EXHAUSTED")
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
        default=str(
            ROOT
            / "dataset"
            / "ground_truth.jsonl"
        ),
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

    p.add_argument(
        "--no-upload",
        action="store_true",
    )

    p.add_argument(
        "--judge",
        action="store_true",
    )

    a = p.parse_args()

    results, out = run(
        a.dataset,
        a.base,
        a.limit,
        not a.no_upload,
        a.judge,
    )

    print()
    print(f"Saved {out}")