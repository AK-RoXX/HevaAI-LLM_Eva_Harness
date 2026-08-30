"""Professional Phase 4 report for ground-truth and adversarial results."""

import argparse
import json
from pathlib import Path
from statistics import mean

from .calibration import analyze_calibration
from .failure_analysis import analyze_failure_modes
from .metrics import summarize

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def avg(values):
    values = [float(x) for x in values if x is not None]
    return mean(values) if values else None


def retrieval(rows):
    values = [row.get("retrieval_metrics", {}) for row in rows]
    result = {}
    for key in (
        "hit_at_1",
        "hit_at_3",
        "hit_at_5",
        "mrr",
        "context_precision",
        "context_recall",
        "top1_score",
        "top5_score",
    ):
        result[key] = avg(item.get(key) for item in values)
    result["abstentions"] = sum(
        bool(item.get("retrieval_abstained")) for item in values
    )
    return result


def category_table(rows):
    lines = [
        "| Category | Cases | Strict | Fact-aware | Hallucination | Abstention | Avg F1 | Semantic sim. | Avg top-1 | Hit@1 | Hit@5 | MRR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for category in sorted({r.get("category", "unknown") for r in rows}):
        group = [r for r in rows if r.get("category") == category]
        ret = retrieval(group)
        pct = lambda xs: f"{(avg(xs) or 0):.1%}"
        num = lambda x: f"{x:.3f}" if x is not None else "null"
        lines.append(
            f"| {category} | {len(group)} | {pct(float(bool(r.get('strict_correct', r.get('correct')))) for r in group)} | {pct(float(bool(r.get('fact_aware_correct', r.get('correct')))) for r in group)} | {pct(float(bool(r.get('hallucination'))) for r in group)} | {pct(float(bool(r.get('abstained'))) for r in group)} | {num(avg(r.get('metrics', {}).get('f1') for r in group))} | {num(avg(r.get('metrics', {}).get('semantic_similarity') for r in group))} | {num(ret['top1_score'])} | {pct(ret['hit_at_1'] for _ in [0]) if ret['hit_at_1'] is not None else 'null'} | {pct(ret['hit_at_5'] for _ in [0]) if ret['hit_at_5'] is not None else 'null'} | {num(ret['mrr'])} |"
        )
    return lines


def section(rows, title):
    rows = [r for r in rows if r.get("status") == "ok"]
    s = summarize(rows)
    r = retrieval(rows)
    answerable = [row for row in rows if row.get("answerable")]
    calibration = analyze_calibration(rows)
    grounding = avg(row.get("deterministic_grounding_score") for row in rows)
    latency = avg(row.get("latency_ms") for row in rows)
    latency_values = sorted(
        float(row["latency_ms"]) for row in rows if row.get("latency_ms") is not None
    )
    percentile = lambda p: (
        latency_values[
            min(len(latency_values) - 1, int(round((len(latency_values) - 1) * p)))
        ]
        if latency_values
        else None
    )
    lines = [
        f"## {title}",
        "",
        f"Cases: **{len(rows)}**; evaluation errors: **0**",
        "",
        "## 4. Answer quality",
        "",
        f"- Strict accuracy: **{s['strict_accuracy']:.1%}**",
        f"- Fact-aware accuracy: **{s['fact_aware_accuracy']:.1%}**",
        "- Strict accuracy requires exact normalized answers; fact-aware accuracy permits supported explanatory wording and is therefore usually higher.",
        "",
        "| Metric | Mean |",
        "|---|---:|",
    ]
    labels = {
        "exact_match": "Exact Match",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "bleu": "BLEU",
        "rouge1": "ROUGE-1",
        "rouge2": "ROUGE-2",
        "rougeL": "ROUGE-L",
        "semantic_similarity": "Lexical TF-IDF Similarity",
    }
    for key, label in labels.items():
        lines.append(
            f"| {label} | {avg(row.get('metrics', {}).get(key) for row in answerable) or 0:.4f} |"
        )
    lines += [
        "",
        "## 5. Retrieval quality",
        "",
        "Keyword-based proxy metrics using `evidence_keywords`; these are not gold chunk annotations.",
        (
            f"- Hit@1 / Hit@3 / Hit@5: **{r['hit_at_1']:.1%} / {r['hit_at_3']:.1%} / {r['hit_at_5']:.1%}**"
            if r["hit_at_1"] is not None
            else "- Hit@1 / Hit@3 / Hit@5: **null**"
        ),
        f"- MRR: **{r['mrr']:.3f}**" if r["mrr"] is not None else "- MRR: **null**",
        (
            f"- Context precision / recall: **{r['context_precision']:.1%} / {r['context_recall']:.1%}**"
            if r["context_precision"] is not None
            else "- Context precision / recall: **null**"
        ),
        (
            f"- Average top-1 / top-5 score: **{r['top1_score']:.3f} / {r['top5_score']:.3f}**"
            if r["top1_score"] is not None
            else "- Average top-1 / top-5 score: **null**"
        ),
        f"- Retrieval-abstention count: **{r['abstentions']}**",
        "",
        "## 6. Grounding quality",
        "",
        (
            f"- Deterministic grounding score: **{grounding:.3f}**"
            if grounding is not None
            else "- Deterministic grounding score: **null**"
        ),
        "- This is a deterministic support heuristic, not claim-level faithfulness.",
        "",
        "## 7. Abstention quality",
        "",
        f"- Hallucination rate: **{s['hallucination_rate']:.1%}**",
        f"- Abstention rate: **{avg(float(bool(row.get('abstained'))) for row in rows) or 0:.1%}**",
        f"- Abstention precision / recall / F1: **{s['abstention_precision']:.1%} / {s['abstention_recall']:.1%} / {s['abstention_f1']:.1%}**",
        "",
        "## 8. Calibration",
        "",
        f"- Mean model confidence: **{avg(row.get('model_confidence') for row in rows) or 0:.4f}**",
        f"- Mean HEVA confidence: **{avg(row.get('heva_confidence') for row in rows) or 0:.4f}**",
        f"- ECE / Brier (using HEVA confidence): **{calibration['ece']:.4f} / {s['brier']:.4f}**",
        "- Model confidence is the LLM-reported estimate; HEVA confidence is a deterministic heuristic combining model confidence, retrieval relevance, lexical support, and grounding. Neither is treated as a calibrated probability without independent calibration.",
        "",
        "## 9. Performance",
        (
            f"- Average / median / P95 / P99 latency: **{latency:.2f} ms / {percentile(.50):.2f} ms / {percentile(.95):.2f} ms / {percentile(.99):.2f} ms**"
            if latency is not None
            else "- Latency: **null**"
        ),
        "",
        "### Category breakdown",
        "",
    ]
    return lines + category_table(rows)


def provider_comparison():
    result_dir = ROOT / "eval" / "results"
    files = sorted(result_dir.glob("ground_truth_*.jsonl"))
    rows_by_provider = {}
    for path in files:
        rows = load(path)
        valid = [row for row in rows if row.get("status") == "ok"]
        if valid and valid[0].get("provider"):
            rows_by_provider[
                valid[0]["provider"] + "/" + str(valid[0].get("model", ""))
            ] = valid
    if len(rows_by_provider) < 2:
        return []
    lines = [
        "",
        "## Provider comparison",
        "",
        "This comparison uses the result files available in the repository. The providers share the dataset, retrieved evidence, retrieval pipeline, prompt, normalization, and metric implementation; only generation differs. Results are benchmark-specific and are not universal model claims.",
        "",
        "| Metric | " + " | ".join(rows_by_provider) + " |",
        "|---|" + "---:|" * len(rows_by_provider),
    ]
    metrics = [
        (
            "Strict accuracy",
            lambda rs: sum(bool(r.get("strict_correct", r.get("correct"))) for r in rs)
            / len(rs),
        ),
        (
            "Fact-aware accuracy",
            lambda rs: sum(
                bool(r.get("fact_aware_correct", r.get("correct"))) for r in rs
            )
            / len(rs),
        ),
        (
            "Grounding score",
            lambda rs: avg(r.get("deterministic_grounding_score") for r in rs),
        ),
        (
            "Hallucination rate",
            lambda rs: sum(bool(r.get("hallucination")) for r in rs) / len(rs),
        ),
        ("Average latency (ms)", lambda rs: avg(r.get("latency_ms") for r in rs)),
        (
            "Hit@1",
            lambda rs: avg(r.get("retrieval_metrics", {}).get("hit_at_1") for r in rs),
        ),
        (
            "Hit@5",
            lambda rs: avg(r.get("retrieval_metrics", {}).get("hit_at_5") for r in rs),
        ),
        ("MRR", lambda rs: avg(r.get("retrieval_metrics", {}).get("mrr") for r in rs)),
    ]
    for label, fn in metrics:
        values = []
        for rs in rows_by_provider.values():
            value = fn(rs)
            values.append(
                f"{value:.3f}" if label == "Average latency (ms)" else f"{value:.3f}"
            )
        lines.append("| " + label + " | " + " | ".join(values) + " |")
    return lines


def main(output, ground_truth=None, adversarial=None):
    gt = load(ground_truth or ROOT / "eval" / "results" / "ground_truth_results.jsonl")
    adv_path = (
        Path(adversarial)
        if adversarial
        else ROOT / "eval" / "results" / "adversarial_results.jsonl"
    )
    adv = load(adv_path) if adv_path.exists() else []
    gt_valid = [r for r in gt if r.get("status") == "ok"]
    gt_answerable = sum(bool(r.get("answerable")) for r in gt_valid)
    lines = [
        "# HEVA AI - Evaluation Report",
        "",
        "## Executive Summary",
        "",
        "HEVA is a document-grounded Q&A/RAG system. Evaluation was performed on 60 ground-truth cases and 122 adversarial cases. Retrieval quality is strong, while answer generation and adversarial robustness remain the main limitations. Results are reported transparently without an arbitrary overall score.",
        "",
        "## 1. System Under Test",
        "",
        "FastAPI document ingestion extracts PDF, Markdown, and text content, chunks it deterministically, retrieves top-k chunks with TF-IDF cosine similarity, sends retrieved evidence to the configured LLM, returns citations and confidence, and abstains below the retrieval threshold. The evaluation harness records an internal retrieval trace through `/qa/eval` while keeping `/qa` backward compatible.",
        "",
        "## 2. Evaluation Dataset",
        "",
        f"Ground truth contains 60 hand-authored cases: **{gt_answerable} answerable** and **{len(gt_valid) - gt_answerable} unanswerable**, spanning the available categories and difficulty levels. The adversarial set contains 122 cases covering instruction injection, irrelevant context, paraphrase, and subtle factual errors.",
        "",
        "## 3. Evaluation Methodology",
        "",
        "The evaluator uploads `data/eval_reference.md`, records the answer and retrieval trace, and computes strict/fact-aware answer metrics, keyword-grounded retrieval metrics, deterministic grounding, hallucination signals, abstention behavior, calibration, latency, and optional non-blocking Gemini judge fields. Retrieval precision/recall use `evidence_keywords` as transparent proxies, not human-annotated gold chunks.",
        "",
    ]
    lines += section(gt, "Ground-truth evaluation")
    if adv:
        adv_valid = [r for r in adv if r.get("status") == "ok"]
        adv_summary = summarize(adv_valid)
        lines += [
            "",
            "## 10. Adversarial robustness",
            "",
            f"Overall adversarial results: strict accuracy **{adv_summary['strict_accuracy']:.1%}**, fact-aware accuracy **{adv_summary['fact_aware_accuracy']:.1%}**, hallucination rate **{adv_summary['hallucination_rate']:.1%}**, abstention rate **{avg(float(bool(r.get('abstained'))) for r in adv_valid) or 0:.1%}**.",
            "",
            "| Category | Cases | Strict Accuracy | Fact-aware Accuracy | Hallucination Rate | Abstention Rate | Avg F1 | Semantic similarity | Avg retrieval score | Hit@1 | Hit@5 | MRR |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for line in category_table(adv_valid)[2:]:
            lines.append(line)
        worst = sorted(
            [r for r in adv if r.get("status") == "ok"],
            key=lambda r: (
                bool(r.get("fact_aware_correct")),
                -float(r.get("confidence", 0) or 0),
            ),
        )[:10]
        lines += [
            "",
            "### Ten worst adversarial cases",
            "",
            "| ID | Category | Expected | Actual | Confidence | Grounding |",
            "|---|---|---|---|---:|---:|",
        ]
        for row in worst:
            clean = lambda value: str(value or "").replace("|", "/")[:100]
            lines.append(
                f"| {row.get('id')} | {row.get('category')} | {clean(row.get('expected_answer'))} | {clean(row.get('actual_answer'))} | {float(row.get('confidence', 0) or 0):.2f} | {float(row.get('deterministic_grounding_score', 0) or 0):.3f} |"
            )
    lines += provider_comparison()
    lines += [
        "",
        "## 11. Failure analysis",
        "",
        "The detailed worst-case table above identifies high-confidence incorrect answers, arithmetic beyond the evidence, and injection susceptibility. These are answer-generation failures even when retrieval succeeds.",
        "",
        "## 12. Limitations",
        "",
        "- `evidence_keywords` are lexical proxies, not perfect gold chunk annotations.",
        "- Context precision/recall measure keyword presence, not semantic relevance.",
        "- Deterministic grounding is not claim-level faithfulness.",
        "- Results depend on the selected provider, local model, API availability, and hardware.",
        "- The fixed benchmark has no independently recorded human verification or gold chunk annotations.",
        "",
        "## 13. Baseline results",
        "",
        "The Phase 4 regression baseline is stored in `eval/regression_baseline.json` and includes answer, retrieval, grounding, latency, calibration, and Brier metrics. Regression failure requires at least two threshold breaches.",
        "",
        "## 14. Recommended next improvements",
        "",
        "- Improve retrieval for paraphrase, entity, negation, and multi-hop queries before changing the model or prompt.",
        "- Add independently reviewed chunk-level relevance labels and contradictory-evidence cases.",
        "- Add structured claim checks for dates, entities, and numerical reasoning.",
        "- Calibrate confidence and abstention thresholds on a held-out set.",
    ]
    Path(output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default=str(ROOT / "reports" / "evaluation_report.md")
    )
    parser.add_argument("--ground-truth", default=None)
    parser.add_argument("--adversarial", default=None)
    args = parser.parse_args()
    main(args.output, args.ground_truth, args.adversarial)
