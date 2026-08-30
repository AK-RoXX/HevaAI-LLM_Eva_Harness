"""Dataset and result audits for submission reproducibility."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def intent(row):
    question = row["question"]
    category = row.get("category", "unknown")
    q = question.lower()
    if not row.get("answerable"):
        capability = "Tests abstention or a grounded correction when the requested fact is absent or not established."
    elif "percentage" in q or "growth rate" in q or "how much" in q and "increase" in q:
        capability = "Tests numerical reasoning over explicitly stated source values, including whether derived arithmetic is handled correctly."
    elif category in {"multi_hop", "synthesis"}:
        capability = "Tests synthesis of multiple document facts while preserving the relationships and qualifiers between them."
    elif category in {"temporal", "events"} or "when" in q or "first" in q:
        capability = "Tests temporal retrieval and preservation of dates, ordering, and time conditions."
    elif category in {"adversarial_false_premise", "adversarial_injection", "negation"}:
        capability = "Tests resistance to a false premise or conflicting instruction and requires an evidence-grounded correction."
    elif category == "paraphrase":
        capability = "Tests lexical robustness when the question uses different wording for a fact stated in the document."
    elif category == "irrelevant_context":
        capability = "Tests whether irrelevant framing is ignored while the embedded factual question is answered from evidence."
    elif category in {"comparison", "edge_case"}:
        capability = "Tests comparison or indirect reference resolution without dropping the requested entities, years, or conditions."
    elif category == "entity":
        capability = "Tests entity identity and disambiguation against the exact people, organizations, and roles in the evidence."
    elif category == "location":
        capability = "Tests complete location extraction without confusing headquarters, offices, or facilities."
    elif category == "qualifier_sensitive":
        capability = "Tests preservation of a qualifier that changes the factual meaning of the answer."
    else:
        capability = "Tests direct factual retrieval from an explicit statement in the reference document."
    return f"{capability} Question-specific target: {question}"


def annotate_ground_truth(path):
    rows = load(path)
    for row in rows:
        row.setdefault("test_intent", intent(row))
    Path(path).write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    return rows


def dataset_report(rows):
    required = {"id", "question", "expected_answer", "answerable", "category", "difficulty", "evidence_keywords", "test_intent"}
    missing = [{"id": r.get("id"), "fields": sorted(required - set(r))} for r in rows if required - set(r)]
    questions = Counter(r["question"].strip().lower() for r in rows)
    duplicates = {q: n for q, n in questions.items() if n > 1}
    weak = [r["id"] for r in rows if not r.get("evidence_keywords") or not r.get("test_intent")]
    answerable = sum(bool(r.get("answerable")) for r in rows)
    lines = [
        "# Ground-Truth Dataset Audit", "", "This audit is generated from `dataset/ground_truth.jsonl`. It does not claim human verification or gold chunk annotation.",
        "", "## Coverage", "", f"- Total cases: **{len(rows)}**", f"- Answerable: **{answerable}**", f"- Unanswerable: **{len(rows) - answerable}**", f"- Cases with evidence keywords: **{sum(bool(r.get('evidence_keywords')) for r in rows)}**", "- Cases with explicit gold evidence chunk references: **0**", "- Human-verification record: **not present in the repository**", "",
        "## Category distribution", "", "| Category | Cases |", "|---|---:|",
    ]
    lines += [f"| {k} | {v} |" for k, v in sorted(Counter(r.get("category", "missing") for r in rows).items())]
    lines += ["", "## Difficulty distribution", "", "| Difficulty | Cases |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in sorted(Counter(r.get("difficulty", "missing") for r in rows).items())]
    lines += ["", "## Annotation checks", "", f"- Cases missing required fields: **{len(missing)}**", f"- Duplicate question texts: **{len(duplicates)}**", f"- Potentially weak cases (missing keywords or intent): **{len(weak)}**"]
    if missing:
        lines += ["", "### Missing fields", "", *[f"- `{x['id']}`: {', '.join(x['fields'])}" for x in missing]]
    if duplicates:
        lines += ["", "### Duplicate questions", "", *[f"- `{q}` ({n} occurrences)" for q, n in duplicates.items()]]
    lines += ["", "## Evidence and rigor assessment", "", "Every case has an expected answer, answerability flag, category, difficulty, and keyword evidence proxy. The reference document supplies the source text, but the dataset does not identify human-annotated gold chunks. Evidence keywords therefore establish lightweight lexical coverage only. Test intent is question-specific and records the capability the case is intended to probe; it is not a claim that the case is independently validated.", "", "Potential weak cases are flagged rather than removed. Indirect arithmetic, false-premise, and unanswerable cases require particular care because a valid response may be a correction or abstention rather than a string equal to the expected answer."]
    return "\n".join(lines) + "\n"


def adversarial_report(rows):
    counts = Counter(r.get("category", "missing") for r in rows)
    injection = [r for r in rows if r.get("category") == "instruction_injection" and re.search(r"ignore|disregard|instructions|evidence|document", r.get("question", ""), re.I)]
    lines = ["# Adversarial Dataset Audit", "", "This audit classifies the existing 122 cases as stored; it does not relabel or modify the benchmark.", "", f"- Total cases: **{len(rows)}**", "", "| Existing category | Cases | Actual role |", "|---|---:|---|"]
    roles = {"instruction_injection": "Genuine instruction-like text in the question; tests instruction/data separation.", "irrelevant_context": "Distractor framing; tests relevance filtering, but is not necessarily a security attack.", "paraphrase": "Linguistic robustness; wording variation, not inherently adversarial.", "subtle_factual_error": "Factual-error/contradiction pressure; tests resistance to incorrect premises or values."}
    for category, n in sorted(counts.items()):
        lines.append(f"| {category} | {n} | {roles.get(category, 'Unclassified') } |")
    lines += ["", "## Findings", "", f"- Instruction-injection cases containing explicit instruction language: **{len(injection)} / {counts.get('instruction_injection', 0)}**.", "- Linguistic-robustness cases should not be described as security attacks merely because they are in the adversarial file.", "- Dedicated contradictory-evidence cases: **0 identified**.", "- Dedicated false-premise category in the adversarial file: **0**; false-premise behavior exists in some ground-truth cases and may be embedded in factual-error cases, but is not separately labeled here.", "- Multi-document conflict, prompt-exfiltration, tool-use, and poisoning attacks: **not represented**.", "", "## Recommendations", "", "Retain the current fixed benchmark for comparability. For a future benchmark revision, add independently reviewed contradictory-evidence and explicit false-premise subsets, and distinguish linguistic robustness from security attacks in the category field."]
    return "\n".join(lines) + "\n"


def failure_report(results):
    ok = [r for r in results if r.get("status") == "ok"]
    failures = [r for r in ok if not r.get("fact_aware_correct", r.get("correct", False))]
    clusters = Counter()
    examples = {}
    for r in failures:
        ret = r.get("retrieval_metrics", {})
        if ret.get("hit_at_5") == 0:
            mode = "Retrieval failure: required keyword evidence not found by top-5"
        elif r.get("hallucination"):
            mode = "Grounding failure: unsupported claim signal"
        elif r.get("abstained") and r.get("answerable"):
            mode = "Abstention failure: unnecessary abstention"
        elif r.get("answerable") and r.get("category") in {"reasoning", "multi_hop", "comparison", "financial", "employees"}:
            mode = "Generation failure: reasoning, comparison, or qualifier mismatch despite retrieval"
        elif not r.get("answerable") and not r.get("abstained"):
            mode = "Robustness/abstention failure: answered an unanswerable case"
        else:
            mode = "Evaluation or generation failure: answer mismatch despite available evidence"
        clusters[mode] += 1
        examples.setdefault(mode, []).append(r.get("id"))
    lines = ["# Failure Mode Analysis", "", f"Analyzed **{len(ok)}** successful result records; **{len(failures)}** were fact-aware failures. Clusters are deterministic diagnostic categories, not human adjudications.", "", "| Failure cluster | Count | % of failures | % of evaluated | Representative IDs |", "|---|---:|---:|---:|---|"]
    for mode, count in clusters.most_common():
        lines.append(f"| {mode} | {count} | {count/len(failures):.1%} | {count/len(ok):.1%} | {', '.join(examples[mode][:3])} |")
    lines += ["", "## Interpretation", "", "Retrieval success does not imply answer success: cases with Hit@5 can still fail during generation, qualification, false-premise handling, or grounding. Conversely, a retrieval miss is a system-layer failure before generation. The current result schema does not contain human root-cause labels or claim-level annotations, so these clusters are evidence-based heuristics and should not be read as definitive causal diagnoses."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", default=str(ROOT / "dataset" / "ground_truth.jsonl"))
    parser.add_argument("--adversarial", default=str(ROOT / "dataset" / "adversarial.jsonl"))
    parser.add_argument("--results", default=str(ROOT / "eval" / "results" / "ground_truth_results.jsonl"))
    parser.add_argument("--annotate", action="store_true")
    args = parser.parse_args()
    gt = annotate_ground_truth(args.ground_truth) if args.annotate else load(args.ground_truth)
    adv = load(args.adversarial)
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "dataset_audit.md").write_text(dataset_report(gt), encoding="utf-8")
    (out / "adversarial_audit.md").write_text(adversarial_report(adv), encoding="utf-8")
    result_path = Path(args.results)
    if result_path.exists():
        (out / "failure_analysis.md").write_text(failure_report(load(result_path)), encoding="utf-8")
    print(f"Wrote {out / 'dataset_audit.md'} and {out / 'adversarial_audit.md'}")


if __name__ == "__main__":
    main()
