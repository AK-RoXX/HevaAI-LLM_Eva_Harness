import json
from pathlib import Path
from collections import Counter
from .metrics import summarize

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = list((ROOT / "eval" / "results").glob("*_results.jsonl"))
    lines = [
        "# Heva AI — Adversarial Q&A Evaluation Report",
        "",
        "Generated from the custom evaluation harness. No RAGAS, TruLens, or DeepEval is used.",
        "",
    ]
    for f in files:
        rows = [
            json.loads(x)
            for x in f.read_text(encoding="utf-8").splitlines()
            if x.strip()
        ]
        s = summarize(rows)
        lines += [
            f"## {f.stem}",
            "",
            f"- Cases: {s['n']}",
            f"- Accuracy: {s['accuracy']:.1%}",
            f"- Abstention rate: {s['abstention_rate']:.1%}",
            f"- Hallucination rate: {s['hallucination_rate']:.1%}",
            f"- Expected Calibration Error: {s['ece']:.4f}",
            f"- Brier score: {s['brier']:.4f}",
            "",
        ]
        cats = {}
        for r in rows:
            cats.setdefault(r.get("category", "unknown"), []).append(r)
        lines.append("### Accuracy by input type")
        for c, rs in sorted(cats.items()):
            lines.append(
                f"- {c}: {sum(x['correct'] for x in rs)/len(rs):.1%} ({len(rs)} cases)"
            )
        lines += [
            "",
            "### Failure mode candidates",
            "",
            "The harness preserves every failed case with its category, retrieval citations, semantic-support score, confidence and actual output. Review failures before assigning final causal labels.",
            "",
        ]
        failures = [r for r in rows if not r["correct"]]
        for r in failures[:30]:
            lines.append(
                f"- **{r['id']}** [{r.get('category')}] confidence={r.get('confidence',0):.2f}, support={r.get('semantic_support',0):.2f}: {r.get('actual_answer','')}"
            )
    (ROOT / "reports" / "evaluation_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(ROOT / "reports" / "evaluation_report.md")


if __name__ == "__main__":
    main()
