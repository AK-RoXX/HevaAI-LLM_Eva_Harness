import json
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
from .metrics import summarize

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return [
        json.loads(x)
        for x in Path(path).read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def cluster(r):
    if r.get("correct"):
        return None
    c = r.get("category", "")
    if r.get("abstained") and r.get("answerable"):
        return "False abstention / retrieval recall"
    if r.get("hallucination"):
        return "Grounding / hallucination"
    if "injection" in c:
        return "Instruction-injection susceptibility"
    if "paraphrase" in c:
        return "Distribution-shift / paraphrase"
    if "subtle" in c:
        return "False-premise / factual-error handling"
    if r.get("semantic_support", 1) < 0.18:
        return "Retrieval mismatch"
    return "Reasoning / answer correctness"


def main(path):
    rows = load(path)
    outdir = ROOT / "reports" / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    bins = []
    for i in range(10):
        x = [
            r
            for r in rows
            if i / 10 <= r.get("confidence", 0) < (i + 1) / 10
            or (i == 9 and r.get("confidence", 0) == 1)
        ]
        if x:
            bins.append(
                (
                    (i + 0.5) / 10,
                    sum(r["correct"] for r in x) / len(x),
                    sum(r.get("confidence", 0) for r in x) / len(x),
                )
            )
    if bins:
        plt.figure(figsize=(7, 5))
        plt.plot(
            [x[0] for x in bins], [x[1] for x in bins], "o-", label="Observed accuracy"
        )
        plt.plot([0, 1], [0, 1], "--", label="Perfect calibration")
        plt.xlabel("Mean confidence")
        plt.ylabel("Accuracy")
        plt.title("Confidence calibration")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "calibration.png", dpi=160)
        plt.close()
    clusters = Counter(cluster(r) for r in rows if cluster(r))
    clusters.pop(None, None)
    (ROOT / "reports" / "failure_clusters.json").write_text(
        json.dumps(clusters, indent=2), encoding="utf-8"
    )
    report = ROOT / "reports" / "evaluation_report.md"
    lines = [
        (
            report.read_text(encoding="utf-8")
            if report.exists()
            else "# Evaluation Report"
        ),
        "\n## Failure Mode Distribution",
        "",
    ]
    for k, v in clusters.most_common():
        lines.append(f"- **{k}**: {v}")
    lines += ["", "Calibration plot: `figures/calibration.png`", ""]
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f'Analysis written to {ROOT / "reports"}')


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("results")
    a = p.parse_args()
    main(a.results)
