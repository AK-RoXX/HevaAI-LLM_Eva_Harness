"""
Regression test runner: detect if prompt changes cause regressions.
Compare current results against baseline to find performance degradation.
"""
import argparse
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from statistics import mean

from eval.metrics import summarize

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "eval" / "results" / "ground_truth_results.jsonl"
SNAP = ROOT / "eval" / "regression_baseline.json"


REGRESSION_THRESHOLDS = {
    "fact_aware_accuracy": 0.05,
    "f1": 0.05,
    "hallucination_rate": 0.03,
    "grounding_score": 0.05,
    "hit_at_1": 0.05,
    "hit_at_5": 0.05,
    "mrr": 0.05,
    "latency": 0.20,
    "ece": 0.05,
    "brier": 0.05,
}


def phase4_metrics(results):
    valid = [r for r in results if r.get("status") == "ok"]
    s = summarize(valid)
    retrieval = [r.get("retrieval_metrics", {}) for r in valid]
    average = lambda key: mean([float(x[key]) for x in retrieval if x.get(key) is not None]) if any(x.get(key) is not None for x in retrieval) else None
    grounding = [r.get("deterministic_grounding_score") for r in valid if r.get("deterministic_grounding_score") is not None]
    f1 = [r.get("metrics", {}).get("f1") for r in valid if r.get("metrics", {}).get("f1") is not None]
    latency = [r.get("latency_ms") for r in valid if r.get("latency_ms") is not None]
    return {
        "fact_aware_accuracy": s.get("fact_aware_accuracy", 0.0),
        "f1": mean(f1) if f1 else None,
        "hallucination_rate": s.get("hallucination_rate", 0.0),
        "grounding_score": mean(grounding) if grounding else None,
        "hit_at_1": average("hit_at_1"), "hit_at_5": average("hit_at_5"), "mrr": average("mrr"),
        "latency": mean(latency) if latency else None,
        "ece": s.get("ece"), "brier": s.get("brier"),
    }


def phase4_save(results):
    baseline = {"timestamp": datetime.now().isoformat(), "name": "phase4_ground_truth", "metrics": phase4_metrics(results), "results": {r["id"]: {"correct": bool(r.get("correct")), "fact_aware_correct": bool(r.get("fact_aware_correct")), "answer": r.get("actual_answer", "")} for r in results}}
    SNAP.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    print(f"Saved Phase 4 baseline: {len(baseline['results'])} cases")


def phase4_check(results):
    if not SNAP.exists():
        raise RuntimeError("No regression baseline found. Run 'save' first.")
    baseline = json.loads(SNAP.read_text(encoding="utf-8"))
    old = baseline.get("metrics", {})
    new = phase4_metrics(results)
    breaches = []
    for name, threshold in REGRESSION_THRESHOLDS.items():
        if old.get(name) is None or new.get(name) is None:
            continue
        change = new[name] - old[name]
        adverse = change < -threshold if name not in {"hallucination_rate", "latency", "ece", "brier"} else change > threshold
        if adverse:
            breaches.append({"metric": name, "baseline": old[name], "current": new[name], "change": change, "threshold": threshold})
    print("Phase 4 regression comparison")
    for name in REGRESSION_THRESHOLDS:
        if old.get(name) is not None and new.get(name) is not None:
            print(f"{name}: {old[name]:.4f} -> {new[name]:.4f} ({new[name]-old[name]:+.4f})")
    print(f"Threshold breaches: {len(breaches)}; regression failure requires at least 2 breaches.")
    return breaches


def load(p):
    """Load evaluation results from JSONL file."""
    return [
        json.loads(x)
        for x in Path(p).read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def save_baseline(results, name="baseline"):
    """Save current results as regression baseline."""
    baseline = {
        "timestamp": datetime.now().isoformat(),
        "name": name,
        "results": {r["id"]: {
            "correct": bool(r.get("correct")),
            "confidence": float(r.get("confidence", 0)),
            "category": r.get("category"),
            "answer": r.get("actual_answer", ""),
        } for r in results}
    }
    
    SNAP.write_text(json.dumps(baseline, indent=2))
    print(f"✓ Saved regression baseline: {len(baseline['results'])} cases")
    return baseline


def check_regressions(results, baseline=None):
    """
    Compare current results against baseline.
    Returns dict with regressions, improvements, and summary.
    """
    if baseline is None:
        if not SNAP.exists():
            raise RuntimeError("No regression baseline found. Run 'save' first.")
        baseline = json.loads(SNAP.read_text())
    
    old_results = baseline.get("results", {})
    current = {r["id"]: {
        "correct": bool(r.get("correct")),
        "confidence": float(r.get("confidence", 0)),
        "category": r.get("category"),
        "answer": r.get("actual_answer", ""),
    } for r in results}
    
    regressions = []  # Was passing, now failing
    improvements = []  # Was failing, now passing
    confidence_drops = []  # Same result but confidence dropped
    
    for test_id, old in old_results.items():
        if test_id not in current:
            continue
        
        new = current[test_id]
        
        # Regression: was correct, now incorrect
        if old["correct"] and not new["correct"]:
            regressions.append({
                "id": test_id,
                "category": old.get("category"),
                "previous_answer": old.get("answer", ""),
                "current_answer": new.get("answer", ""),
            })
        
        # Improvement: was incorrect, now correct
        elif not old["correct"] and new["correct"]:
            improvements.append({
                "id": test_id,
                "category": new.get("category"),
                "answer": new.get("answer", ""),
            })
        
        # Confidence drop (same correctness but lower confidence)
        elif (old["correct"] == new["correct"] and 
              old["confidence"] > 0 and 
              new["confidence"] < old["confidence"] * 0.9):
            confidence_drops.append({
                "id": test_id,
                "old_confidence": old["confidence"],
                "new_confidence": new["confidence"],
            })
    
    summary = {
        "regressions": regressions,
        "improvements": improvements,
        "confidence_drops": confidence_drops,
        "baseline_passed": sum(1 for r in old_results.values() if r["correct"]),
        "current_passed": sum(1 for r in current.values() if r["correct"]),
    }
    
    return summary


def print_regression_report(summary):
    """Print formatted regression report."""
    print("\n" + "=" * 70)
    print("REGRESSION TEST REPORT")
    print("=" * 70)
    
    baseline_passed = summary["baseline_passed"]
    current_passed = summary["current_passed"]
    change = current_passed - baseline_passed
    
    print(f"\nBaseline: {baseline_passed} passing")
    print(f"Current:  {current_passed} passing")
    print(f"Change:   {change:+d} ({change/baseline_passed*100:+.1f}%)")
    
    if summary["regressions"]:
        print(f"\n❌ REGRESSIONS: {len(summary['regressions'])}")
        print("-" * 70)
        for item in summary["regressions"][:5]:
            print(f"  • {item['id']} ({item['category']})")
            print(f"    Was: {item['previous_answer'][:50]}")
            print(f"    Now: {item['current_answer'][:50]}")
        if len(summary["regressions"]) > 5:
            print(f"  ... and {len(summary['regressions'])-5} more")
    
    if summary["improvements"]:
        print(f"\n✅ IMPROVEMENTS: {len(summary['improvements'])}")
        for item in summary["improvements"][:3]:
            print(f"  ✓ {item['id']}")
    
    if summary["confidence_drops"]:
        print(f"\n⚠️  CONFIDENCE DROPS: {len(summary['confidence_drops'])}")
        for item in summary["confidence_drops"][:3]:
            print(f"  • {item['id']}: {item['old_confidence']:.2f} → {item['new_confidence']:.2f}")
    
    print("\n" + "=" * 70)
    
    # Return success if no regressions
    return len(summary["regressions"]) == 0


def main():
    p = argparse.ArgumentParser(description="Regression test runner for evaluation")
    p.add_argument("command", choices=["save", "check"], 
                  help="save=create baseline, check=compare to baseline")
    p.add_argument("--results", default=str(BASE),
                  help="Path to evaluation results JSONL")
    p.add_argument("--baseline", default=str(SNAP),
                  help="Path to save/load baseline")
    
    a = p.parse_args()
    results = load(a.results)
    
    if a.command == "save":
        phase4_save(results)
    else:
        exit(1 if len(phase4_check(results)) >= 2 else 0)


if __name__ == "__main__":
    main()
