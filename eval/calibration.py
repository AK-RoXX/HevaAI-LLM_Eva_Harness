"""
Confidence calibration analysis: does the model's confidence match its accuracy?
"""
import json
import math
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def compute_ece(results, n_bins=10):
    """
    Expected Calibration Error: measures gap between confidence and accuracy.
    Lower is better (well-calibrated models have ECE close to 0).
    """
    if not results:
        return 0.0

    bins = defaultdict(lambda: {"correct": 0, "total": 0, "confidences": []})

    for r in results:
        if r.get("status") != "ok":
            continue

        conf = float(r.get("heva_confidence", r.get("confidence", 0.0)) or 0.0)
        correct = bool(r.get("fact_aware_correct", r.get("correct", False)))

        # Assign to bin based on confidence
        bin_idx = min(n_bins - 1, int(conf * n_bins))
        bins[bin_idx]["total"] += 1
        bins[bin_idx]["confidences"].append(conf)
        if correct:
            bins[bin_idx]["correct"] += 1

    ece = 0.0
    total = sum(b["total"] for b in bins.values())

    for bin_idx in range(n_bins):
        if bin_idx not in bins or bins[bin_idx]["total"] == 0:
            continue

        b = bins[bin_idx]
        accuracy = b["correct"] / b["total"]
        avg_confidence = sum(b["confidences"]) / len(b["confidences"])

        weight = b["total"] / total
        ece += weight * abs(accuracy - avg_confidence)

    return ece


def analyze_calibration(results):
    """
    Analyze confidence calibration and return detailed statistics.
    """
    by_confidence = defaultdict(lambda: {"correct": 0, "total": 0})

    for r in results:
        if r.get("status") != "ok":
            continue

        conf_bucket = min(9, int(float(r.get("heva_confidence", r.get("confidence", 0.0)) or 0.0) * 10))
        by_confidence[conf_bucket]["total"] += 1
        if r.get("fact_aware_correct", r.get("correct")):
            by_confidence[conf_bucket]["correct"] += 1

    calibration_data = []
    for bucket in range(10):
        if bucket not in by_confidence:
            calibration_data.append({
                "confidence_range": f"{bucket * 0.1:.1f}-{(bucket + 1) * 0.1:.1f}",
                "expected_confidence": (bucket + 0.5) * 0.1,
                "actual_accuracy": 0.0,
                "count": 0,
            })
        else:
            data = by_confidence[bucket]
            accuracy = data["correct"] / data["total"] if data["total"] > 0 else 0
            calibration_data.append({
                "confidence_range": f"{bucket * 0.1:.1f}-{(bucket + 1) * 0.1:.1f}",
                "expected_confidence": (bucket + 0.5) * 0.1,
                "actual_accuracy": accuracy,
                "count": data["total"],
            })

    ece = compute_ece(results)

    return {
        "ece": ece,
        "calibration_by_bin": calibration_data,
    }


def calibration_report(results):
    """Generate human-readable calibration report."""
    analysis = analyze_calibration(results)
    ece = analysis["ece"]

    print("\n" + "=" * 70)
    print("CONFIDENCE CALIBRATION ANALYSIS")
    print("=" * 70)
    print(f"\nExpected Calibration Error (ECE): {ece:.4f}")
    print("Interpretation: Gap between model's confidence and actual accuracy.")
    print("  0.00 = perfect calibration")
    print("  < 0.10 = well calibrated")
    print("  > 0.20 = poorly calibrated")

    if ece < 0.10:
        calibration_quality = "✅ Well calibrated"
    elif ece < 0.20:
        calibration_quality = "⚠️  Acceptable calibration"
    else:
        calibration_quality = "❌ Poor calibration"

    print(f"Quality: {calibration_quality}\n")

    print("Confidence buckets (Expected vs Actual Accuracy):")
    print("-" * 70)
    print(
        f"{'Confidence':^15} | {'Expected Acc':^15} | {'Actual Acc':^15} | {'Cases':^8}"
    )
    print("-" * 70)

    for b in analysis["calibration_by_bin"]:
        conf_str = b["confidence_range"]
        exp_acc = b["expected_confidence"]
        actual_acc = b["actual_accuracy"]
        count = b["count"]

        bar_length = int(actual_acc * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)

        print(
            f"{conf_str:^15} | {exp_acc:^15.2%} | {actual_acc:^15.2%} | {count:^8}"
        )

    print("=" * 70)

    return analysis
