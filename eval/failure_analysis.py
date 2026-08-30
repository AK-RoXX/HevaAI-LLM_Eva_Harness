"""
Failure mode clustering: group failures by root cause and identify patterns.
"""
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]


def cluster_failures(results):
    """
    Cluster failures into distinct failure modes based on characteristics.
    Returns: dict mapping failure mode -> list of failed test cases
    """

    failures = [r for r in results if r.get("status") == "ok" and not r.get("correct")]

    if not failures:
        return {}

    clusters = defaultdict(list)

    for r in failures:
        failure_mode = categorize_failure(r)
        clusters[failure_mode].append(r)

    return dict(clusters)


def categorize_failure(result):
    """
    Determine the failure mode of a single result.
    Returns the category as a string.
    """

    # Check for hallucination
    if result.get("hallucination"):
        if result.get("semantic_support", 0) < 0.15:
            return "HALLUCINATION_LOW_SUPPORT"
        else:
            return "HALLUCINATION_WITH_SUPPORT"

    # Check for abstention when answer expected
    if result.get("abstained") and result.get("answerable"):
        return "INCORRECT_ABSTENTION"

    # Check for confidence/accuracy mismatch
    conf = result.get("confidence", 0)
    if conf > 0.7:
        return "CONFIDENT_INCORRECT"
    elif conf < 0.3:
        return "LOW_CONFIDENCE_INCORRECT"

    # Check for semantic support
    semantic_support = result.get("semantic_support", 0)
    if semantic_support < 0.1:
        return "LOW_SEMANTIC_SUPPORT"
    elif semantic_support < 0.25:
        return "WEAK_SEMANTIC_SUPPORT"

    # Check answer length mismatch
    expected_len = len(result.get("expected_answer", "").split())
    actual_len = len(result.get("actual_answer", "").split())

    if actual_len == 0 and not result.get("abstained"):
        return "EMPTY_ANSWER"
    elif actual_len < expected_len / 2:
        return "INCOMPLETE_ANSWER"
    elif actual_len > expected_len * 2:
        return "VERBOSE_ANSWER"

    # Check for reasoning questions
    if result.get("category") in ["reasoning", "multi_hop"]:
        return "REASONING_FAILURE"

    # Check for unanswerable questions
    if not result.get("answerable") and not result.get("abstained"):
        return "FAILED_UNANSWERABLE"

    # Default
    return "OTHER"


def analyze_failure_modes(results):
    """
    Detailed analysis of failure modes with insights.
    """
    clusters = cluster_failures(results)

    if not clusters:
        return {
            "total_failures": 0,
            "clusters": {},
        }

    analysis = {
        "total_failures": sum(len(items) for items in clusters.values()),
        "clusters": {},
    }

    for mode, items in sorted(
        clusters.items(), key=lambda x: len(x[1]), reverse=True
    ):
        categories = defaultdict(int)
        avg_confidence = 0
        avg_semantic_support = 0

        for item in items:
            categories[item.get("category", "unknown")] += 1
            avg_confidence += item.get("confidence", 0)
            avg_semantic_support += item.get("semantic_support", 0)

        avg_confidence = avg_confidence / len(items) if items else 0
        avg_semantic_support = avg_semantic_support / len(items) if items else 0

        analysis["clusters"][mode] = {
            "count": len(items),
            "percentage": 100 * len(items) / analysis["total_failures"],
            "avg_confidence": avg_confidence,
            "avg_semantic_support": avg_semantic_support,
            "most_common_categories": dict(
                sorted(categories.items(), key=lambda x: x[1], reverse=True)
            ),
            "examples": [
                {
                    "id": item["id"],
                    "question": item["question"][:60],
                    "expected": item.get("expected_answer", "")[:40],
                    "got": item.get("actual_answer", "")[:40],
                    "confidence": item.get("confidence", 0),
                }
                for item in items[:3]
            ],
        }

    return analysis


def failure_insights(failure_mode, examples):
    """
    Generate model-level insight for a failure mode.
    Describes what the model is doing wrong, not just what the outputs look like.
    """

    insights = {
        "HALLUCINATION_LOW_SUPPORT": {
            "description": "Model generates confident answers unsupported by evidence",
            "behavior": "LLM is fabricating facts or reasoning beyond document scope",
            "model_level": "Insufficient prompt constraints; model defaulting to prior knowledge",
            "improvement": "Add explicit instruction to ONLY use supplied evidence; implement citation requirement validation; reduce temperature further (currently 0.0)",
        },
        "HALLUCINATION_WITH_SUPPORT": {
            "description": "Answer partially supported but adds details not in evidence",
            "behavior": "Model is elaborating on evidence with inferences not requested",
            "model_level": "Weak grounding signal; model fills gaps in knowledge with plausible but unverified details",
            "improvement": "Tighten abstention threshold; require explicit evidence markers in prompt; implement strict fact validation against citations",
        },
        "INCORRECT_ABSTENTION": {
            "description": "Model refuses to answer when answer is available",
            "behavior": "Overly cautious; abstaining due to low retrieval scores or weak semantic match",
            "model_level": "Abstention threshold too high (0.08); retrieval missing relevant chunks",
            "improvement": "Lower abstain_score_threshold; improve chunk size/overlap in retrieval; add query expansion for edge cases",
        },
        "CONFIDENT_INCORRECT": {
            "description": "High confidence (>0.7) but wrong answer",
            "behavior": "Model overconfident in incorrect reasoning",
            "model_level": "Confidence calibration issue; model not learning from weak evidence signals",
            "improvement": "Add confidence penalty for low evidence support; implement entropy-based uncertainty; train on harder negatives",
        },
        "LOW_SEMANTIC_SUPPORT": {
            "description": "Answer has minimal semantic overlap with citations",
            "behavior": "Drift between question and answer; off-topic responses",
            "model_level": "Prompt confusion or poor instruction following",
            "improvement": "Add instruction to maintain topical coherence; implement semantic consistency check; use retrieval-augmented generation with reranking",
        },
        "WEAK_SEMANTIC_SUPPORT": {
            "description": "Answer only loosely matches citations",
            "behavior": "Answers tangentially related to evidence",
            "model_level": "Model extracting partial information but missing key relationships",
            "improvement": "Improve evidence selection (rerank by relevance); add relationship extraction task; implement coreference resolution",
        },
        "EMPTY_ANSWER": {
            "description": "Model returns empty or null response without abstaining",
            "behavior": "API or LLM failure manifesting as empty output",
            "model_level": "JSON parsing issue or LLM generating invalid output",
            "improvement": "Add output validation; implement fallback to abstention; improve error handling",
        },
        "INCOMPLETE_ANSWER": {
            "description": "Answer is partial or truncated",
            "behavior": "Model stops before completing thought",
            "model_level": "Token limit hit or early termination",
            "improvement": "Increase max_tokens; use structured output format; implement completion checking",
        },
        "VERBOSE_ANSWER": {
            "description": "Answer is unnecessarily long and rambling",
            "behavior": "Model padding response; over-explaining",
            "model_level": "No length constraint in prompt; model defaulting to verbosity",
            "improvement": "Add length constraint to prompt; use conciseness penalty in scoring",
        },
        "REASONING_FAILURE": {
            "description": "Multi-step reasoning questions answered incorrectly",
            "behavior": "Model fails at intermediate steps (e.g., arithmetic, aggregation)",
            "model_level": "LLM not designed for precise reasoning; context length issues",
            "improvement": "Add step-by-step reasoning prompt; break into sub-questions; implement calculator tool; verify arithmetic separately",
        },
        "FAILED_UNANSWERABLE": {
            "description": "Model provides answer when question is unanswerable",
            "behavior": "Model hallucinating answer for unknown question",
            "model_level": "Abstention signal weak for negative cases",
            "improvement": "Add explicit 'not in document' training; lower confidence threshold for abstention",
        },
    }

    return insights.get(failure_mode, {
        "description": failure_mode,
        "behavior": "Unknown failure mode",
        "model_level": "Needs investigation",
        "improvement": "Investigate specific examples",
    })


def print_failure_analysis(analysis):
    """Print formatted failure mode analysis."""
    print("\n" + "=" * 80)
    print("FAILURE MODE ANALYSIS")
    print("=" * 80)

    if analysis["total_failures"] == 0:
        print("✅ No failures detected!")
        return

    print(f"\nTotal failures: {analysis['total_failures']}\n")

    for mode, data in analysis["clusters"].items():
        insights = failure_insights(mode, data["examples"])

        print("=" * 80)
        print(f"CLUSTER: {mode}")
        print("=" * 80)
        print(f"Count: {data['count']} ({data['percentage']:.1f}% of failures)")
        print(f"Avg Confidence: {data['avg_confidence']:.3f}")
        print(f"Avg Semantic Support: {data['avg_semantic_support']:.3f}")
        print(f"Categories: {dict(data['most_common_categories'])}")
        print()
        print(f"Description: {insights['description']}")
        print(f"Observed Behavior: {insights['behavior']}")
        print(f"Root Cause (Model Level): {insights['model_level']}")
        print(f"\nProposed Solution:\n  {insights['improvement']}")
        print()
        print("Examples:")
        for ex in data["examples"]:
            print(f"  • {ex['id']}: '{ex['question']}' → '{ex['got'][:30]}'")
        print()
