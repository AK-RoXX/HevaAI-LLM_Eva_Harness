from eval.metrics import deterministic_grounding_score, retrieval_metrics


def trace(*texts, abstained=False):
    return {
        "retrieved_chunks": [
            {"chunk_id": f"c{i}", "rank": i + 1, "score": 0.5 - i * 0.1, "text": text, "document_id": "doc"}
            for i, text in enumerate(texts)
        ],
        "retrieval_threshold": 0.08,
        "retrieval_abstained": abstained,
    }


def test_hit_at_k_and_mrr_require_all_keywords():
    row = {"evidence_keywords": ["founded in 2016", "Maya Rao"]}
    metrics = retrieval_metrics(row, trace("founded in 2016", "Maya Rao"))
    assert metrics["hit_at_1"] == 0.0
    assert metrics["hit_at_3"] == 1.0
    assert metrics["hit_at_5"] == 1.0
    assert metrics["mrr"] == 0.5


def test_context_precision_recall_and_scores():
    row = {"evidence_keywords": ["revenue 2024", "24.7 million"]}
    metrics = retrieval_metrics(row, trace("revenue 2024 was $24.7 million", "unrelated text"))
    assert metrics["context_precision"] == 0.5
    assert metrics["context_recall"] == 1.0
    assert metrics["top1_score"] == 0.5
    assert metrics["top5_score"] == 0.45


def test_no_keywords_returns_null_retrieval_quality_metrics():
    metrics = retrieval_metrics({}, trace("anything"))
    assert metrics["hit_at_1"] is None
    assert metrics["hit_at_5"] is None
    assert metrics["mrr"] is None
    assert metrics["context_precision"] is None
    assert metrics["context_recall"] is None


def test_empty_and_malformed_trace_are_safe():
    row = {"evidence_keywords": ["fact"]}
    empty = retrieval_metrics(row, {"retrieved_chunks": [], "retrieval_abstained": True})
    malformed = retrieval_metrics(row, {"retrieved_chunks": [None, {"score": "bad"}]})
    assert empty["hit_at_1"] == 0.0 and empty["retrieval_abstained"] is True
    assert malformed["hit_at_5"] == 0.0


def test_deterministic_grounding_score_penalizes_hallucination():
    row = {"evidence_keywords": ["founded in 2016"]}
    result = {"answer": "founded in 2016", "citations": [{"text": "founded in 2016"}]}
    assert deterministic_grounding_score(row, result, False, 1.0) == 1.0
    assert deterministic_grounding_score(row, result, True, 1.0) == 0.0
