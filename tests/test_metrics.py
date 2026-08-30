from eval.metrics import (
    answer_metrics,
    answer_correct,
    exact_match,
    token_f1,
    token_precision,
    token_recall,
    evaluation_state,
    fact_aware_match,
    is_grounded_negative,
)


def test_exact_match_normalizes_factual_text():
    assert exact_match("Revenue was $24.7 million.", " revenue was $24.7 million ") == 1.0
    assert exact_match("2016", "2017") == 0.0


def test_token_precision_recall_and_f1():
    reference = "Maya Rao founded Helio in 2016"
    prediction = "Maya Rao founded Helio"
    assert token_precision(reference, prediction) == 1.0
    assert token_recall(reference, prediction) == 4 / 6
    assert round(token_f1(reference, prediction), 6) == round(4 / 5, 6)


def test_bleu_and_rouge_handle_short_answers():
    metrics = answer_metrics("2016", "2016")
    assert metrics["bleu"] == 1.0
    assert metrics["rouge1"] == 1.0
    assert metrics["rouge2"] == 1.0
    assert metrics["rougeL"] == 1.0


def test_empty_answer_is_safe():
    metrics = answer_metrics("2016", "")
    assert metrics["exact_match"] == 0.0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["bleu"] == 0.0
    assert metrics["rouge1"] == 0.0


def test_empty_reference_and_prediction_match():
    metrics = answer_metrics("", "")
    assert all(value == 1.0 for value in metrics.values())


def test_semantic_similarity_is_lexical_tfidf():
    metrics = answer_metrics("Helio launched in 2016", "Helio launched in 2016")
    assert metrics["semantic_similarity"] == 1.0


def test_unanswerable_correctness_uses_abstention_only():
    row = {
        "answerable": False,
        "expected_answer": "The document does not provide that information.",
    }
    assert answer_correct(row, {"abstained": True, "answer": "anything"}) is True
    assert answer_correct(row, {"abstained": False, "answer": "anything"}) is False


def test_answerable_correctness_uses_exact_match_not_old_partial_overlap():
    row = {"answerable": True, "expected_answer": "Maya Rao"}
    assert answer_correct(row, {"abstained": False, "answer": "Maya Rao"}) is True
    assert answer_correct(row, {"abstained": False, "answer": "Maya"}) is False


def test_fact_aware_match_allows_supported_explanatory_text():
    assert fact_aware_match("2016", "Helio Logistics was founded in 2016.") is True
    assert fact_aware_match("2016", "Helio Logistics was founded in 2018.") is False


def test_fact_aware_match_handles_currency_and_rounded_percentages():
    assert fact_aware_match("$24.7 million", "$24.7M") is True
    assert fact_aware_match("Approximately 36.6%", "36.59%") is True


def test_grounded_negative_requires_negative_language_and_evidence():
    row = {
        "answerable": False,
        "evidence_keywords": ["does not disclose 2025 revenue"],
    }
    result = {
        "abstained": False,
        "answer": "The document does not disclose 2025 revenue.",
        "citations": [{"text": "The company does not disclose 2025 revenue."}],
    }
    assert is_grounded_negative(row, result) is True
    assert evaluation_state(row, result) == "grounded_negative"


def test_fabricated_unanswerable_answer_is_unsupported():
    row = {"answerable": False, "evidence_keywords": ["does not disclose 2025 revenue"]}
    result = {
        "abstained": False,
        "answer": "The 2025 revenue was $50 million.",
        "citations": [{"text": "The company does not disclose 2025 revenue."}],
    }
    assert evaluation_state(row, result) == "unsupported"
