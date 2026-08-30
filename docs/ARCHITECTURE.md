# HEVA AI Architecture

## 1. System Overview

HEVA is a local document-grounded Q&A service paired with an evaluation harness. The application stores uploaded document chunks in memory, retrieves them with TF-IDF cosine similarity, and sends the retrieved text to Ollama or Gemini. The evaluator calls the service, records the answer and retrieval trace, and calculates answer, retrieval, grounding, abstention, calibration, latency, and adversarial metrics.

```text
User / evaluator
      |
      v
FastAPI: upload, health, /qa, /qa/eval
      |
      v
DocumentStore: extract -> normalize/chunk -> TF-IDF matrix
      |
      v
QAService: top-k search -> threshold abstention -> evidence
      |
      v
      LLMClient -> LLMProvider: OllamaProvider or GeminiProvider
      |
      v
QAResponse: answer + confidence + abstained + citations
      |
      v
Evaluation runner: trace + metric calculations -> JSONL -> report
```

## 2. Repository Structure

```text
HEVA_qna/
├── app/
│   ├── config.py
│   ├── llm.py
│   ├── main.py
│   ├── models.py
│   ├── service.py
│   └── store.py
├── data/
│   ├── eval_reference.md
│   └── test_document.txt
├── dataset/
│   ├── ground_truth.jsonl
│   └── adversarial.jsonl
├── eval/
│   ├── adversarial.py
│   ├── calibration.py
│   ├── failure_analysis.py
│   ├── judge.py
│   ├── metrics.py
│   ├── regression.py
│   ├── report_phase4.py
│   └── runner.py
├── reports/
├── static/
├── tests/
├── .env.example
├── README.md
├── requirements.txt
└── scripts and supplementary analysis files
```

Important generated artifacts under `eval/results/` are ignored by Git but are produced locally by the evaluator. `reports/` contains committed/generated Markdown and supplementary analysis outputs. `.venv/`, `.env`, caches, and temporary debug files are excluded by `.gitignore`.

## 3. File-by-File Responsibilities

### `app/config.py`

Purpose: Defines settings loaded from `.env` with defaults.

Responsibilities: provider/model selection, Ollama/Gemini connection values, generation controls, top-k retrieval, and abstention threshold.

Dependencies: Pydantic Settings.

### `app/models.py`

Purpose: Pydantic request/response schemas.

Responsibilities: `QARequest`, backward-compatible `QAResponse`, citations, and evaluation-only retrieval trace items.

### `app/store.py`

Purpose: In-memory document storage and lexical retrieval.

Responsibilities: PDF/TXT/Markdown extraction, whitespace normalization, 650-character chunks with 100-character overlap, TF-IDF indexing, cosine search, and document deletion/listing.

Dependencies: PyMuPDF and scikit-learn.

### `app/service.py`

Purpose: Application-level retrieval and QA orchestration.

Responsibilities: top-k search, threshold-based abstention, evidence construction, LLM calls, citations, and evaluation trace creation. `ask()` returns the normal response; `ask_with_trace()` returns the response plus trace.

### `app/llm.py`

Purpose: Provider-independent structured LLM client.

Responsibilities: the common `LLMProvider` contract, `OllamaProvider`, `GeminiProvider`, provider selection, fixed temperature/seed options, JSON schema handling, lenient Ollama JSON fallback parsing, and the grounded system prompt with retrieved-context delimiters.

### `app/main.py`

Purpose: FastAPI application entry point.

Endpoints: `/health`, `/documents` GET/POST, `/documents/{document_id}` DELETE, `/qa`, `/qa/eval`, `/`, and FastAPI-generated `/docs`/OpenAPI endpoints.

### `eval/runner.py`

Purpose: Execute a JSONL dataset against the running API.

Responsibilities: optional reference upload, provider/model selection, limit/start/case-ID selection, `/qa/eval` calls, response/error capture, retrieval trace capture, answer metrics, hallucination detection, deterministic grounding, HEVA confidence, optional Gemini judge, latency, and provider-specific JSONL output.

### `eval/metrics.py`

Purpose: Pure metric and evaluation-state calculations.

Responsibilities: answer metrics, strict/fact-aware correctness, abstention states, keyword retrieval metrics, deterministic grounding score, ECE, and Brier.

### `eval/report_phase4.py`

Purpose: Generate the current professional `reports/evaluation_report.md` from both result JSONL files, including ground-truth and adversarial summaries, category breakdowns, and worst cases.

### `eval/adversarial.py`

Purpose: Generate `dataset/adversarial.jsonl` from the ground-truth set using irrelevant-context, instruction-injection, paraphrase, and subtle factual-error transformations.

### `eval/calibration.py`

Purpose: Analyze confidence bins and calibration behavior.

### `eval/failure_analysis.py`

Purpose: Group failed cases into behavior-level failure modes.

### `eval/judge.py`

Purpose: Optional Gemini grounding judge. It is invoked only when requested and is non-blocking for normal evaluation.

### `eval/regression.py`

Purpose: Save a Phase 4 baseline and compare later ground-truth runs. Threshold breaches are reported across multiple metrics; regression failure requires at least two adverse breaches.

### `dataset/ground_truth.jsonl`

Purpose: 60 hand-authored evaluation cases with expected answers, answerability, category, difficulty, and evidence keywords.

### `dataset/adversarial.jsonl`

Purpose: 122 fixed adversarial cases with a `parent_id`, adversarial category, and inherited evaluation metadata.

### `data/eval_reference.md` and `data/test_document.txt`

Purpose: Reference document used for evaluation and a small local document useful for manual smoke tests.

### `tests/`

Purpose: Unit tests for chunking/search, answer metrics, retrieval metrics, grounding, abstention, empty retrieval, and malformed traces.

### `static/`

Purpose: Minimal browser UI for uploading documents and asking questions through `/documents` and `/qa`.

## 4. Document Ingestion Pipeline

The client uploads a file to `POST /documents`. `DocumentStore.extract()` accepts `.pdf`, `.txt`, and `.md`. PDF text is extracted with PyMuPDF; text and Markdown are decoded as UTF-8 with replacement for invalid bytes. Whitespace is collapsed, then text is split into approximately 650-character chunks with 100-character overlap. Chunks receive IDs such as `<document_id>:c0000`. The store keeps document metadata, chunks, a TF-IDF vectorizer, and a sparse matrix in memory; a new upload with a new generated ID does not persist across process restarts.

## 5. Retrieval Pipeline

For a question, `DocumentStore.search()` transforms the query with the document TF-IDF vocabulary and computes cosine similarity against every chunk. Results are ranked descending and limited to `TOP_K` (default 5); zero-score chunks are removed. `QAService` abstains when there are no results or when the top score is below `ABSTAIN_SCORE_THRESHOLD` (default 0.08).

The evaluation-only `/qa/eval` response records each retrieved chunk's ID, rank, score, text, document ID, the threshold, whether retrieval caused abstention, and an abstention reason. The normal `/qa` response remains unchanged.

## 6. Question Answering Pipeline

The question and retrieved evidence are passed to `LLMClient`. Evidence is enclosed in `<retrieved_context>` delimiters and explicitly labeled as untrusted data. The configured provider returns JSON with `answer`, `confidence`, and `abstained`. The service clamps confidence to `[0, 1]`, creates citations from the retrieved chunks, and returns:

```json
{
  "answer": "...",
  "confidence": 0.0,
  "abstained": false,
  "citations": [{
    "chunk_id": "...",
    "document_id": "...",
    "text": "...",
    "relevance": 0.0
  }],
  "model": "ollama/qwen2.5-coder:14b"
}
```

## 7. Grounding and Hallucination

The evaluator first computes answer-to-citation TF-IDF lexical support. Its deterministic grounding score combines that support, evidence-keyword coverage, whether answer numbers appear in citations, and the existing hallucination signal. Hallucination detection checks low lexical support, missing evidence keywords, unsupported numbers, and lightweight unsupported capitalized entity phrases. These are transparent heuristics, not claim-level faithfulness.

When `--judge` is used, `eval/judge.py` can call Gemini to return supported/unsupported judgment data. Judge availability, grounded status, and score fields are retained when available; Gemini failure does not stop ordinary evaluation.

## 8. Evaluation Architecture

```text
JSONL dataset -> eval.runner -> POST /qa/eval -> response + retrieval trace
              -> answer/retrieval/grounding/calibration calculations
              -> eval/results/*.jsonl -> eval.report_phase4 -> Markdown report
```

`eval/runner.py` uses the first evaluation to upload `data/eval_reference.md`; subsequent adversarial runs use `--no-upload` and the already loaded in-memory document. Each result includes the source case, answer, citations, retrieval trace, metrics, grounding, confidence, error/status, and latency.

## 9. Metrics Methodology

### Answer metrics

- Exact Match: normalized reference/prediction string equality. Strict and intentionally unforgiving.
- Token precision: overlapping normalized token count divided by predicted token count.
- Token recall: overlapping normalized token count divided by reference token count.
- F1: harmonic mean of token precision and recall.
- Fact-aware accuracy: requires reference facts/tokens and numeric values, allows supported explanatory wording, and rejects detected hallucinations.
- BLEU: normalized SacreBLEU sentence score with smoothing.
- ROUGE-1/2/L: F-measure from `rouge-score`.
- Lexical TF-IDF similarity: cosine similarity between normalized reference and prediction.

### Retrieval metrics

- Hit@K: all non-empty `evidence_keywords` must be present cumulatively in the top K retrieved chunks.
- MRR: reciprocal of the first rank at which all keywords are covered; zero when not covered.
- Context recall: covered evidence keywords divided by the number of evidence keywords.
- Context precision: retrieved top-5 chunks containing at least one evidence keyword divided by retrieved top-5 chunk count.
- Top-1/top-5 score: mean top result score and mean score of returned top-five chunks.

These retrieval and context metrics are keyword-based proxies, not human gold chunk annotations. Cases without evidence keywords return null for keyword quality metrics.

### Grounding and hallucination

Deterministic grounding is a bounded heuristic combining lexical support, keyword coverage, numeric support, and hallucination status. Hallucination is a boolean signal based on unsupported lexical/evidence/numeric/entity checks. Neither is equivalent to a human claim-level faithfulness assessment.

### Abstention

Abstention is true when the model abstains or retrieval fails the service threshold. The evaluator reports abstention rate, accuracy, precision, recall, F1, false-answer behavior, and grounded-negative states for unanswerable cases.

### Calibration

`model_confidence` is the LLM-reported confidence. `heva_confidence` is deterministic: model confidence multiplied by retrieval, lexical-support, and grounding factors. ECE groups HEVA confidence into ten bins and compares mean confidence with fact-aware accuracy. Brier is the mean squared difference between HEVA confidence and the fact-aware correctness label. HEVA confidence is a heuristic and is not assumed to be a calibrated probability.

### Latency

Latency is measured with `time.perf_counter()` around each evaluator HTTP request. Reports include average, median, P95, and P99 when available; it includes retrieval, model generation, HTTP, and serialization time as observed by the client.

## 10. Ground Truth Dataset

Location: `dataset/ground_truth.jsonl`. It contains 60 cases: 47 answerable and 13 unanswerable in the current dataset. Each record contains `id`, `category`, `question`, `expected_answer`, `answerable`, `evidence_keywords`, and `difficulty`.

Example:

```json
{"id":"GT001","category":"direct_fact","question":"When was Helio Logistics founded?","expected_answer":"2016","answerable":true,"evidence_keywords":["founded in 2016"],"difficulty":"easy"}
```

## 11. Adversarial Dataset

Location: `dataset/adversarial.jsonl`. It contains 122 fixed cases. The generator creates irrelevant-context, instruction-injection, and paraphrase variants from the first 35 ground-truth cases, plus subtle factual-error prompts for selected factual cases. Each generated record carries `parent_id` and `adversarial: true`.

## 12. Regression Testing

The Phase 4 baseline is saved in `eval/regression_baseline.json` and includes case snapshots plus fact-aware accuracy, F1, hallucination rate, grounding, Hit@1, Hit@5, MRR, latency, ECE, and Brier. Use:

```bash
python -m eval.regression save --results eval/results/ground_truth_results.jsonl
python -m eval.regression check --results eval/results/ground_truth_results.jsonl
```

The configured adverse thresholds are 5 percentage points for quality/retrieval/grounding/calibration metrics, 3 percentage points for hallucination rate, and 20% for latency. A check reports all changes and exits failure only when at least two thresholds are breached.

## 13. Testing

Run the complete suite with:

```bash
python -m pytest -q
```

Tests cover deterministic search/chunking, answer metrics, fact-aware matching, grounded negatives, retrieval Hit@K/MRR, context precision/recall, grounding penalties, no keywords, empty retrieval, abstention, and malformed trace handling.

## 14. Configuration

Settings are read from `.env` through Pydantic Settings. The default provider is Ollama, default model `qwen2.5-coder:14b`, temperature 0, seed 42, top-k 5, and threshold 0.08. `LLM_FALLBACK_*` values are defined in configuration but the current `LLMClient` does not implement automatic provider fallback.

## 15. Data Flow Example

1. Upload `data/eval_reference.md` to `/documents`; the store extracts and chunks it.
2. Ask `Who founded Helio Logistics?` through `/qa` or `/qa/eval`.
3. TF-IDF retrieves chunks containing the founding statement.
4. The LLM receives the question and delimited retrieved data.
5. The API returns an answer, model confidence, and citations pointing to chunk IDs.
6. The evaluator compares the answer with the case's expected answer, checks keyword retrieval, computes support/grounding, and writes a JSONL record.

## 16. Extension Points

- Another retrieval method can be added behind `DocumentStore.search()` while preserving the service contract.
- Another LLM provider can implement the `LLMClient` provider branch and return the existing JSON schema.
- Another metric can be added to `eval/metrics.py` and attached to each runner result.
- Another dataset can be passed to `eval.runner` if it follows the case schema.
- Another adversarial category can be added in `eval/adversarial.py` without changing the evaluator contract.
- Another report can consume the existing JSONL result schema.

## 17. Known Limitations

- Qwen2.5-Coder 14B is local through Ollama by default; output quality and runtime are model-dependent.
- TF-IDF is lexical and can miss semantic matches.
- Evidence-keyword retrieval metrics are not human-annotated gold retrieval evaluation.
- Deterministic grounding is not human claim-level faithfulness.
- The adversarial set is a fixed benchmark and does not represent all attacks.
- Confidence calibration is heuristic unless independently calibrated.
- Documents and indexes are in memory and are lost when the API process restarts.
- Local model load and hardware affect latency.
