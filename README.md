# HEVA AI

HEVA AI is a local, evaluation-oriented document-grounded question-answering system. It ingests PDF, Markdown, and text documents, retrieves evidence with deterministic TF-IDF search, asks a configured LLM to answer from that evidence, returns citations and confidence, and measures answer quality, retrieval, grounding, abstention, calibration, latency, and adversarial robustness.

## What HEVA AI Does

```text
Document -> ingestion -> cleaning/chunking -> TF-IDF index
         -> evidence retrieval -> LLM answer generation
         -> citations/confidence/abstention -> quantitative evaluation
```

Unlike a normal RAG chatbot, HEVA records an evaluation trace for every request. It tests whether the right evidence was retrieved, whether the answer is supported, whether the system abstains when appropriate, how confidence behaves, and how the system responds to adversarial questions.

## Key Features

- PDF, Markdown, and UTF-8 text ingestion.
- Deterministic overlapping chunking and TF-IDF cosine retrieval.
- Configurable top-k retrieval and retrieval-score abstention.
- LLM answers with citations, confidence, and an abstained flag.
- Backward-compatible `/qa` and evaluation-only `/qa/eval` retrieval traces.
- Answer, retrieval, grounding, hallucination, abstention, calibration, latency, adversarial, regression, and Markdown report tooling.

## How HEVA Is Evaluated

Answer metrics measure generated responses. Strict exact match is intentionally exact after normalization; fact-aware accuracy permits supported explanatory wording. Retrieval metrics measure evidence retrieval, not answer correctness. Grounding metrics measure deterministic support signals, not human claim-level faithfulness. Adversarial metrics measure robustness rather than ordinary QA accuracy.

### Answer Quality

Exact Match, fact-aware accuracy, token precision, recall, F1, BLEU, ROUGE-1/2/L, and lexical TF-IDF similarity are computed for answerable cases. Unanswerable cases are evaluated through abstention and grounded-negative states.

### Retrieval Quality

Hit@1, Hit@3, Hit@5, MRR, context precision, and context recall use `evidence_keywords`. They are transparent keyword-grounded proxies, not human-annotated gold retrieval chunks.

### Grounding, Abstention, Calibration, and Performance

The deterministic grounding score combines answer-to-citation lexical support, keyword coverage, numeric support, and the hallucination signal. The evaluator records both LLM `model_confidence` and deterministic `heva_confidence`; ECE and Brier use HEVA confidence. Latency is client-observed request time and includes average, median, P95, and P99 where available.

## Evaluation Formulas

The evaluator uses the following formulas. Unless noted otherwise, metrics are calculated per case and then averaged across the relevant set of cases. Answer-quality metrics are calculated for answerable cases; unanswerable cases are evaluated through abstention and grounded-negative states.

### Text normalization and answer quality

Text is lowercased, punctuation is removed, Unicode punctuation is normalized, and repeated whitespace is collapsed. Numbers, decimals, currencies, percentages, and common magnitude suffixes are preserved.

- **Exact match**

  `ExactMatch = 1` when `normalize(reference) = normalize(prediction)`, otherwise `0`.

- **Token precision and recall**

  Let `O` be the multiset overlap between reference and prediction tokens, `P` the number of predicted tokens, and `R` the number of reference tokens:

  `Precision = |O| / P`

  `Recall = |O| / R`

- **Token F1**

  `F1 = 2 × Precision × Recall / (Precision + Recall)`

  F1 is `0` when both precision and recall are `0`.

- **Fact-aware accuracy**

  An answerable case is correct when it is not abstained, is not flagged as a hallucination, contains every required reference token, and contains every reference numeric value within the implemented rounding tolerance. Explanatory wording around the required facts is allowed. For an unanswerable case, correctness requires abstention or a grounded negative answer.

- **BLEU and ROUGE**

  BLEU is the normalized SacreBLEU sentence score with exponential smoothing. ROUGE-1, ROUGE-2, and ROUGE-L use F1 scores over unigram, bigram, and longest-common-subsequence overlap respectively.

- **Lexical semantic similarity**

  The reference and prediction are converted to TF-IDF vectors and compared with cosine similarity:

  `cosine(A, B) = (A · B) / (||A|| × ||B||)`

### TF-IDF retrieval

For each document chunk, TF-IDF weights are calculated for terms and queried with the same vocabulary. The query and each chunk are represented as vectors, then ranked by cosine similarity:

`score(query, chunk) = cosine(TFIDF(query), TFIDF(chunk))`

Only the top `TOP_K` non-zero-score chunks are returned. Retrieval abstention occurs when no chunks are returned or when the top score is below `ABSTAIN_SCORE_THRESHOLD`.

- **Hit@k**: `1` if all evidence keywords are covered by the first `k` retrieved chunks, otherwise `0`.
- **MRR**: `1 / rank` of the first prefix containing all evidence keywords; `0` if no prefix through rank 5 contains them.
- **Context precision**: number of the first five chunks containing at least one evidence keyword divided by the number of returned chunks considered, up to five.
- **Context recall**: number of distinct evidence keywords found in all retrieved chunks divided by the number of distinct evidence keywords.

These retrieval metrics use `evidence_keywords` as transparent lexical proxies; they are not human-annotated gold chunk labels.

### Grounding, hallucination, and HEVA confidence

- **Semantic support** is the maximum TF-IDF cosine similarity between the answer and its cited chunk texts.
- **Hallucination signal** is raised when a low-support or invented-fact signal is present together with missing evidence keywords, unsupported numeric values, or unsupported capitalized entities. It is a deterministic heuristic, not a human faithfulness judgment.
- **Deterministic grounding score**:

  `Grounding = 0.5 × SemanticSupport + 0.3 × KeywordCoverage + 0.2 × NumericSupport`

  `KeywordCoverage = covered evidence keywords / total evidence keywords`.

  `NumericSupport = 1` when no answer number is absent from the citations, otherwise `0`. A hallucination forces the final grounding score to `0`; the result is clipped to `[0, 1]`.

- **HEVA confidence**:

  `RetrievalSignal = min(1, Top1Relevance / 0.30)`

  `SupportSignal = min(1, max(0, SemanticSupport))`

  `GroundingSignal = 0` for a hallucination, otherwise `1`.

  `HEVAConfidence = ModelConfidence × (0.4 + 0.6 × RetrievalSignal) × (0.4 + 0.6 × SupportSignal) × GroundingSignal`

  The result is clipped to `[0, 1]`. Abstentions receive confidence `0`.

### Abstention, calibration, and latency

- **Abstention precision**: correct abstentions divided by all predicted abstentions.
- **Abstention recall**: correct abstentions divided by all expected abstentions.
- **Abstention F1**: `2 × AbstentionPrecision × AbstentionRecall / (AbstentionPrecision + AbstentionRecall)`.
- **Expected Calibration Error (ECE)**: confidence values are divided into 10 bins. For each bin `b`:

  `ECE = Σ_b (n_b / N) × |Accuracy_b − MeanConfidence_b|`

- **Brier score**:

  `Brier = (1 / N) × Σ_i (HEVAConfidence_i − Correct_i)²`

  where `Correct_i` is `1` for a fact-aware correct case and `0` otherwise. Lower ECE and Brier are better.

- **Latency** is measured around each client HTTP request with a monotonic timer. The report includes arithmetic mean, median, P95, and P99 percentiles over recorded request latencies.

## Demo

The local frontend is available at `http://127.0.0.1:8000/` after starting the API server.

### Application interface

![HEVA AI user interface](assets/UI.png)

### Uploading a document

![HEVA AI document upload](assets/Upload.png)

### Asking a question and viewing citations

![HEVA AI question and answer view](assets/QnA.png)

## Current Evaluation Results

Latest verified local results using Ollama with `qwen2.5-coder:14b`. Local model output and latency may vary between runs.

### Ground Truth - 60 cases

| Metric | Result |
|---|---:|
| Strict accuracy | 35.0% |
| Fact-aware accuracy | 56.7% |
| Hallucination rate | 30.0% |
| Abstention rate | 11.7% |
| Deterministic grounding | 0.392 |
| Hit@1 / Hit@3 / Hit@5 | 90.0% / 98.3% / 98.3% |
| MRR | 0.942 |
| Context precision / recall | 66.7% / 98.3% |
| Average top-1 retrieval score | 0.251 |
| Retrieval abstentions | 1 |
| Average latency | 1,713 ms |

### Adversarial - 122 cases

| Metric | Result |
|---|---:|
| Strict accuracy | 13.9% |
| Fact-aware accuracy | 45.1% |
| Hallucination rate | 27.9% |
| Abstention rate | 4.9% |
| Deterministic grounding | 0.418 |
| Hit@1 / Hit@5 | 85.2% / 97.5% |
| MRR | 0.914 |
| Average latency | 1,746 ms |

| Adversarial category | Fact-aware accuracy |
|---|---:|
| Instruction injection | 14.3% |
| Irrelevant context | 62.9% |
| Paraphrase | 60.0% |
| Subtle factual errors | 41.2% |

## Tech Stack

- Python, FastAPI, and Uvicorn.
- PyMuPDF (`fitz`) for PDF extraction.
- scikit-learn TF-IDF and cosine similarity for retrieval and lexical support.
- Ollama by default with `qwen2.5-coder:14b`; Gemini is an optional provider and optional grounding judge.
- Pydantic/Pydantic Settings, pytest, SacreBLEU, `rouge-score`, and matplotlib.

## Project Requirements

Use Python 3.10+ recommended, install `requirements.txt`, and run Ollama with `qwen2.5-coder:14b` for the default configuration. Gemini is optional unless `LLM_PROVIDER=gemini` is selected or the optional judge is run. The Python commands work on Windows, Linux, and macOS.

## Installation

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

## Environment Variables

Copy `.env.example` to `.env`. Defaults are suitable for local Ollama.

| Variable | Required | Purpose |
|---|---|---|
| `LLM_PROVIDER` | Optional | `ollama` (default) or `gemini`. |
| `OLLAMA_BASE_URL` | For Ollama | Ollama URL, normally `http://127.0.0.1:11434`. |
| `OLLAMA_MODEL` | For Ollama | Local model, normally `qwen2.5-coder:14b`. |
| `GEMINI_API_KEY` | For Gemini | Gemini credential; never commit a real value. |
| `GEMINI_MODEL` | Optional | Gemini model name. |
| `LLM_FALLBACK_ENABLED` | Optional | Defined fallback setting; automatic provider fallback is not currently implemented by `LLMClient`. |
| `LLM_FALLBACK_PROVIDER` / `LLM_FALLBACK_MODEL` | Optional | Defined fallback settings; not used by the current request path. |
| `LLM_TEMPERATURE` / `LLM_SEED` | Optional | Generation controls; defaults `0` and `42`. |
| `TOP_K` | Optional | Retrieved chunk count; default `5`. |
| `ABSTAIN_SCORE_THRESHOLD` | Optional | Minimum top retrieval score; default `0.08`. |

## Running HEVA Locally

Start Ollama and ensure the model exists:

```bash
ollama serve
ollama pull qwen2.5-coder:14b
```

If the `ollama` CLI is not on `PATH`, start Ollama through its installed desktop/service launcher and verify `http://127.0.0.1:11434/api/tags` instead. The API and model availability, rather than the CLI itself, are what HEVA requires.

In another terminal, start the API:

```bash
python -m uvicorn app.main:app --reload --port 8000
```

Verify it:

```bash
curl http://127.0.0.1:8000/health
```

Upload a document:

```bash
curl -X POST http://127.0.0.1:8000/documents -F "file=@data/eval_reference.md"
```

Ask a question:

```bash
curl -X POST http://127.0.0.1:8000/qa -H "Content-Type: application/json" -d '{"question":"Who founded Helio Logistics?"}'
```

PowerShell alternative:

```powershell
$body = @{ question = "Who founded Helio Logistics?" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/qa -ContentType "application/json" -Body $body
```

The response contains `answer`, `confidence`, `abstained`, `citations`, and `model`. Citations contain chunk ID, document ID, source text, and relevance.

## Quick Start

```bash
python -m pip install -r requirements.txt
cp .env.example .env
ollama serve
python -m uvicorn app.main:app --reload --port 8000
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/documents -F "file=@data/eval_reference.md"
curl -X POST http://127.0.0.1:8000/qa -H "Content-Type: application/json" -d '{"question":"Who founded Helio Logistics?"}'
python -m pytest -q
python -m eval.runner --provider ollama --base http://127.0.0.1:8000
python -m eval.runner --provider ollama --adversarial --no-upload --base http://127.0.0.1:8000
python -m eval.report_phase4
```

Generate the benchmark audit reports and deterministic failure clusters:

```bash
python -m eval.audit --results eval/results/ground_truth_results.jsonl
```

Audit the benchmark and generate the dataset/adversarial/failure audit files with:

```bash
python -m eval.audit --annotate
```

## Choosing the LLM Provider

The evaluation harness is shared: both providers use the same document, retrieval, prompt construction, datasets, answer normalization, metrics, and reports. Only generation changes.

### Ollama

```bash
python -m eval.runner --provider ollama
python -m eval.runner --provider ollama --model qwen2.5-coder:14b --limit 3
```

The default model is `qwen2.5-coder:14b`.

### Gemini

Set `GEMINI_API_KEY` in `.env`, then run `python -m eval.runner --provider gemini`. The default is `GEMINI_MODEL` (currently `gemini-3.6-flash`). A missing key fails clearly; Gemini never silently falls back to Ollama.

```bash
python -m eval.runner --provider gemini --limit 3
python -m eval.runner --provider gemini --model gemini-3.6-flash --cases GT001,GT015,GT023
```

Use `--limit N`, `--start N`, and `--cases ID1,ID2`. `--cases` takes precedence over `--limit`. Add `--adversarial` for the adversarial dataset.

```bash
python -m eval.runner --provider ollama --limit 3
python -m eval.runner --provider gemini --limit 3
python -m eval.runner --provider gemini --cases GT001,GT015,GT023
python -m eval.runner --provider ollama
python -m eval.runner --provider gemini
python -m eval.runner --provider ollama --adversarial --limit 10
python -m eval.runner --provider gemini --adversarial --limit 10
```

## Running Evaluation

Start the API first. The ground-truth command uploads `data/eval_reference.md`; the adversarial command reuses that in-memory document.

```bash
python -m eval.runner dataset/ground_truth.jsonl --base http://127.0.0.1:8000
python -m eval.runner dataset/adversarial.jsonl --no-upload --base http://127.0.0.1:8000
python -m pytest -q
python -m eval.report_phase4
```

The optional Gemini judge is non-blocking and can be requested with `--judge`. Save/check the Phase 4 baseline with:

```bash
python -m eval.regression save --results eval/results/ground_truth_results.jsonl
python -m eval.regression check --results eval/results/ground_truth_results.jsonl
```

## Results & Reports

- `eval/results/ground_truth_<provider>_<model>.jsonl`: provider-specific ground-truth results.
- `eval/results/adversarial_<provider>_<model>.jsonl`: provider-specific adversarial results.
- `eval/regression_baseline.json`: baseline metrics and case snapshots.
- `reports/evaluation_report.md`: generated on demand by `python -m eval.report_phase4`. Pass `--ground-truth`, `--adversarial`, and `--output` to generate a provider-specific report without overwriting another run.

## Reproducing the Evaluation

Install dependencies, copy `.env.example` to `.env`, start Ollama with the configured model, start the API on port 8000, run ground truth, run adversarial with `--no-upload`, run tests, generate the report, then save/check the baseline. Results are model- and hardware-dependent.

## Limitations

- Qwen2.5-Coder 14B is used locally through Ollama by default.
- TF-IDF retrieval is lexical and may miss semantic matches.
- Context precision/recall use `evidence_keywords`, not human-annotated gold chunks.
- Deterministic grounding is not human claim-level faithfulness.
- Adversarial results describe this fixed benchmark only.
- Confidence calibration is heuristic unless independently calibrated.
- Local model load and hardware affect latency and output variability.

## Repository Documentation

For the complete architecture, repository structure, file-by-file responsibilities, data flow, evaluation methodology, and implementation details, see `docs/ARCHITECTURE.md`.
