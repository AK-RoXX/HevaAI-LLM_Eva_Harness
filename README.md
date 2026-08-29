# Heva AI Assignment 3 — Adversarially Evaluated Document Q&A

A complete Q&A system + custom adversarial evaluation harness for the Heva AI ML Engineer Assignment 3. The assignment requires a working LLM system, a >=50-case verified ground truth set, hallucination detection, adversarial inputs, calibration testing, failure clustering, regression tests and a structured report. This implementation does not use RAGAS, TruLens or DeepEval.

## Architecture

`Documents -> extraction -> deterministic TF-IDF retrieval -> Gemini structured generation -> answer/confidence/abstention/citations`

The deliberate design choice is **citation-first, abstention-aware grounding**. Retrieved evidence is passed to the model as data, while document instructions are explicitly treated as untrusted content.

## Setup — Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Put your Gemini key in `.env`:

```env
GEMINI_API_KEY=your_key
GEMINI_MODEL=gemini-3.6-flash
LLM_TEMPERATURE=0
LLM_SEED=42
TOP_K=5
ABSTAIN_SCORE_THRESHOLD=0.08
```

## Run the application

```powershell
python -m uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/` for the frontend or `/docs` for Swagger.

## Build adversarial dataset

The human-authored ground truth contains 60 cases in `dataset/ground_truth.jsonl`. The reference document is `data/eval_reference.md`. Generate adversarial variants:

```powershell
python -m eval.adversarial
```

This produces >100 additional cases across irrelevant-context injection, instruction injection, paraphrase/distribution-shift wording and subtle factual errors.

## Run evaluations

Start the API first, then:

```powershell
python -m eval.runner dataset/ground_truth.jsonl
python -m eval.runner dataset/adversarial.jsonl
```

For a quick smoke test:

```powershell
python -m eval.runner dataset/ground_truth.jsonl --limit 10
```

Results are saved under `eval/results/`.

## Regression tests

After establishing a known-good baseline:

```powershell
python -m eval.regression save
```

After a prompt/model/retrieval change:

```powershell
python -m eval.runner dataset/ground_truth.jsonl --no-upload
python -m eval.regression check
```

## Report

```powershell
python -m eval.report
```

The report includes accuracy, abstention rate, hallucination rate, ECE, Brier score, accuracy by input type and preserved failed cases for causal clustering.

## Ground truth methodology

The 60 ground-truth cases are manually authored against the bundled reference document. Each case records the expected answer, answerability, category, difficulty and human-authored evidence keywords. LLM output is never used as ground truth. Adversarial variants inherit their expected behavior from their verified parent case.

## Hallucination detection

The harness uses source-grounded checks rather than simple answer-string equality: it computes semantic similarity between the answer and retrieved evidence using TF-IDF bigrams, checks verified evidence facts, and flags unsupported/invented numerical claims. The resulting signals are retained per test so the evaluator can inspect false positives/negatives. For a stronger second-stage judge, a Gemini claim-verification module can be added without changing the SUT.
