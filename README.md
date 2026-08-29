# Heva AI Assignment 3 — Adversarially Evaluated Document Q&A
## **Day 1**

A document question-answering system built for Heva AI (AI ML Eng)Assignment.

The assignment explicitly asks for a working LLM system that accepts text and produces structured output, followed by a custom evaluation harness probing hallucination, adversarial inputs, distribution shift, failure clustering, confidence calibration, and regressions. Pre-built evaluation frameworks such as RAGAS, TruLens, and DeepEval are not used.

## Design choice

The system is **citation-first and abstention-aware** rather than a naive `question -> LLM` pipeline:

1. Documents are chunked with stable IDs.
2. Relevant evidence is retrieved before generation.
3. The model must answer only from supplied evidence.
4. The structured response contains an answer, confidence, citations, and an explicit `abstained` flag.
5. If evidence is insufficient, the system is instructed to abstain instead of filling gaps.

This gives the evaluation harness observable evidence for hallucination and calibration, and creates meaningful adversarial failure modes.

## Stack

- Python 3.11+
- FastAPI
- Pydantic
- PyMuPDF for PDF extraction
- scikit-learn TF-IDF retrieval (deterministic baseline)
- GeminiAI-API (provider can be changed)

## Run

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env  # Windows
# cp .env.example .env  # Linux/macOS
uvicorn app.main:app --reload
```

Then open `http://127.0.0.1:8000/docs`.

## API

- `POST /documents` — upload a `.txt`, `.md`, or `.pdf`
- `POST /qa` — ask a question against the indexed documents
- `GET /documents` — list indexed documents
- `DELETE /documents/{id}` — remove a document

## Next phase

The evaluation harness will be added separately so that the SUT remains clean. It will include a manually verified >=50-case ground truth dataset, adversarial variants, programmatic source-grounded hallucination checks, calibration metrics/curves, failure clustering, and regression testing.
