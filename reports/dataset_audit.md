# Ground-Truth Dataset Audit

This audit is generated from `dataset/ground_truth.jsonl`. It does not claim human verification or gold chunk annotation.

## Coverage

- Total cases: **60**
- Answerable: **47**
- Unanswerable: **13**
- Cases with evidence keywords: **60**
- Cases with explicit gold evidence chunk references: **0**
- Human-verification record: **not present in the repository**

## Category distribution

| Category | Cases |
|---|---:|
| adversarial_false_premise | 4 |
| adversarial_injection | 2 |
| comparison | 2 |
| direct_fact | 4 |
| edge_case | 3 |
| employees | 3 |
| entity | 2 |
| events | 4 |
| financial | 6 |
| irrelevant_context | 2 |
| leadership | 1 |
| location | 2 |
| multi_hop | 3 |
| negation | 2 |
| paraphrase | 4 |
| products | 5 |
| reasoning | 2 |
| synthesis | 1 |
| temporal | 2 |
| unanswerable | 6 |

## Difficulty distribution

| Difficulty | Cases |
|---|---:|
| easy | 20 |
| hard | 16 |
| medium | 24 |

## Annotation checks

- Cases missing required fields: **0**
- Duplicate question texts: **0**
- Potentially weak cases (missing keywords or intent): **0**

## Evidence and rigor assessment

Every case has an expected answer, answerability flag, category, difficulty, and keyword evidence proxy. The reference document supplies the source text, but the dataset does not identify human-annotated gold chunks. Evidence keywords therefore establish lightweight lexical coverage only. Test intent is question-specific and records the capability the case is intended to probe; it is not a claim that the case is independently validated.

Potential weak cases are flagged rather than removed. Indirect arithmetic, false-premise, and unanswerable cases require particular care because a valid response may be a correction or abstention rather than a string equal to the expected answer.
