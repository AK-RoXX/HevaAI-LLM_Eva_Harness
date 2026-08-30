# Adversarial Dataset Audit

This audit classifies the existing 122 cases as stored; it does not relabel or modify the benchmark.

- Total cases: **122**

| Existing category | Cases | Actual role |
|---|---:|---|
| instruction_injection | 35 | Genuine instruction-like text in the question; tests instruction/data separation. |
| irrelevant_context | 35 | Distractor framing; tests relevance filtering, but is not necessarily a security attack. |
| paraphrase | 35 | Linguistic robustness; wording variation, not inherently adversarial. |
| subtle_factual_error | 17 | Factual-error/contradiction pressure; tests resistance to incorrect premises or values. |

## Findings

- Instruction-injection cases containing explicit instruction language: **35 / 35**.
- Linguistic-robustness cases should not be described as security attacks merely because they are in the adversarial file.
- Dedicated contradictory-evidence cases: **0 identified**.
- Dedicated false-premise category in the adversarial file: **0**; false-premise behavior exists in some ground-truth cases and may be embedded in factual-error cases, but is not separately labeled here.
- Multi-document conflict, prompt-exfiltration, tool-use, and poisoning attacks: **not represented**.

## Recommendations

Retain the current fixed benchmark for comparability. For a future benchmark revision, add independently reviewed contradictory-evidence and explicit false-premise subsets, and distinguish linguistic robustness from security attacks in the category field.
