# Failure Mode Analysis

Analyzed **60** successful result records; **26** were fact-aware failures. Clusters are deterministic diagnostic categories, not human adjudications.

| Failure cluster | Count | % of failures | % of evaluated | Representative IDs |
|---|---:|---:|---:|---|
| Grounding failure: unsupported claim signal | 15 | 57.7% | 25.0% | GT011, GT012, GT018 |
| Evaluation or generation failure: answer mismatch despite available evidence | 7 | 26.9% | 11.7% | GT016, GT025, GT034 |
| Generation failure: reasoning, comparison, or qualifier mismatch despite retrieval | 3 | 11.5% | 5.0% | GT027, GT030, GT032 |
| Abstention failure: unnecessary abstention | 1 | 3.8% | 1.7% | GT015 |

## Interpretation

Retrieval success does not imply answer success: cases with Hit@5 can still fail during generation, qualification, false-premise handling, or grounding. Conversely, a retrieval miss is a system-layer failure before generation. The current result schema does not contain human root-cause labels or claim-level annotations, so these clusters are evidence-based heuristics and should not be read as definitive causal diagnoses.
