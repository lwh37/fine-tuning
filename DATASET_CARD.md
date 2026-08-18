# Dataset Card

## Dataset summary

This repository documents a field-adaptive Chinese content-moderation dataset. A record contains one to eight abstract fields. The target is `是` only when every field is compliant; otherwise it is `否`.

The public repository contains metadata, synthetic examples and reproducible construction scripts. It does **not** contain production moderation logs or the full training corpus.

## Task

- Language: Chinese
- Task: binary text classification
- Input: variable-length multi-field text
- Labels: `是` / `否`
- Aggregation: logical OR over field-level risks
- Original model target: Qwen3-Base-4B with LoRA

## Version summary

| Version | Records | Positive | Negative | Main change |
| --- | ---: | ---: | ---: | --- |
| v1 | 50,000 | 25,000 | 25,000 | Three risk pools and controlled field structure |
| v4 | 100,000 | 50,000 | 50,000 | Larger offline pools, low-frequency coverage and shorter instruction |

## Record schema

```json
{
  "id": 1,
  "instruction": "<rules and abstract fields>",
  "input": "",
  "output": "是",
  "task_type": "classification"
}
```

The public builder writes JSONL for streaming and easier validation. Each line follows the schema above.

## Construction

1. Detect source encoding, delimiter, sheet and columns.
2. Normalize empty values and remove malformed/model-response records.
3. Build deduplicated compliant and risk-specific pools.
4. Sample field count `N` from 1–8.
5. For negative records, sample risk-field count `k` subject to `k <= N`.
6. Round-robin all position combinations for each `(N, k)` pair.
7. Rename fields to `字段1…字段N`.
8. Add a strict instruction and `是/否` label.
9. Validate labels, schema, field distribution and duplicate hashes.

## Intended uses

- Research on variable-field binary classification.
- Reproducing controlled sampling and position-invariance experiments.
- Testing data-validation and slice-evaluation pipelines with synthetic or authorized data.

## Out-of-scope uses

- Direct moderation decisions without independent evaluation and human escalation.
- Inferring sensitive attributes or identifying individuals.
- Training from private production data without authorization and privacy review.
- Treating reported accuracy as evidence of universal safety performance.

## Limitations

- Original metrics are accuracy-oriented; Precision, Recall, F1, FPR and FNR should also be reported.
- Risk classes are imbalanced, especially in early versions.
- Abstract field names differ from production field names.
- Higher risk recall can increase false positives on compliant data.
- Synthetic examples do not represent the linguistic diversity of production traffic.

## Privacy and release status

The original sources may contain user-authored text, contact information, internal identifiers and sensitive content. They are not redistributed here. Anyone building a derivative dataset must document consent or lawful basis, remove direct and indirect identifiers, review copyright, define retention controls and perform leakage checks before release.

## License

No license is granted for the original private dataset. A repository-wide code or documentation license must be selected by the owner separately before third-party reuse.
