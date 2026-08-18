# Dataset Card: New Label Cold Start

## Summary

This public package documents a cascaded Chinese content-moderation workflow for bootstrapping a newly introduced risk label. It combines a high-recall ModernBERT binary filter with a Qwen3 LoRA semantic reviewer.

## Public contents

- Aggregate dataset statistics.
- Synthetic labeled and prediction examples.
- Standard-library sanitization, stratified split and cascade evaluation scripts.
- No production text, identifiers, endpoints, checkpoints or full datasets.

## Private dataset schema

ModernBERT records:

```json
{"instruction": "<text>", "output": "000"}
```

Qwen3 SFT source records use a closed label followed by a tab and the text. The public scripts convert authorized TSV input to JSONL records with `id`, `text` and `label`.

## Labels

- `safe`: no target risk.
- `risk`: target or configured adjacent risk.
- Production codes and internal taxonomy identifiers are intentionally omitted.

## Risks and limitations

- Real moderation data may contain personal information and harmful content.
- Synthetic examples are suitable for pipeline tests, not model training.
- Reported metrics are dataset-specific and may not transfer to another traffic distribution.
- False-positive rate, routing coverage, latency and inference cost must be evaluated together.

## Release status

The original dataset is private and not licensed for redistribution. This repository releases methods and synthetic examples only.
