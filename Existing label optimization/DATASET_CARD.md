# Dataset Card: Existing Label Optimization

## Summary

This public package documents a Chinese content-moderation workflow for improving an existing abuse/unfriendly-language label under distribution shift. It combines a character-level TextCNN filter with a Qwen3 LoRA semantic reviewer and a targeted Bad Case feedback loop.

## Verified private-data statistics

- TextCNN: 98,144 train, 3,000 validation and 3,000 test records; each split is class-balanced.
- Qwen SFT: 24,532 records across safe, abuse, unfriendly, vulgar and threat labels.
- Offline evaluation: 1,533 risk records and 1,000 random online safe records.
- Reported result on that slice: recall 72.7% to 87.3%, false-positive rate 0.2% after optimization.

## Public contents

- Aggregate statistics and methodology documentation.
- Synthetic TSV and JSONL examples.
- Standard-library sanitization, split, Bad Case mining and cascade evaluation scripts.
- No production text, identifiers, endpoints, checkpoints or full datasets.

## Public label normalization

- `safe`: normal content.
- `risk`: abuse, unfriendly, vulgar or threatening content.

Production taxonomy codes are intentionally omitted. The synthetic prediction schema is:

```json
{"id":"demo-1","text":"<synthetic>","truth":"risk","textcnn_score":0.91,"llm_prediction":"risk"}
```

## Risks and limitations

- Real moderation data may contain personal information and harmful content.
- Synthetic examples are suitable for pipeline tests, not model training.
- Reported metrics are dataset-specific and may not transfer to another traffic distribution.
- A cascade must be evaluated on recall, false-positive rate, routing coverage, latency and cost together.

## Release status

The original dataset is private and not licensed for redistribution. This repository releases methods and synthetic examples only.
