# Dataset Card: Existing Label Optimization

## Summary

This public package documents a Chinese content-moderation workflow for improving an existing abuse/unfriendly-language label under distribution shift. It combines a character-level TextCNN filter with a Qwen3 LoRA semantic reviewer and a targeted Bad Case feedback loop.

## Verified private-data statistics

- TextCNN: 98,144 train, 3,000 validation and 3,000 test records; each split is class-balanced.
- Qwen SFT: 24,532 records across safe, abuse, unfriendly, vulgar and threat labels.
- Offline evaluation: 1,533 risk records and 1,000 random online safe records.
- Reported result on that slice: recall 72.7% to 87.3%, false-positive rate 0.2% after optimization.
- Post-training study: 2,338 silver-labeled Hard/Replay records, split into 2,104 train and 234 validation IDs.
- Development protocol: 4,000-record Calibration set and a separate 6,510-record balanced sealed Test.
- Final post-training result on the sealed Test: pipeline recall 76.50% to 88.60%; the GRPO-only delta over Hard Case SFT was +0.83 percentage points.

## Public contents

- Aggregate statistics and methodology documentation.
- Synthetic TSV and JSONL examples.
- Standard-library sanitization, split, Bad Case mining and cascade evaluation scripts.
- No production text, identifiers, endpoints, checkpoints or full datasets.
- Synthetic Hard Pool and rollout examples for testing SFT/RL view generation and GRPO reward diagnostics.

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
- Post-training labels are model-adjudicated silver labels without human review; they measure agreement with that verifier and may contain systematic bias.
- The 6,510-record Test is class-balanced, so its precision does not estimate production precision under natural prevalence.
- A cascade must be evaluated on recall, false-positive rate, routing coverage, latency and cost together.

## Release status

The original dataset is private and not licensed for redistribution. This repository releases methods and synthetic examples only.
