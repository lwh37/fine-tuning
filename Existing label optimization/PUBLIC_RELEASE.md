# Public release manifest

## Included

- Project README and dataset card.
- Synthetic TSV/JSONL examples.
- Sanitization, stratified split, Bad Case mining and cascade evaluation scripts.
- Example cascade configuration and standard-library tests.
- GRPO experiment report, public-safe training entry point and example configuration.
- Synthetic Hard Pool / rollout data plus SFT-RL view and reward-diagnostic utilities.

## Excluded

- Original XLSX, CSV, TXT and model-training JSON files.
- User-authored text, contact details, account/session identifiers and internal taxonomy codes.
- Internal service names, URLs, API keys, task IDs and absolute workstation paths.
- Model checkpoints, vocabularies, embeddings, notebooks, caches and archives.
- Third-party source trees and original Git history.
- Per-record silver-label adjudication output, private prompts, experiment checkpoints and TensorBoard logs.

The release is a reproducible, public-safe method package rather than a copy of the private corpus.
