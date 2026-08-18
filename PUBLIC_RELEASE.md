# Public release manifest

## Included

- Project README and dataset card.
- Synthetic, non-production text pools.
- Dataset construction, validation and metric scripts.
- Example sampling and LoRA training configurations.
- A standard-library test covering the public pipeline.

## Intentionally excluded

- Production moderation logs and user-authored records.
- Spreadsheets, CSV/TSV exports and full JSON training corpora.
- Contact information, user identifiers and internal scene metadata.
- Internal URLs, object-storage links and local absolute paths.
- Model checkpoints, embeddings and offline validation artifacts.
- Archives and files larger than ordinary source-control limits.
- Notebook outputs, caches, temporary files and nested Git history.

## Release boundary

This repository is a method-and-metadata release, not a redistribution of the private dataset. The included examples are synthetic and exist only to verify that the public scripts run end to end.
