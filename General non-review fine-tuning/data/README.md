# Data directory

The production datasets are intentionally excluded from this public repository.

Use only data that you are authorized to process. Before placing local files under `data/raw/`, remove direct and indirect identifiers, review copyright and consent, scan for secrets, and verify that train/evaluation pools do not overlap.

Expected private input layout:

```text
data/raw/
├── white.txt
├── risk_political.txt
├── risk_sexual.txt
└── risk_violence.txt
```

Each text file contains one sample per line. `data/raw/` is ignored by Git.
