#!/usr/bin/env python3
"""Validate schema, labels, field counts and duplicates in a JSONL dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


FIELD_PATTERN = re.compile(r"^字段(\d+)：", re.MULTILINE)


def validate(path: Path) -> tuple[dict, list[str]]:
    labels: Counter[str] = Counter()
    field_counts: Counter[int] = Counter()
    ids: set[object] = set()
    hashes: set[str] = set()
    errors: list[str] = []

    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSON: {exc.msg}")
                continue

            missing = {"id", "instruction", "input", "output", "task_type"} - record.keys()
            if missing:
                errors.append(f"line {line_number}: missing {sorted(missing)}")
                continue
            if record["id"] in ids:
                errors.append(f"line {line_number}: duplicate id {record['id']}")
            ids.add(record["id"])
            if record["output"] not in {"是", "否"}:
                errors.append(f"line {line_number}: invalid output {record['output']!r}")
            labels[record["output"]] += 1

            numbers = [int(value) for value in FIELD_PATTERN.findall(record["instruction"])]
            if not numbers or numbers != list(range(1, len(numbers) + 1)):
                errors.append(f"line {line_number}: fields must be consecutive from 1")
            field_counts[len(numbers)] += 1

            digest = hashlib.sha256(record["instruction"].encode("utf-8")).hexdigest()
            if digest in hashes:
                errors.append(f"line {line_number}: duplicate instruction")
            hashes.add(digest)

    summary = {
        "records": sum(labels.values()),
        "labels": dict(sorted(labels.items())),
        "field_counts": dict(sorted(field_counts.items())),
        "errors": len(errors),
    }
    return summary, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    summary, errors = validate(args.dataset)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for error in errors[:20]:
        print(error)
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()
