#!/usr/bin/env python3
"""Sanitize an authorized label<TAB>text file and create stratified JSONL splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path


PHONE = re.compile(r"(?<!\d)1\d{10}(?!\d)")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
LONG_ID = re.compile(r"(?<!\d)\d{12,}(?!\d)")


def sanitize(text: str) -> str:
    text = PHONE.sub("[PHONE]", text)
    text = EMAIL.sub("[EMAIL]", text)
    return LONG_ID.sub("[ID]", text).strip()


def load_records(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if line_number == 1 and raw.lower().startswith("label\t"):
            continue
        if "\t" not in raw:
            raise ValueError(f"line {line_number}: expected label<TAB>text")
        label, text = raw.split("\t", 1)
        label, text = label.strip(), sanitize(text)
        if label not in {"safe", "risk"}:
            raise ValueError(f"line {line_number}: label must be safe or risk")
        if not text:
            continue
        digest = hashlib.sha256(f"{label}\0{text}".encode()).hexdigest()
        if digest not in seen:
            seen.add(digest)
            records.append({"label": label, "text": text})
    return records


def stratified_split(records: list[dict[str, str]], valid_rate: float, seed: int):
    if not 0 < valid_rate < 1:
        raise ValueError("valid_rate must be between 0 and 1")
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        groups[record["label"]].append(record)

    train: list[dict[str, str]] = []
    valid: list[dict[str, str]] = []
    for group in groups.values():
        rng.shuffle(group)
        valid_count = max(1, round(len(group) * valid_rate))
        valid.extend(group[:valid_count])
        train.extend(group[valid_count:])
    rng.shuffle(train)
    rng.shuffle(valid)
    return train, valid


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for index, record in enumerate(records, 1):
            output.write(json.dumps({"id": index, **record}, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--valid-output", type=Path, required=True)
    parser.add_argument("--valid-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train, valid = stratified_split(load_records(args.input), args.valid_rate, args.seed)
    write_jsonl(args.train_output, train)
    write_jsonl(args.valid_output, valid)
    print(json.dumps({"train": len(train), "valid": len(valid)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
