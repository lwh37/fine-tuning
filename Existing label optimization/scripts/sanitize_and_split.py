#!/usr/bin/env python3
"""Sanitize, deduplicate and stratify an authorized label/text TSV."""

import argparse
import csv
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
LONG_ID_RE = re.compile(r"(?<!\d)\d{8,}(?!\d)")
URL_RE = re.compile(r"https?://\S+")


def sanitize(text):
    text = URL_RE.sub("[URL]", str(text))
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    return LONG_ID_RE.sub("[ID]", text).strip()


def normalized_label(label):
    value = str(label).strip().lower()
    if value in {"pos", "safe", "white", "白", "0"}:
        return "safe"
    if value in {"neg", "risk", "black", "黑", "1"}:
        return "risk"
    raise ValueError(f"unsupported label: {label!r}")


def read_records(path):
    records = []
    seen = set()
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row_number, row in enumerate(reader, 1):
            if not row or all(not cell.strip() for cell in row):
                continue
            if row_number == 1 and row[0].strip().lower() in {"label", "标签"}:
                continue
            if len(row) < 2:
                raise ValueError(f"row {row_number}: expected label and text")
            label = normalized_label(row[0])
            text = sanitize("\t".join(row[1:]))
            if not text:
                continue
            digest = hashlib.sha256(text.encode()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            records.append({"id": digest[:16], "text": text, "label": label})
    return records


def stratified_split(records, valid_rate, seed):
    groups = defaultdict(list)
    for record in records:
        groups[record["label"]].append(record)
    rng = random.Random(seed)
    train, valid = [], []
    for label in sorted(groups):
        items = groups[label]
        rng.shuffle(items)
        count = round(len(items) * valid_rate)
        valid.extend(items[:count])
        train.extend(items[count:])
    rng.shuffle(train)
    rng.shuffle(valid)
    return train, valid


def write_jsonl(path, records):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--valid-output", required=True)
    parser.add_argument("--valid-rate", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.valid_rate < 1:
        parser.error("--valid-rate must be between 0 and 1")
    records = read_records(args.input)
    train, valid = stratified_split(records, args.valid_rate, args.seed)
    write_jsonl(args.train_output, train)
    write_jsonl(args.valid_output, valid)
    print(json.dumps({"total": len(records), "train": len(train), "valid": len(valid)}))


if __name__ == "__main__":
    main()
