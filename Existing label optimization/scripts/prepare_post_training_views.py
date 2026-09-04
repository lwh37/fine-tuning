#!/usr/bin/env python3
"""Build stratified, ID-matched SFT and RL views from a sanitized Hard Pool."""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ALLOWED_BUCKETS = {"qwen_fn", "qwen_fp", "boundary_replay", "positive_replay"}


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {"sample_id", "text", "silver_label", "bucket"}
            missing = required - row.keys()
            if missing:
                raise ValueError(f"line {number}: missing {sorted(missing)}")
            if row["silver_label"] not in {"safe", "risk"}:
                raise ValueError(f"line {number}: invalid silver_label")
            if row["bucket"] not in ALLOWED_BUCKETS:
                raise ValueError(f"line {number}: invalid bucket")
            rows.append(row)
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("sample_id must be unique")
    return rows


def stratified_split(rows, valid_rate, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[row["bucket"]].append(row)
    rng = random.Random(seed)
    train, valid = [], []
    for bucket in sorted(groups):
        items = groups[bucket]
        rng.shuffle(items)
        valid_size = max(1, round(len(items) * valid_rate))
        valid.extend(items[:valid_size])
        train.extend(items[valid_size:])
    rng.shuffle(train)
    rng.shuffle(valid)
    return train, valid


def risk_response(row):
    if row["silver_label"] == "safe":
        return "无风险"
    mapping = {
        "abuse": "辱骂",
        "unfriendly": "不友好",
        "vulgar": "低俗",
        "threat": "威胁恐吓",
    }
    return mapping.get(row.get("risk_subtype"), "不友好")


def sft_view(row):
    return {
        "id": row["sample_id"],
        "instruction": "判断下面文本是否存在辱骂、不友好、低俗或威胁风险，只输出风险标签或无风险。\n待审核文本：" + row["text"],
        "input": "",
        "output": risk_response(row),
        "bucket": row["bucket"],
    }


def rl_view(row):
    sft = sft_view(row)
    return {
        "sample_id": sft["id"],
        "prompt": sft["instruction"],
        "reference_response": sft["output"],
        "binary_label": int(row["silver_label"] == "risk"),
        "bucket": row["bucket"],
    }


def write_json(path, rows):
    Path(path).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--valid-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    if not 0 < args.valid_rate < 1:
        parser.error("--valid-rate must be between 0 and 1")

    rows = read_jsonl(args.input)
    train, valid = stratified_split(rows, args.valid_rate, args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_sft, valid_sft = [sft_view(row) for row in train], [sft_view(row) for row in valid]
    train_rl = [rl_view(row) for row in train]
    if {row["id"] for row in train_sft} != {row["sample_id"] for row in train_rl}:
        raise RuntimeError("SFT and RL train IDs diverged")
    write_json(output / "sft_train.json", train_sft)
    write_json(output / "sft_valid.json", valid_sft)
    write_jsonl(output / "rl_train.jsonl", train_rl)
    report = {
        "pool": len(rows),
        "train": len(train),
        "valid": len(valid),
        "pool_buckets": dict(Counter(row["bucket"] for row in rows)),
        "train_buckets": dict(Counter(row["bucket"] for row in train)),
        "valid_buckets": dict(Counter(row["bucket"] for row in valid)),
        "sft_rl_train_ids_identical": True,
        "sampling": "stratified split only; no subtype or difficulty oversampling",
    }
    (output / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
