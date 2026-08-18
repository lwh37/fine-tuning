#!/usr/bin/env python3
"""Evaluate a TextCNN-to-LLM moderation cascade from JSONL predictions."""

import argparse
import json
from pathlib import Path


def load_records(path):
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            required = {"truth", "textcnn_score", "llm_prediction"}
            missing = required - record.keys()
            if missing:
                raise ValueError(f"line {line_number}: missing {sorted(missing)}")
            records.append(record)
    return records


def evaluate(records, threshold):
    tp = fp = tn = fn = routed = 0
    for record in records:
        truth = record["truth"]
        if truth not in {"safe", "risk"}:
            raise ValueError(f"unsupported truth label: {truth!r}")
        if float(record["textcnn_score"]) >= threshold:
            routed += 1
            prediction = record["llm_prediction"]
        else:
            prediction = "safe"
        if prediction not in {"safe", "risk"}:
            raise ValueError(f"unsupported prediction label: {prediction!r}")
        if truth == "risk" and prediction == "risk":
            tp += 1
        elif truth == "safe" and prediction == "risk":
            fp += 1
        elif truth == "safe":
            tn += 1
        else:
            fn += 1
    total = len(records)
    return {
        "total": total,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "route_rate": routed / total if total else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args()
    result = evaluate(load_records(args.input), args.threshold)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
