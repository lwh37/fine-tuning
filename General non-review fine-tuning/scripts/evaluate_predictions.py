#!/usr/bin/env python3
"""Report binary classification metrics from JSONL records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate(path: Path, truth_key: str, prediction_key: str) -> dict[str, float | int]:
    tp = tn = fp = fn = 0
    with path.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            truth, prediction = record[truth_key], record[prediction_key]
            if truth not in {"是", "否"} or prediction not in {"是", "否"}:
                raise ValueError("truth and prediction must be 是 or 否")
            if truth == "否" and prediction == "否":
                tp += 1
            elif truth == "是" and prediction == "是":
                tn += 1
            elif truth == "是" and prediction == "否":
                fp += 1
            else:
                fn += 1

    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "records": total,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision_risk": precision,
        "recall_risk": recall,
        "f1_risk": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--truth-key", default="output")
    parser.add_argument("--prediction-key", default="prediction")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.predictions, args.truth_key, args.prediction_key), indent=2))


if __name__ == "__main__":
    main()
