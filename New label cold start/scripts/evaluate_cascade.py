#!/usr/bin/env python3
"""Evaluate a front-model-to-LLM cascade from JSONL predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate(path: Path, threshold: float) -> dict[str, float | int]:
    tp = tn = fp = fn = routed = total = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        record = json.loads(line)
        truth = record["truth"]
        if truth not in {"safe", "risk"} or record["llm_prediction"] not in {"safe", "risk"}:
            raise ValueError(f"line {line_number}: labels must be safe or risk")
        score = float(record["front_score"])
        prediction = "safe"
        if score >= threshold:
            routed += 1
            prediction = record["llm_prediction"]
        total += 1
        if truth == "risk" and prediction == "risk":
            tp += 1
        elif truth == "safe" and prediction == "safe":
            tn += 1
        elif truth == "safe":
            fp += 1
        else:
            fn += 1

    return {
        "records": total,
        "accuracy": (tp + tn) / total if total else 0.0,
        "risk_recall": tp / (tp + fn) if tp + fn else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "llm_route_rate": routed / total if total else 0.0,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.predictions, args.threshold), indent=2))


if __name__ == "__main__":
    main()
