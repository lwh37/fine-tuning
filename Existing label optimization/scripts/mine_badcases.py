#!/usr/bin/env python3
"""Extract sanitized false negatives and false positives from cascade output."""

import argparse
import json
from pathlib import Path

from evaluate_cascade import load_records
from sanitize_and_split import sanitize


def mine(records, threshold):
    badcases = []
    for record in records:
        prediction = (
            record["llm_prediction"]
            if float(record["textcnn_score"]) >= threshold
            else "safe"
        )
        truth = record["truth"]
        if prediction == truth:
            continue
        item = {
            "id": str(record.get("id", "")),
            "text": sanitize(record.get("text", "")),
            "truth": truth,
            "prediction": prediction,
            "error_type": "false_negative" if truth == "risk" else "false_positive",
            "textcnn_score": float(record["textcnn_score"]),
        }
        badcases.append(item)
    return badcases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args()
    badcases = mine(load_records(args.input), args.threshold)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in badcases:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    counts = {
        "false_negative": sum(x["error_type"] == "false_negative" for x in badcases),
        "false_positive": sum(x["error_type"] == "false_positive" for x in badcases),
    }
    print(json.dumps({"badcases": len(badcases), **counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
