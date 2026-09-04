#!/usr/bin/env python3
"""Business binary reward and rollout-variance diagnostics for short-label GRPO."""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

RISK_LABELS = ("辱骂", "不友好", "低俗", "威胁恐吓")


def parse_binary_label(text):
    value = str(text).strip()
    if "无风险" in value or value.lower() == "safe":
        return 0
    if any(label in value for label in RISK_LABELS) or value.lower() == "risk":
        return 1
    return None


def binary_reward(response, binary_label):
    prediction = parse_binary_label(response)
    return 1.0 if prediction is not None and prediction == int(binary_label) else -1.0


def population_std(values):
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def summarize(rows, group_size):
    states = Counter()
    by_bucket = defaultdict(Counter)
    reward_stds = []
    invalid = responses = 0
    for row in rows:
        outputs = row["responses"]
        if len(outputs) != group_size:
            raise ValueError(f"{row.get('sample_id')}: expected {group_size} responses")
        rewards = [binary_reward(output, row["binary_label"]) for output in outputs]
        correct = sum(reward > 0 for reward in rewards)
        state = "all_wrong" if correct == 0 else "all_correct" if correct == group_size else "mixed"
        states[state] += 1
        by_bucket[row.get("bucket", "unknown")][state] += 1
        reward_stds.append(population_std(rewards))
        invalid += sum(parse_binary_label(output) is None for output in outputs)
        responses += len(outputs)
    prompts = len(rows)
    return {
        "prompts": prompts,
        "group_size": group_size,
        "all_wrong_ratio": states["all_wrong"] / prompts if prompts else 0,
        "mixed_ratio": states["mixed"] / prompts if prompts else 0,
        "all_correct_ratio": states["all_correct"] / prompts if prompts else 0,
        "zero_variance_ratio": (states["all_wrong"] + states["all_correct"]) / prompts if prompts else 0,
        "mean_reward_std": sum(reward_stds) / prompts if prompts else 0,
        "invalid_response_ratio": invalid / responses if responses else 0,
        "state_counts": dict(states),
        "bucket_state_counts": {key: dict(value) for key, value in by_bucket.items()},
    }


def load(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--group-size", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(summarize(load(args.input), args.group_size), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
