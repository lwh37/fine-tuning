#!/usr/bin/env python3
"""Build a field-adaptive binary-classification dataset from authorized text pools."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
from collections import defaultdict, deque
from pathlib import Path


POLLUTION_MARKERS = (
    "chat.completion",
    "completion_tokens",
    "prompt_tokens",
    '"choices"',
)


def load_pool(path: Path) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        value = raw.strip()
        if not value or value.lower() in {"null", "none", "nan", "na", "n/a"}:
            continue
        if any(marker in value for marker in POLLUTION_MARKERS) or value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def weighted_choice(rng: random.Random, weights: dict[int, float]) -> int:
    return rng.choices(list(weights), weights=list(weights.values()), k=1)[0]


class PositionSampler:
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.queues: dict[tuple[int, int], deque[tuple[int, ...]]] = defaultdict(deque)

    def take(self, n: int, k: int) -> tuple[int, ...]:
        key = (n, k)
        if not self.queues[key]:
            combinations = list(itertools.combinations(range(n), k))
            self.rng.shuffle(combinations)
            self.queues[key].extend(combinations)
        return self.queues[key].popleft()


def choose_black_values(
    rng: random.Random,
    pools: dict[str, list[str]],
    count: int,
    force_cross_category: bool,
) -> list[str]:
    categories = [name for name, values in pools.items() if values]
    if not categories:
        raise ValueError("at least one non-empty black pool is required")
    if len({value for values in pools.values() for value in values}) < count:
        raise ValueError(f"black pools need at least {count} unique values")

    selected: list[str] = []
    if force_cross_category and count >= 2 and len(categories) >= 2:
        for category in rng.sample(categories, 2):
            selected.append(rng.choice(pools[category]))

    flattened = [(category, value) for category in categories for value in pools[category]]
    while len(selected) < count:
        value = rng.choice(flattened)[1]
        if value not in selected:
            selected.append(value)
    return selected


def render_fields(values: list[str]) -> str:
    lines = ["待审帖子的各字段："]
    lines.extend(f"字段{index}：{value}" for index, value in enumerate(values, 1))
    return "\n".join(lines)


def build_dataset(
    config: dict,
    prompt: str,
    white_pool: list[str],
    black_pools: dict[str, list[str]],
):
    rng = random.Random(config["seed"])
    n_weights = {int(key): value for key, value in config["n_weights"].items()}
    k_weights = {int(key): value for key, value in config["k_weights"].items()}
    labels = ["是"] * config["positive_count"] + ["否"] * config["negative_count"]
    rng.shuffle(labels)

    if len(white_pool) < max(n_weights):
        raise ValueError("white pool must contain at least max(N) unique values")

    positions = PositionSampler(rng)
    seen_records: set[str] = set()
    for record_id, label in enumerate(labels, 1):
        for _ in range(1000):
            n = weighted_choice(rng, n_weights)
            values = rng.sample(white_pool, n)
            if label == "否":
                feasible_k = {k: weight for k, weight in k_weights.items() if k <= n}
                k = weighted_choice(rng, feasible_k)
                risk_positions = positions.take(n, k)
                black_values = choose_black_values(
                    rng,
                    black_pools,
                    k,
                    rng.random() < config["cross_category_rate"],
                )
                for position, black_value in zip(risk_positions, black_values):
                    values[position] = black_value

            fields = render_fields(values)
            digest = hashlib.sha256(fields.encode("utf-8")).hexdigest()
            if digest in seen_records:
                continue
            seen_records.add(digest)
            yield {
                "id": record_id,
                "instruction": f"{prompt.rstrip()}\n\n{fields}",
                "input": "",
                "output": label,
                "task_type": "classification",
            }
            break
        else:
            raise RuntimeError("unable to generate enough unique records from the supplied pools")


def parse_black_pool(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("black pool must use CATEGORY=PATH")
    category, path = value.split("=", 1)
    return category, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--white", type=Path, required=True)
    parser.add_argument("--black", action="append", type=parse_black_pool, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    white_pool = load_pool(args.white)
    black_pools = {category: load_pool(path) for category, path in args.black}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for record in build_dataset(
            config,
            args.prompt.read_text(encoding="utf-8"),
            white_pool,
            black_pools,
        ):
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
