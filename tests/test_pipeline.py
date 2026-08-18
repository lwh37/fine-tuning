from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_adaptive_dataset import build_dataset, load_pool  # noqa: E402
from evaluate_predictions import evaluate  # noqa: E402
from validate_dataset import validate  # noqa: E402


class PipelineTest(unittest.TestCase):
    def test_build_validate_and_evaluate(self) -> None:
        config = {
            "seed": 7,
            "positive_count": 4,
            "negative_count": 4,
            "cross_category_rate": 1.0,
            "n_weights": {"2": 1.0},
            "k_weights": {"1": 0.5, "2": 0.5},
        }
        white = [f"正常示例{i}" for i in range(8)]
        black = {"a": ["风险示例A1", "风险示例A2"], "b": ["风险示例B1", "风险示例B2"]}
        records = list(build_dataset(config, "审核并输出是或否。", white, black))
        self.assertEqual(8, len(records))
        self.assertEqual({"是", "否"}, {record["output"] for record in records})

        dataset = ROOT / "tests" / ".tmp-dataset.jsonl"
        predictions = ROOT / "tests" / ".tmp-predictions.jsonl"
        try:
            dataset.write_text(
                "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
                encoding="utf-8",
            )
            summary, errors = validate(dataset)
            self.assertEqual([], errors)
            self.assertEqual(8, summary["records"])

            for record in records:
                record["prediction"] = record["output"]
            predictions.write_text(
                "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
                encoding="utf-8",
            )
            self.assertEqual(1.0, evaluate(predictions, "output", "prediction")["accuracy"])
        finally:
            dataset.unlink(missing_ok=True)
            predictions.unlink(missing_ok=True)

    def test_pool_cleaning(self) -> None:
        path = ROOT / "tests" / ".tmp-pool.txt"
        try:
            path.write_text("正常\n正常\nnull\n{\"choices\": []}\n有效\n", encoding="utf-8")
            self.assertEqual(["正常", "有效"], load_pool(path))
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
