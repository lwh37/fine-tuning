from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_cascade import evaluate  # noqa: E402
from sanitize_and_split import load_records, sanitize, stratified_split  # noqa: E402


class PipelineTest(unittest.TestCase):
    def test_sanitize_and_stratified_split(self) -> None:
        self.assertEqual("联系[PHONE]或[EMAIL]，编号[ID]", sanitize("联系13800138000或a@example.com，编号123456789012"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.tsv"
            path.write_text(
                "label\ttext\n"
                "safe\t正常示例一\n"
                "safe\t正常示例二\n"
                "risk\t风险示例一\n"
                "risk\t风险示例二\n",
                encoding="utf-8",
            )
            train, valid = stratified_split(load_records(path), 0.5, 42)
            self.assertEqual(2, len(train))
            self.assertEqual(2, len(valid))
            self.assertEqual({"safe", "risk"}, {record["label"] for record in valid})

    def test_cascade_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            records = [
                {"truth": "safe", "front_score": 0.1, "llm_prediction": "risk"},
                {"truth": "safe", "front_score": 0.9, "llm_prediction": "safe"},
                {"truth": "risk", "front_score": 0.9, "llm_prediction": "risk"},
                {"truth": "risk", "front_score": 0.2, "llm_prediction": "risk"},
            ]
            path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
            metrics = evaluate(path, 0.85)
            self.assertEqual(0.75, metrics["accuracy"])
            self.assertEqual(0.5, metrics["risk_recall"])
            self.assertEqual(0.5, metrics["llm_route_rate"])


if __name__ == "__main__":
    unittest.main()
