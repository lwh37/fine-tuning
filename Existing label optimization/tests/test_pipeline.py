import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PipelineTest(unittest.TestCase):
    def test_split_sanitizes_and_preserves_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            train = Path(temp) / "train.jsonl"
            valid = Path(temp) / "valid.jsonl"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/sanitize_and_split.py"),
                    str(ROOT / "examples/synthetic_labeled.tsv"),
                    "--train-output",
                    str(train),
                    "--valid-output",
                    str(valid),
                    "--valid-rate",
                    "0.25",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            rows = [json.loads(line) for path in (train, valid) for line in path.read_text().splitlines()]
            self.assertEqual(len(rows), 8)
            self.assertEqual({row["label"] for row in rows}, {"safe", "risk"})

    def test_evaluation_and_badcase_mining(self):
        evaluated = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/evaluate_cascade.py"),
                str(ROOT / "examples/synthetic_predictions.jsonl"),
                "--threshold",
                "0.60",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        metrics = json.loads(evaluated.stdout)
        self.assertEqual(metrics["total"], 6)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 2)

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "badcases.jsonl"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/mine_badcases.py"),
                    str(ROOT / "examples/synthetic_predictions.jsonl"),
                    "--output",
                    str(output),
                    "--threshold",
                    "0.60",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            badcases = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(badcases), 3)
            self.assertEqual({item["error_type"] for item in badcases}, {"false_negative", "false_positive"})


if __name__ == "__main__":
    unittest.main()
