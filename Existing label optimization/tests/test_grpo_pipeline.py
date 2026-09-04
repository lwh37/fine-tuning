import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from grpo_reward import binary_reward, summarize


class GRPOPipelineTest(unittest.TestCase):
    def test_business_binary_reward(self):
        self.assertEqual(binary_reward("辱骂", 1), 1.0)
        self.assertEqual(binary_reward("低俗", 1), 1.0)
        self.assertEqual(binary_reward("无风险", 0), 1.0)
        self.assertEqual(binary_reward("无风险", 1), -1.0)
        self.assertEqual(binary_reward("unknown", 0), -1.0)

    def test_rollout_variance_summary(self):
        rows = [
            {"sample_id": "a", "binary_label": 1, "bucket": "qwen_fn", "responses": ["辱骂"] * 8},
            {"sample_id": "b", "binary_label": 1, "bucket": "qwen_fn", "responses": ["无风险"] * 8},
            {"sample_id": "c", "binary_label": 0, "bucket": "qwen_fp", "responses": ["无风险", "辱骂"] * 4},
        ]
        result = summarize(rows, 8)
        self.assertEqual(result["state_counts"], {"all_correct": 1, "all_wrong": 1, "mixed": 1})
        self.assertAlmostEqual(result["zero_variance_ratio"], 2 / 3)

    def test_sft_and_rl_views_share_train_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "views"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/prepare_post_training_views.py"),
                    str(ROOT / "examples/synthetic_hard_pool.jsonl"),
                    "--output-dir",
                    str(output),
                    "--valid-rate",
                    "0.25",
                    "--seed",
                    "42",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            sft = json.loads((output / "sft_train.json").read_text())
            rl = [json.loads(line) for line in (output / "rl_train.jsonl").read_text().splitlines()]
            self.assertEqual({row["id"] for row in sft}, {row["sample_id"] for row in rl})
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertTrue(manifest["sft_rl_train_ids_identical"])


if __name__ == "__main__":
    unittest.main()
