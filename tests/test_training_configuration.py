"""
tests/test_training_configuration.py
──────────────────────────────────────
Unit tests for TrainingConfig validation.
"""

import tempfile
import pathlib
import json
import unittest

from src.training.trainer_interface import TrainingConfig


class TestTrainingConfig(unittest.TestCase):
    def _make_dataset(self) -> str:
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w",
                                        delete=False, encoding="utf-8") as f:
            json.dump({"messages": [
                {"role": "system", "content": "S"},
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A"},
            ]}, f)
            return f.name

    def test_valid_config(self):
        ds = self._make_dataset()
        cfg = TrainingConfig(
            base_model_path="Qwen/Qwen2.5-0.5B-Instruct",
            dataset_path=ds,
            output_dir="/tmp/test_out",
            display_name="test",
            epochs=1,
            lr=2e-4,
            batch_size=1,
            lora_r=8,
        )
        errors = cfg.validate()
        pathlib.Path(ds).unlink()
        self.assertEqual(errors, [])

    def test_missing_model_path(self):
        ds = self._make_dataset()
        cfg = TrainingConfig(
            base_model_path="",
            dataset_path=ds,
            output_dir="/tmp/out",
        )
        errors = cfg.validate()
        pathlib.Path(ds).unlink()
        self.assertTrue(any("base_model_path" in e for e in errors))

    def test_missing_dataset(self):
        cfg = TrainingConfig(
            base_model_path="some/model",
            dataset_path="/nonexistent/path.jsonl",
            output_dir="/tmp/out",
        )
        errors = cfg.validate()
        self.assertTrue(any("dataset" in e.lower() for e in errors))

    def test_serialize_deserialize(self):
        ds = self._make_dataset()
        cfg = TrainingConfig(
            base_model_path="some/model",
            dataset_path=ds,
            output_dir="/tmp/out",
            epochs=5,
            lr=3e-4,
        )
        d = cfg.to_dict()
        cfg2 = TrainingConfig.from_dict(d)
        pathlib.Path(ds).unlink()
        self.assertEqual(cfg2.epochs, 5)
        self.assertAlmostEqual(cfg2.lr, 3e-4)


if __name__ == "__main__":
    unittest.main()
