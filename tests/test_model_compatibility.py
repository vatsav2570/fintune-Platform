"""
tests/test_model_compatibility.py
──────────────────────────────────
Unit tests for model compatibility checking.
"""

import unittest
from src.model_management.model_compatibility import (
    check_model_compatibility, estimate_vram_gb, SUPPORTED_ARCHITECTURES
)


class TestCompatibility(unittest.TestCase):
    def test_known_arch_compatible(self):
        cfg = {"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2"}
        report = check_model_compatibility(cfg)
        self.assertTrue(report.compatible)
        self.assertEqual(report.architecture, "Qwen2ForCausalLM")
        self.assertGreater(len(report.lora_targets), 0)

    def test_unknown_arch_incompatible(self):
        cfg = {"architectures": ["SomeExoticModelForCausalLM"], "model_type": "exotic"}
        report = check_model_compatibility(cfg)
        self.assertFalse(report.compatible)
        self.assertGreater(len(report.issues), 0)

    def test_multimodal_incompatible(self):
        cfg = {"architectures": ["LlavaForCausalLM"], "model_type": "llava_vision"}
        report = check_model_compatibility(cfg)
        self.assertFalse(report.compatible)
        self.assertFalse(report.merge_ok)

    def test_missing_architectures(self):
        cfg = {"model_type": "unknown"}
        report = check_model_compatibility(cfg)
        self.assertFalse(report.compatible)

    def test_vram_estimates(self):
        params = 7_000_000_000  # 7B
        qlora = estimate_vram_gb(params, "qlora")
        lora  = estimate_vram_gb(params, "lora")
        full  = estimate_vram_gb(params, "full_sft")
        self.assertLess(qlora, lora)
        self.assertLess(lora, full)

    def test_all_curated_models_have_targets(self):
        for arch in SUPPORTED_ARCHITECTURES:
            cfg = {"architectures": [arch], "model_type": "generic"}
            report = check_model_compatibility(cfg)
            self.assertGreater(len(report.lora_targets), 0, f"No targets for {arch}")


if __name__ == "__main__":
    unittest.main()
