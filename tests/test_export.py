"""
tests/test_export.py
─────────────────────
Unit tests for export utilities.
"""

import json
import pathlib
import tempfile
import unittest
import zipfile

from src.export.adapter_export import export_adapter
from src.export.full_model_export import export_full_model


def _make_fake_adapter_dir() -> str:
    """Create a minimal fake adapter directory."""
    tmp = tempfile.mkdtemp()
    adapter_dir = pathlib.Path(tmp) / "fake_adapter"
    adapter_dir.mkdir()

    # Minimal adapter files
    (adapter_dir / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": "test/model",
        "peft_type": "LORA",
        "r": 16,
        "lora_alpha": 32,
        "task_type": "CAUSAL_LM",
    }))
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"\x00" * 128)

    return str(adapter_dir)


class TestAdapterExport(unittest.TestCase):
    def test_export_creates_zip(self):
        adapter_dir = _make_fake_adapter_dir()
        with tempfile.TemporaryDirectory() as out_dir:
            ok, result = export_adapter(
                adapter_dir=adapter_dir,
                base_model_id="test/model",
                method="qlora_sft",
                output_dir=out_dir,
            )
            self.assertTrue(ok, f"Export failed: {result}")
            self.assertTrue(result.endswith(".zip"))
            self.assertTrue(pathlib.Path(result).exists())

            with zipfile.ZipFile(result) as zf:
                names = zf.namelist()
                # README should be included
                self.assertIn("README.md", names)
                # Metadata should be included
                self.assertIn("export_metadata.json", names)

    def test_export_missing_dir(self):
        with tempfile.TemporaryDirectory() as out_dir:
            ok, result = export_adapter(
                adapter_dir="/nonexistent/path",
                base_model_id="test/model",
                method="qlora_sft",
                output_dir=out_dir,
            )
            self.assertFalse(ok)
            self.assertIn("not found", result)


class TestFullModelExport(unittest.TestCase):
    def test_export_creates_zip(self):
        with tempfile.TemporaryDirectory() as model_dir:
            # Create fake model files
            pathlib.Path(model_dir + "/config.json").write_text('{"model_type":"test"}')
            pathlib.Path(model_dir + "/tokenizer.json").write_text('{}')

            with tempfile.TemporaryDirectory() as out_dir:
                ok, result = export_full_model(
                    model_dir=model_dir,
                    base_model_id="test/model",
                    output_dir=out_dir,
                )
                self.assertTrue(ok, f"Export failed: {result}")
                self.assertTrue(pathlib.Path(result).exists())

                with zipfile.ZipFile(result) as zf:
                    self.assertIn("README.md", zf.namelist())


if __name__ == "__main__":
    unittest.main()
