import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "06_clinical_diagnostic_interface.py"


def load_app_module():
    os.environ["BTD_SKIP_MODEL_LOAD"] = "1"
    spec = importlib.util.spec_from_file_location("clinical_interface", APP_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ClinicalInterfaceSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = load_app_module()

    def test_empty_diagnostic_prompts_for_upload(self):
        message, heatmap, detection, probabilities, report = self.app.run_diagnostic([])
        self.assertIn("Upload a brain scan", message)
        self.assertIsNone(heatmap)
        self.assertIsNone(detection)
        self.assertIsNone(probabilities)
        self.assertIsNone(report)

    def test_upload_validation_accepts_supported_image(self):
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            Image.new("RGB", (16, 16), color=(16, 16, 18)).save(tmp.name)
            error, paths = self.app._validate_uploads([tmp])
        self.assertIsNone(error)
        self.assertEqual(len(paths), 1)

    def test_upload_validation_rejects_unsupported_file(self):
        with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
            error, _ = self.app._validate_uploads([tmp])
        self.assertIn("unsupported file type", error)

    def test_status_html_reports_skipped_models(self):
        html = self.app._status_html()
        self.assertIn("Files uploaded", html)
        self.assertIn("Gatekeeper", html)
        self.assertIn("chip-off", html)


if __name__ == "__main__":
    unittest.main()
