"""Tests for the local setup files."""
import os
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements-local.txt"
SETUP = ROOT / "setup_local.sh"


class LocalSetupTest(unittest.TestCase):
    def test_requirements_pin_known_good_versions(self):
        text = REQUIREMENTS.read_text(encoding="utf-8")
        for pin in ("numpy==1.26.4", "paddlepaddle==3.2.0", "paddleocr==3.2.0"):
            self.assertIn(pin, text)
        self.assertNotIn("opencv-python-headless", text)

    def test_setup_script_is_executable_and_complete(self):
        self.assertTrue(SETUP.exists())
        self.assertTrue(os.access(SETUP, os.X_OK))
        text = SETUP.read_text(encoding="utf-8")
        for needle in ("python3.12", ".venv312", "requirements-local.txt",
                       "ipykernel install", "MarksOCR (local)"):
            self.assertIn(needle, text)
        subprocess.run(["bash", "-n", str(SETUP)], check=True)

    def test_readme_documents_local_run(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        for needle in ("Run locally", "setup_local.sh", "MARKSOCR_INPUT",
                       "bruh_local.ipynb"):
            self.assertIn(needle, text)


if __name__ == "__main__":
    unittest.main()
