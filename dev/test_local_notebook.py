"""Regression tests for bruh_local.ipynb (the non-Colab notebook)."""
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
LOCAL_NB = ROOT / "bruh_local.ipynb"
SOURCE_NB = ROOT / "bruh.ipynb"


def code_cells(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def cell_starting(cells, prefix):
    for source in cells:
        if source.startswith(prefix):
            return source
    raise AssertionError("no cell starts with %r" % prefix)


class LocalNotebookTest(unittest.TestCase):
    def setUp(self):
        self.local = code_cells(LOCAL_NB)
        self.source = code_cells(SOURCE_NB)

    def test_no_colab_or_drive_references(self):
        text = "\n".join(self.local)
        for needle in ("google.colab", "drive.mount", "/content/drive",
                       "paddlepaddle-gpu", "nvidia-smi", "Google Drive", "Colab"):
            self.assertNotIn(needle, text)

    def test_pipeline_cells_are_unchanged(self):
        for prefix in ("# 2. Dynamic mark-sheet pipeline",
                       "# 2b. Glyph-first front end"):
            self.assertEqual(cell_starting(self.local, prefix),
                             cell_starting(self.source, prefix))

    def test_outputs_and_colab_metadata_are_stripped(self):
        nb = json.loads(LOCAL_NB.read_text(encoding="utf-8"))
        for cell in nb["cells"]:
            self.assertNotIn("colab", cell.get("metadata", {}))
            if cell["cell_type"] == "code":
                self.assertEqual(cell.get("outputs"), [])
                self.assertIsNone(cell.get("execution_count"))

    def test_environment_cell_names_the_setup_script(self):
        first = self.local[0]
        self.assertIn("setup_local.sh", first)
        self.assertIn("MarksOCR (local)", first)

    def _run_discovery(self, cwd, env):
        config = cell_starting(self.local, "# 3. Discover local mark sheets")
        namespace = {}
        old = os.getcwd()
        os.chdir(cwd)
        try:
            with mock.patch.dict(os.environ, env, clear=True), \
                    contextlib.redirect_stdout(io.StringIO()):
                exec(compile(config, str(LOCAL_NB), "exec"), namespace)
        finally:
            os.chdir(old)
        return namespace

    def test_discovery_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp).resolve()
            inp = tmp / "Random_Images_Sample"
            (inp / "sub").mkdir(parents=True)
            (inp / "a.jpg").write_bytes(b"x")
            (inp / "sub" / "b.png").write_bytes(b"x")

            ns = self._run_discovery(tmp, {})

            self.assertEqual(ns["INPUT_DIR"], inp)
            self.assertEqual(ns["OUTPUT_DIR"], tmp / "OCR_Results")
            self.assertEqual(ns["DRIVE_FOLDER"], ns["INPUT_DIR"])
            self.assertEqual([p.name for p in ns["image_files"]],
                             ["a.jpg", "b.png"])
            self.assertTrue(ns["OUTPUT_DIR"].is_dir())

    def test_discovery_env_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp).resolve()
            inp = tmp / "other"
            inp.mkdir()
            (inp / "c.bmp").write_bytes(b"x")
            out = tmp / "other_out"

            ns = self._run_discovery(tmp, {
                "MARKSOCR_INPUT": str(inp),
                "MARKSOCR_OUTPUT": str(out),
            })

            self.assertEqual(ns["INPUT_DIR"], inp)
            self.assertEqual(ns["OUTPUT_DIR"], out)
            self.assertEqual(len(ns["image_files"]), 1)

    def test_missing_input_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                self._run_discovery(tmp, {})

    def test_empty_input_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "Random_Images_Sample").mkdir()
            with self.assertRaises(RuntimeError):
                self._run_discovery(tmp, {})


if __name__ == "__main__":
    unittest.main()
