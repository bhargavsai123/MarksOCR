# Local Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local (non-Colab) copy of `bruh.ipynb` that runs on this Mac with a Python 3.12 venv and local input/output folders.

**Architecture:** `dev/make_local_notebook.py` reads `bruh.ipynb`, replaces only the install cell, the Google Drive cell and the Linux font lookup, strips Colab outputs, and writes `bruh_local.ipynb`. `setup_local.sh` builds `./.venv312` from `requirements-local.txt` and registers a Jupyter kernel. Regression tests pin the pipeline cells to the original notebook.

**Tech Stack:** Python 3.12, PaddlePaddle 3.2.0 (CPU), PaddleOCR 3.2.0, OpenCV, openpyxl, matplotlib, Jupyter/ipykernel, unittest.

**Spec:** `docs/superpowers/specs/2026-11-01-local-notebook-design.md`

## Global Constraints

- Python 3.12 venv at `./.venv312`. Python 3.14 is rejected: PaddlePaddle 3.2.0 has no cp314 wheel.
- Pins: `numpy==1.26.4`, `paddlepaddle==3.2.0`, `paddleocr==3.2.0`. Do not install `opencv-python-headless` (paddlex pins `opencv-contrib-python==4.10.0.84`).
- The local notebook's code cells must not contain `google.colab`, `drive.mount`, `/content/drive`, `paddlepaddle-gpu`, or `nvidia-smi`.
- Cells starting `# 2. Dynamic mark-sheet pipeline` and `# 2b. Glyph-first front end` must be byte-identical to `bruh.ipynb`.
- The discovery cell must define `INPUT_DIR`, `OUTPUT_DIR`, `image_files`, and `DRIVE_FOLDER = INPUT_DIR` (the batch and Excel cells still use `DRIVE_FOLDER`).
- Input defaults to `./Random_Images_Sample`; output to `./OCR_Results`. Override with `MARKSOCR_INPUT` / `MARKSOCR_OUTPUT`.
- `bruh_local.ipynb` has no stored outputs, no `execution_count`, and no Colab/Kaggle/widget metadata.

## Review Focus

- Wrong kernel (base Python 3.14): the environment cell must raise a message naming `./setup_local.sh`, not a bare `ModuleNotFoundError`.
- Missing input folder: discovery must raise `FileNotFoundError` with the resolved path.
- Empty input folder: discovery must raise `RuntimeError`, not start an empty batch.
- Launching Jupyter outside the repository root: the default input path changes; README documents this and env overrides cover it.
- Paddle import failure on arm64: `setup_local.sh` runs an import smoke test so the failure is visible at setup time, not mid-notebook.
- Downstream `DRIVE_FOLDER` references: the alias must exist or cells 9/11/12 raise `NameError`.

---

### Task 1: Generate `bruh_local.ipynb` and pin it with tests

**Files:**
- Create: `dev/test_local_notebook.py`
- Create: `dev/make_local_notebook.py`
- Create: `bruh_local.ipynb` (generated)
- Read: `bruh.ipynb`

**Interfaces:**
- `dev/make_local_notebook.py` exposes `main()`; running it writes `ROOT / "bruh_local.ipynb"` and prints `wrote <path>`.
- `dev/test_local_notebook.py` reads `ROOT / "bruh.ipynb"` and `ROOT / "bruh_local.ipynb"`.

- [ ] **Step 1: Write the failing tests**

Create `dev/test_local_notebook.py` with exactly:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest dev/test_local_notebook.py -v`

Expected: errors because `bruh_local.ipynb` does not exist yet.

- [ ] **Step 3: Write the generator**

Create `dev/make_local_notebook.py` with exactly:

```python
"""Generate bruh_local.ipynb from bruh.ipynb.

bruh.ipynb stays the source of truth for the pipeline. This script replaces
only the Colab-specific cells, strips stored outputs, and writes the local
notebook. Re-run it after changing the pipeline in bruh.ipynb.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "bruh.ipynb"
TARGET = ROOT / "bruh_local.ipynb"

ENV_CHECK = '''# 0. Local environment check (no installs at runtime)
# Run ./setup_local.sh once, then select the "MarksOCR (local)" Jupyter kernel.
import platform
import sys

if not ((3, 10) <= sys.version_info[:2] < (3, 14)):
    raise RuntimeError(
        "Python %d.%d is not supported by PaddlePaddle 3.2.0. "
        "Use the project venv: source .venv312/bin/activate"
        % sys.version_info[:2]
    )

missing = []
for name in ("cv2", "numpy", "PIL", "openpyxl", "matplotlib", "paddle", "paddleocr"):
    try:
        __import__(name)
    except Exception as exc:
        missing.append("%s (%s)" % (name, exc))
if missing:
    raise RuntimeError(
        "Missing local dependencies:\\n  " + "\\n  ".join(missing) +
        "\\nRun ./setup_local.sh, then select the 'MarksOCR (local)' Jupyter kernel."
    )

import cv2
import matplotlib
import numpy
import openpyxl
import paddle
from PIL import Image, ImageDraw, ImageFont
from paddleocr import TextRecognition

print("Python:", sys.version.split()[0], "|", platform.platform())
print("Paddle:", paddle.__version__, "| CUDA compiled:", paddle.is_compiled_with_cuda())
print("OpenCV:", cv2.__version__, "| NumPy:", numpy.__version__,
      "| openpyxl:", openpyxl.__version__)
print("PaddleOCR device: cpu (local notebook)")
'''

DISCOVERY = '''# 3. Discover local mark sheets
from collections import Counter
from pathlib import Path
import os

PROJECT_DIR = Path.cwd()
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}

INPUT_DIR = Path(os.environ.get("MARKSOCR_INPUT",
                                PROJECT_DIR / "Random_Images_Sample")).expanduser().resolve()
OUTPUT_DIR = Path(os.environ.get("MARKSOCR_OUTPUT",
                                 PROJECT_DIR / "OCR_Results")).expanduser().resolve()

if not INPUT_DIR.exists() or not INPUT_DIR.is_dir():
    raise FileNotFoundError(
        "Input folder not found: %s\\n"
        "Put the mark sheets there, or set MARKSOCR_INPUT to the parent folder." % INPUT_DIR
    )

image_files = sorted(
    (p for p in INPUT_DIR.rglob('*')
     if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
    key=lambda p: str(p).lower()
)
if not image_files:
    raise RuntimeError("No supported image files found under %s" % INPUT_DIR)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# The batch and Excel cells still use this name; it now points at the local input.
DRIVE_FOLDER = INPUT_DIR

branch_counts = Counter()
parent_dir_counts = Counter()
for p in image_files:
    relative = p.relative_to(INPUT_DIR)
    branch = relative.parts[0] if len(relative.parts) > 1 else '(images directly in root)'
    branch_counts[branch] += 1
    parent_dir_counts[relative.parent.as_posix()] += 1

print(f"Found {len(image_files):,} image(s) recursively under: {INPUT_DIR}")
print(f"First-level folders found: {len(branch_counts):,}")
print("\\nImages per top-level folder:")
for folder, count in sorted(branch_counts.items(), key=lambda item: item[0].lower()):
    print(f"  {folder}: {count:,}")
print(f"\\nDistinct folders containing images: {len(parent_dir_counts):,}")
print(f"Excel reports and resumable checkpoints will be saved to: {OUTPUT_DIR}")
print("\\nSample input paths:")
for p in image_files[:20]:
    print(" -", p.relative_to(INPUT_DIR))
if len(image_files) > 20:
    print(f" ... and {len(image_files) - 20:,} more")
'''

FONT_OLD = '''from PIL import Image, ImageDraw, ImageFont
test_img = Image.new("RGB", (180, 80), "white")
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 56)
except Exception:
    font = ImageFont.load_default()'''

FONT_NEW = '''from PIL import Image, ImageDraw, ImageFont
test_img = Image.new("RGB", (180, 80), "white")
_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
font = next((ImageFont.truetype(p, 56) for p in _FONT_CANDIDATES if os.path.exists(p)),
            ImageFont.load_default())'''

COLAB_COMMENT_OLD = '''# Colab: select Runtime > Change runtime type > T4 GPU (if available) for faster server fallback.
# PaddleOCR accepts device="gpu:0" or "cpu". If GPU is unavailable, use CPU.'''

COLAB_COMMENT_NEW = '''# Local: Paddle runs on CPU (Apple Silicon has no CUDA device).
# PaddleOCR accepts device="gpu:0" or "cpu"; this notebook uses cpu.'''

TEXT_REPLACEMENTS = [
    ("# 8. Resumable full-batch OCR from Google Drive, with one Excel report per folder",
     "# 8. Resumable full-batch OCR from a local folder, with one Excel report per folder"),
    ("# Completed sheets are checkpointed to Drive. This versioned checkpoint starts fresh for v23 single-10-mark scoring; later reruns resume it.",
     "# Completed sheets are checkpointed to OUTPUT_DIR. This versioned checkpoint starts fresh for v23 single-10-mark scoring; later reruns resume it."),
    ('print("If Colab disconnects, reconnect and rerun this cell; completed folders and sheets resume from the checkpoint.")',
     'print("If the run is interrupted, rerun this cell; completed folders and sheets resume from the checkpoint.")'),
    ('"Source folder (relative to Drive root)"',
     '"Source folder (relative to input root)"'),
    ('print("Open the combined workbook from Google Drive:", OUTPUT_XLSX)',
     'print("Open the combined workbook:", OUTPUT_XLSX)'),
]


def _text(cell):
    return "".join(cell["source"])


def _set_source(cell, text):
    cell["source"] = text.splitlines(keepends=True)


def _replace_once(text, old, new):
    if old not in text:
        raise SystemExit("text to replace not found: %r" % old[:80])
    return text.replace(old, new, 1)


def _local_header(header):
    header = _replace_once(
        header,
        "# Mark-sheet OCR — Google Colab + Google Drive (user-rule updates v6)",
        "# Mark-sheet OCR — local run (user-rule updates v6)",
    )
    header = _replace_once(
        header,
        "This version preserves dynamic box detection",
        "This notebook is the local (macOS/Linux) version of `bruh.ipynb`: the same "
        "pipeline without Google Colab or Google Drive. Run `./setup_local.sh` once, "
        "then select the `MarksOCR (local)` kernel and run the cells in order. Input "
        "defaults to `./Random_Images_Sample`; output goes to `./OCR_Results` "
        "(override with `MARKSOCR_INPUT` / `MARKSOCR_OUTPUT`).\n\n"
        "This version preserves dynamic box detection",
    )
    header = _replace_once(
        header,
        "The input folder is read recursively from Google Drive; results and "
        "checkpoints are saved to Drive.",
        "The input folder is read recursively from a local directory; results and "
        "checkpoints are saved under `OCR_Results`.",
    )
    return header


def main():
    nb = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = nb["cells"]

    if not _text(cells[0]).startswith("# Mark-sheet OCR"):
        raise SystemExit("unexpected first cell in bruh.ipynb")
    _set_source(cells[0], _local_header(_text(cells[0])))

    install_i = next(i for i, c in enumerate(cells)
                     if c["cell_type"] == "code"
                     and _text(c).startswith("# 1. Install dependencies"))
    _set_source(cells[install_i], ENV_CHECK)

    drive_i = next(i for i, c in enumerate(cells)
                   if _text(c).startswith("# 3. Mount Google Drive"))
    _set_source(cells[drive_i], DISCOVERY)

    model_i = next(i for i, c in enumerate(cells)
                   if _text(c).startswith("# 5. Load a two-stage PaddleOCR"))
    model_src = _text(cells[model_i])
    model_src = _replace_once(model_src, FONT_OLD, FONT_NEW)
    model_src = _replace_once(model_src, COLAB_COMMENT_OLD, COLAB_COMMENT_NEW)
    _set_source(cells[model_i], model_src)

    for old, new in TEXT_REPLACEMENTS:
        hits = 0
        for cell in cells:
            if cell["cell_type"] != "code":
                continue
            src = _text(cell)
            if old in src:
                _set_source(cell, src.replace(old, new, 1))
                hits += 1
        if hits != 1:
            raise SystemExit("expected one match for %r, got %d"
                             % (old[:60], hits))

    for cell in cells:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        else:
            cell.pop("outputs", None)
            cell.pop("execution_count", None)
        meta = cell.get("metadata", {})
        for key in ("colab", "outputId", "kaggle", "widgets"):
            meta.pop(key, None)
        cell["metadata"] = meta

    old_meta = nb.get("metadata", {})
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "MarksOCR (local)",
            "language": "python",
            "name": "marksocr-local",
        },
        "language_info": {"name": "python", "version": "3.12"},
    }
    if old_meta.get("v6_notes"):
        nb["metadata"]["v6_notes"] = old_meta["v6_notes"]
    nb["nbformat"] = 4
    nb["nbformat_minor"] = 5

    TARGET.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n",
                      encoding="utf-8")
    print("wrote", TARGET)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Generate the notebook**

Run: `python3 dev/make_local_notebook.py`

Expected: `wrote /Users/bhargav/Downloads/MarksOCR/bruh_local.ipynb`

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 -m pytest dev/test_local_notebook.py -v`

Expected: 8 passed.

- [ ] **Step 6: Commit**

```bash
git add bruh.ipynb bruh_local.ipynb dev/make_local_notebook.py dev/test_local_notebook.py \
  docs/superpowers/specs/2026-11-01-local-notebook-design.md \
  docs/superpowers/plans/2026-11-01-local-notebook.md
git commit -m "feat: add local notebook generated from bruh.ipynb"
```

Leave the pre-existing deletions of the v6/v7 notebooks and the old `dev/`
helpers unstaged; they are not part of this change.

---

### Task 2: Add the Python 3.12 environment

**Files:**
- Create: `requirements-local.txt`
- Create: `setup_local.sh`
- Create: `dev/test_local_setup.py`
- Modify: `.gitignore`

**Interfaces:**
- `setup_local.sh` is executable, creates `./.venv312`, installs `requirements-local.txt`, registers the `marksocr-local` kernel, and runs an import smoke test.
- `requirements-local.txt` is consumed by `setup_local.sh`.

- [ ] **Step 1: Write the failing tests**

Create `dev/test_local_setup.py` with exactly:

```python
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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest dev/test_local_setup.py -v`

Expected: errors because the two files do not exist.

- [ ] **Step 3: Create `requirements-local.txt`**

```text
# Local (macOS arm64 / Linux) environment for bruh_local.ipynb.
# Python 3.12 is required: PaddlePaddle 3.2.0 has no Python 3.14 wheel.
numpy==1.26.4
paddlepaddle==3.2.0
paddleocr==3.2.0
openpyxl>=3.1,<4
matplotlib>=3.8,<4
ipykernel>=6.29,<8
```

- [ ] **Step 4: Create `setup_local.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3.12}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "error: $PYTHON not found. Install Python 3.12 (brew install python@3.12)" >&2
  echo "       or set PYTHON=/path/to/python3.12" >&2
  exit 1
fi

if [ ! -d .venv312 ]; then
  "$PYTHON" -m venv .venv312
fi

. .venv312/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-local.txt
python -m ipykernel install --user --name marksocr-local \
  --display-name "MarksOCR (local)"

python - <<'PY'
from importlib.metadata import version

import cv2
import numpy
import paddle
import paddleocr
from paddleocr import TextRecognition

print("Paddle:", paddle.__version__, "| CUDA compiled:", paddle.is_compiled_with_cuda())
print("PaddleOCR:", version("paddleocr"),
      "| OpenCV:", cv2.__version__, "| NumPy:", numpy.__version__)
PY

echo
echo "Environment ready."
echo "  Jupyter kernel: MarksOCR (local)"
echo "  VS Code interpreter: $(pwd)/.venv312/bin/python"
echo "  Next: open bruh_local.ipynb and run the cells in order."
```

Then make it executable: `chmod +x setup_local.sh`.

- [ ] **Step 5: Update `.gitignore`**

Append:

```text
.venv312/
OCR_Results/
.ipynb_checkpoints/
*.jsonl
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python3 -m pytest dev/test_local_setup.py -v`

Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
git add requirements-local.txt setup_local.sh dev/test_local_setup.py .gitignore
git commit -m "chore: add local Python 3.12 environment"
```

---

### Task 3: Build the venv and verify Paddle on this Mac

This is the risky task. Do not change the OCR backend without approval.

- [ ] **Step 1: Run the setup script**

Run: `./setup_local.sh`

Expected: pip finishes, the smoke test prints Paddle, PaddleOCR, OpenCV and NumPy versions, then `Environment ready.`

- [ ] **Step 2: If the Paddle import fails**

Try these in order, rerunning `./setup_local.sh` after each change to `requirements-local.txt`:

1. `brew install libomp`, then rerun.
2. Change the pin to `paddlepaddle==3.2.2`, rerun.
3. Change the pin to `paddlepaddle==3.3.1`, rerun.

If none work, stop and report the exact traceback. Do not replace PaddleOCR with EasyOCR/Tesseract without approval.

If a pin changes, update the expected pin in `dev/test_local_setup.py`
(`test_requirements_pin_known_good_versions`) in the same commit.

- [ ] **Step 3: Verify the recognizer class loads**

Run:

```bash
./.venv312/bin/python -c "from paddleocr import TextRecognition; print('TextRecognition ok')"
```

Expected: `TextRecognition ok`.

- [ ] **Step 4: Commit any pin change**

If `requirements-local.txt` changed:

```bash
git add requirements-local.txt dev/test_local_setup.py
git commit -m "chore: pin the working PaddlePaddle build"
```

---

### Task 4: End-to-end notebook run on two images

**Files:**
- Read: `bruh_local.ipynb`
- Output: `/tmp/marksocr_smoke/` (not committed)

- [ ] **Step 1: Prepare a two-image input folder**

```bash
rm -rf /tmp/marksocr_smoke
mkdir -p /tmp/marksocr_smoke/input
cp "Random_Images_Sample/II__22MTC04__840060_1035026.bmp" /tmp/marksocr_smoke/input/
cp "Random_Images_Sample/II__22CSC03__840060_1041117.bmp" /tmp/marksocr_smoke/input/
```

- [ ] **Step 2: Execute the whole notebook**

```bash
MPLBACKEND=Agg \
MARKSOCR_INPUT=/tmp/marksocr_smoke/input \
MARKSOCR_OUTPUT=/tmp/marksocr_smoke/out \
jupyter nbconvert --to notebook --execute bruh_local.ipynb \
  --output-dir /tmp/marksocr_smoke --output executed.ipynb \
  --ExecutePreprocessor.kernel_name=marksocr-local \
  --ExecutePreprocessor.timeout=1800
```

Expected: exit code 0. The inspect cell renders the overlay; the pilot cell prints two records; the batch cell prints `newly processed 2`.

- [ ] **Step 3: Check the Excel report and checkpoint**

```bash
python3 - <<'PY'
import json
from pathlib import Path

import openpyxl

out = Path("/tmp/marksocr_smoke/out")
xlsx = out / "_root_images.xlsx"
wb = openpyxl.load_workbook(xlsx)
print("sheets:", wb.sheetnames)
for name in ("Summary", "All Results", "Review Needed",
             "OCR Diagnostics", "Field Audit"):
    assert name in wb.sheetnames, name

ckpt = out / "ocr_checkpoint_pilotfix_v25_audit.jsonl"
lines = ckpt.read_text(encoding="utf-8").splitlines()
print("checkpoint lines:", len(lines))
assert len(lines) == 2
PY
```

Expected: all five sheet names present, `checkpoint lines: 2`.

- [ ] **Step 4: Re-run to verify checkpoint resume**

```bash
MPLBACKEND=Agg \
MARKSOCR_INPUT=/tmp/marksocr_smoke/input \
MARKSOCR_OUTPUT=/tmp/marksocr_smoke/out \
jupyter nbconvert --to notebook --execute bruh_local.ipynb \
  --output-dir /tmp/marksocr_smoke --output executed2.ipynb \
  --ExecutePreprocessor.kernel_name=marksocr-local \
  --ExecutePreprocessor.timeout=1800
```

Then:

```bash
python3 - <<'PY'
import json

nb = json.load(open("/tmp/marksocr_smoke/executed2.ipynb", encoding="utf-8"))
text = "\n".join(
    "".join(output.get("text", []))
    for cell in nb["cells"] if cell["cell_type"] == "code"
    for output in cell.get("outputs", [])
)
assert "newly processed 0" in text, text[-2000:]
print("resume ok")
PY
```

Expected: `resume ok`.

- [ ] **Step 5: If execution fails**

Fix the local notebook or setup, regenerate with `python3 dev/make_local_notebook.py`, and rerun Steps 2-4. Commit fixes with:

```bash
git add bruh_local.ipynb dev/make_local_notebook.py setup_local.sh requirements-local.txt
git commit -m "fix: make the local notebook execute end to end"
```

---

### Task 5: Document the local workflow

**Files:**
- Modify: `README.md`
- Modify: `dev/test_local_setup.py`

- [ ] **Step 1: Add the failing documentation test**

Append this method to `LocalSetupTest` in `dev/test_local_setup.py`:

```python
    def test_readme_documents_local_run(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        for needle in ("Run locally", "setup_local.sh", "MARKSOCR_INPUT",
                       "bruh_local.ipynb"):
            self.assertIn(needle, text)
```

Run: `python3 -m pytest dev/test_local_setup.py -v`

Expected: `test_readme_documents_local_run` fails.

- [ ] **Step 2: Update the README Contents**

Replace the first list in the Contents section with:

```markdown
- `bruh.ipynb` — the Colab pipeline and batch runner (source of truth for the pipeline).
- `bruh_local.ipynb` — local (non-Colab) copy generated from `bruh.ipynb`.
- `requirements-local.txt` / `setup_local.sh` — local Python 3.12 environment.
- `dev/make_local_notebook.py` — regenerates `bruh_local.ipynb` from `bruh.ipynb`.
- `dev/test_local_notebook.py`, `dev/test_local_setup.py` — local regression tests.
- `README.md` — setup, usage, output layout, and validation notes.
- `Random_Images_Sample/` — 186 local test images (jpg/bmp/tif).

The v6/v7 notebooks and the old `dev/` helpers remain in git history.
```

- [ ] **Step 3: Add the local run section**

Insert this section immediately before `## How to run`:

````markdown
## Run locally (macOS / Linux, CPU)

`bruh_local.ipynb` is the local version of `bruh.ipynb`: same pipeline, no
Google Drive, local input and output folders.

### Setup

```bash
./setup_local.sh
```

This creates `./.venv312` with Python 3.12 (PaddlePaddle 3.2.0 has no Python
3.14 wheel), installs `requirements-local.txt`, and registers the
`MarksOCR (local)` Jupyter kernel.

### Run

1. Open `bruh_local.ipynb` in JupyterLab/Notebook or VS Code.
2. Select the `MarksOCR (local)` kernel (VS Code: `.venv312/bin/python`).
3. Run the cells in order.

Input defaults to `./Random_Images_Sample`; Excel reports and the checkpoint
go to `./OCR_Results`. Override either folder:

```bash
MARKSOCR_INPUT=/path/to/sheets MARKSOCR_OUTPUT=/path/to/results \
  jupyter lab bruh_local.ipynb
```

Launch Jupyter from the repository root so the default paths resolve.

### What changes from the Colab notebook

- The pip-install cell becomes an environment check (no runtime installs).
- `google.colab.drive` is gone; `INPUT_DIR` / `OUTPUT_DIR` are local paths.
- The smoke-test font lookup works on macOS and Linux.
- Pipeline cells (2, 2b, 3) are unchanged; `dev/test_local_notebook.py`
  asserts that.

### Expected speed and models

CPU-only on Apple Silicon is roughly 10 s per sheet. The PP-OCRv5 mobile and
server models are cached in `~/.paddlex/official_models`; if missing,
PaddleOCR downloads them once on first use.

### Troubleshooting

- `No module named paddle` or a Python 3.14 error: the notebook is using the
  base kernel. Select `MarksOCR (local)` or `.venv312/bin/python`.
- `libomp` / `libmkl` import error: `brew install libomp`, then rerun
  `./setup_local.sh`.
- To rebuild the local notebook after editing `bruh.ipynb`:
  `python3 dev/make_local_notebook.py`.
````

- [ ] **Step 4: Run the full test suite**

Run: `python3 -m pytest dev/ -v`

Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add README.md dev/test_local_setup.py
git commit -m "docs: document the local notebook workflow"
```

---

## Self-Review Notes

- Spec coverage: local notebook (Task 1), environment (Tasks 2-3), end-to-end
  verification (Task 4), documentation (Task 5). The spec's `DRIVE_FOLDER`
  alias, output stripping, and path overrides are tested in Task 1.
- Type consistency: `INPUT_DIR`, `OUTPUT_DIR`, `image_files`, `DRIVE_FOLDER`
  are defined once in `DISCOVERY` and used unchanged by the batch/Excel cells.
- Review Focus tests: wrong kernel (environment cell text), missing/empty
  input (Task 1 tests), env overrides (Task 1 test), Paddle import (Task 3
  smoke), `DRIVE_FOLDER` alias (Task 1 test).
