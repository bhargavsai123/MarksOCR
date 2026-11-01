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
