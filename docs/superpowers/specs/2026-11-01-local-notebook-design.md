# Local notebook design (2026-11-01)

## Intent

Run `bruh.ipynb` on the local Mac without Google Colab or Google Drive,
keeping the OCR/validation/Excel pipeline unchanged and changing only the
environment-dependent cells.

## Goals

- `bruh_local.ipynb` opens in JupyterLab/Notebook or VS Code, uses a Python
  3.12 venv, and runs the existing inspect -> pilot -> batch -> review flow on
  `./Random_Images_Sample`.
- Same OCR code, validation rules, Excel layout, checkpoint/resume.
- One command creates the environment; the notebook never pip-installs at
  runtime.

## Non-goals

- No package extraction, no command-line runner, no pipeline refactor, no OCR
  backend swap.
- `bruh.ipynb` itself is not modified.

## Anti-goals

- Changing recognition behavior or Excel columns.
- A notebook that only works after manually editing `/content/drive` strings.

## Constraints

- Apple Silicon, macOS 26.6. The default Python is 3.14; PaddlePaddle 3.2.0
  has wheels only up to cp313. Homebrew `python3.12` is installed.
- CPU only, roughly 10 s/sheet. The PP-OCRv5 mobile/server models are already
  cached in `~/.paddlex/official_models`.
- Pin `numpy==1.26.4`, `paddlepaddle==3.2.0`, `paddleocr==3.2.0`.
  PaddleOCR pulls `paddlex[ocr-core]` and `opencv-contrib-python==4.10.0.84`.
  Do not install `opencv-python-headless` alongside it.
- Input defaults to `./Random_Images_Sample`; output to `./OCR_Results`.
  Override with `MARKSOCR_INPUT` / `MARKSOCR_OUTPUT`.
- The batch/report cells still reference the name `DRIVE_FOLDER`; keep it as
  an alias for `INPUT_DIR` so those cells stay unchanged.

## Approaches considered

1. **Patch in place (chosen).** Copy the notebook, replace the install and
   Drive cells, fix the Linux font lookup, strip Colab outputs. A generator
   (`dev/make_local_notebook.py`) keeps the pipeline cells byte-identical to
   `bruh.ipynb`. Lowest risk and exactly the notebook the user knows.
2. **Self-installing notebook.** Same replacements, but the first cell creates
   the venv and pip-installs. One file, but fragile and requires a kernel
   restart mid-notebook.
3. **Thin notebook over a package.** Extract cells 2/2b/3 into `marksocr/` and
   import it. Most maintainable, but a much larger change than requested.

## Decisions

- Cell 1 (install) becomes an environment check. It verifies Python 3.10-3.13
  and imports the dependencies; on failure it names `./setup_local.sh` and the
  `MarksOCR (local)` kernel.
- Cell 5 (Drive) becomes local discovery. It keeps the recursive scan, folder
  counts and sample listing, defines `INPUT_DIR`, `OUTPUT_DIR`, `image_files`
  and `DRIVE_FOLDER = INPUT_DIR`.
- Cell 8 (models) keeps the two-stage PaddleOCR load; the smoke-test font
  becomes a macOS/Linux candidate list. Device resolves to CPU.
- Cells 2, 2b and 3 are not edited. `dev/test_local_notebook.py` asserts they
  match `bruh.ipynb`.
- Colab outputs, `execution_count`, and Colab/Kaggle/widget metadata are
  stripped. The notebook kernelspec is `marksocr-local`.
- `requirements-local.txt` and `setup_local.sh` create `./.venv312`, install
  the pins, register the kernel, and print a version smoke test.
- `.gitignore` ignores `.venv312/`, `OCR_Results/`, `.ipynb_checkpoints/`,
  `*.jsonl`.

## Left to the builder

Exact cell text, generator code, test code, setup script and README wording
are specified in `docs/superpowers/plans/2026-11-01-local-notebook.md`.

## Verification

- `./setup_local.sh` completes and the venv imports Paddle, PaddleOCR, OpenCV.
- Inspect cell renders an overlay for one sample.
- Five-sheet pilot produces answers/totals/status.
- End-to-end `nbconvert` run on a two-image folder writes the Excel report and
  the checkpoint; a second run resumes with zero new sheets.
- Regression tests pass and confirm the pipeline cells are unchanged.

## Risks

- PaddlePaddle 3.2.0 may fail to import on arm64 (missing `libomp`). Mitigate
  with `brew install libomp`; if it still fails, try `paddlepaddle==3.2.2` or
  `3.3.1`, and stop to report before changing the OCR backend.
- Wrong kernel selection produces import errors; the environment-check cell
  names the fix.
- Launching Jupyter outside the repo root changes `PROJECT_DIR`; documented,
  and env overrides cover it.
