# MarksOCR

Google Colab pipeline for reading handwritten mark sheets, validating totals against the printed bubble grid, and exporting Excel reports folder by folder.

The repository contains the Colab notebook and this README only.

## Contents

- `Mark Sheet OCR Pilot Fixes v6.ipynb` — the complete OCR pipeline and batch runner.
- `README.md` — setup, usage, output layout, and validation notes.

## What the pipeline does

- Detects the orange grid and cell boundaries dynamically; absolute pixel coordinates are not hard-coded.
- Preserves handwriting that extends outside the printed cells.
- Reads Part A marks, Part B `(a)`, `(b)`, and row totals with PaddleOCR mobile/server models.
- Reads the printed Total Marks and Answer Book SI bubbles.
- Validates question totals, Part A/Part B totals, handwritten totals, and the bubbled grand total.
- Counts only the higher-total question when both alternatives in a Part B pair are attempted.
- Recognizes single 10-mark Part B questions where the student writes one mark and leaves the other `(a)`/`(b)` row blank or dashed.
- Distinguishes genuinely blank cells from non-blank but unreadable handwriting.
- Keeps uncertain or contradictory sheets in a `REVIEW` status instead of forcing a valid result.

## Math and fraction support

Pipeline version: `dynamic-grid-v23-single10`

The parser supports:

- Sums: `3+2`, `3 + 2 = 5`
- Multi-digit expressions: `40+2=42`
- Fractions: `1/2`, `3/2`, `1/2=0.5`
- Mixed fractions: `5 1/2`
- Unicode fractions: `½`, `2½`
- Decimals: `0.5`, `2.5`

Rules:

- An equation prefers the value after `=`.
- A slash is parsed as a fraction and is never treated as a plus sign.
- A contradictory or out-of-range equation is not silently repaired from the left-hand side.
- Connected cursive half-marks use a narrow, low-confidence shape fallback only in Part A. The handwritten totals and bubbled Total Marks still decide whether the sheet passes.
- Some Part B questions are single 10-mark questions rather than `(a)` + `(b)` pairs. A mark of `6`–`10` in one row is accepted only when the other row is blank/dashed and the written row total agrees. It is stored as a total-only question, not as an out-of-range `(a)` or `(b)` mark.

## Input folder structure

The notebook expects a parent folder containing the six top-level folders:

```text
Main/
  Folder1/
    InnerA/
      image1.jpg
      image2.jpg
    InnerB/
      image3.jpg
  Folder2/
    InnerC/
      image4.jpg
  ...
```

Supported extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tif`, `.tiff`.

Set the input and output roots in the notebook:

```python
DRIVE_FOLDER = Path('/content/drive/MyDrive/data2')
OUTPUT_DIR = Path('/content/drive/MyDrive/data2/OCR_Results')
```

`DRIVE_FOLDER` must be the parent of the six folders, not one of the six folders itself.

## Per-inner-folder Excel reports

Images are grouped by the innermost folder that directly contains them. When the final image in an inner folder finishes, its workbook is written immediately.

Example:

```text
OCR_Results/
  Folder1/
    InnerA.xlsx
    InnerB.xlsx
  Folder2/
    InnerC.xlsx
```

If images are directly inside a top-level folder, the fallback is:

```text
OCR_Results/Folder3.xlsx
```

Each workbook contains:

- `Summary`
- `All Results`
- A worksheet for the source folder
- `Review Needed`
- `OCR Diagnostics`
- `Field Audit`

The old combined 15,000-sheet workbook is optional and disabled by default:

```python
BUILD_FINAL_COMBINED_WORKBOOK = False
```

## Checkpoint and resume

Completed sheets are appended to:

```text
OCR_Results/ocr_checkpoint_pilotfix_v23_single10.jsonl
```

If Colab disconnects, rerun the full-batch cell. Completed sheets load from the checkpoint, and missing folder reports are rebuilt without re-running OCR.

Useful switches:

```python
RESET_CHECKPOINT = False
FORCE_REBUILD_FOLDER_REPORTS = False
```

Set `RESET_CHECKPOINT = True` once only if you intentionally want to discard old checkpoint records and rerun every sheet.

## How to run

1. Open the notebook in Google Colab.
2. Run the dependency installation cell.
3. Mount Google Drive and set `DRIVE_FOLDER` / `OUTPUT_DIR`.
4. Run the alignment inspection cell on a sample sheet.
5. Run the model-loading cell.
6. Optionally run the pilot on up to five sheets.
7. Run the full-batch cell. Per-inner-folder Excel reports are written as folders complete.
8. Use the optional 10-sheet accuracy check and review-only listing as needed.

Use a GPU runtime when available. The pipeline checks whether PaddlePaddle is CUDA-enabled and falls back to CPU otherwise.

## Validation snapshot

- Check-folder statuses: 3 OK / 2 REVIEW.
- Ten-sheet ground-truth smoke run: 9/10 valid.
- `840030_533056.bmp` reads the handwritten `3+2` expression in Q8 and validates OK.
- The supplied connected Q5 half-mark shape is read as `0.5`; the sheet remains REVIEW because its computed total does not match the bubble.
- Normal digit cells were checked to ensure the half-mark fallback does not turn them into false fractions.
- Synthetic checks confirm that a single Part B mark of `7` or `10` is stored as a total-only question and validates without a/b unreadable flags.

## Known limits

- Handwriting recognition is not guaranteed. Validation and human review remain required.
- Shape-level half-mark recognition is a weak fallback, not a general classifier.
- Single 10-mark detection requires the other row to be blank/dashed and the row total to agree; ambiguous rows remain in REVIEW.
- Totals and bubbles can disagree because of handwriting, scanning, or bubble-grid quality; those sheets stay in REVIEW.
- A full production run should be done in chunks and monitored, with the checkpoint kept on Google Drive.
