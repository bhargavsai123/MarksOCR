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

Pipeline version: `dynamic-grid-v24-audit`

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

## Audit-driven accuracy update (v24)

Every cell of the 604 locally available sheets was re-read by hand and compared with
the pipeline output. The changes in v24 come from that audit:

- **Part B digits get a second opinion.** When the mobile recogniser returns a single
  weak Part B digit (score below 0.9) the server model reads the same crops as well.
  This removed about 14% of the wrong Part B digits and left the strong reads alone.
- **A dash is a dash.** A cell that holds a clear horizontal dash plus a tiny,
  low-confidence mark is scored `0` instead of `1`. A dash next to a real digit is still
  never collapsed to `0`.
- **Handwritten totals are read as one number.** `Total(A)`, `Total(B)` and
  `TOTAL(A+B)` are additionally read from a white-canvas union of their handwriting
  components (mobile and server), so a `17` or `52` split over two strokes is no longer
  read as `1`, and the printed `TOTAL(A+B)` label is no longer read as the value.
- **No invented Part A marks.** An unreadable Part A cell is no longer filled in from a
  handwritten `Total(A)` unless the written totals agree with the bubbled total and the
  solution is unique. A misread `Total(A)` used to invent two zero marks.
- **No invented marks at all.** If the bubble total can only be reached by giving a
  cell a value that no recogniser read, that value is not stored and the sheet is
  flagged `REVIEW` with the question named. About half of the invented marks turned out
  to be wrong, so they must not be published as if they had been read.
- **Contradictions fail the sheet.** When the written `Total(A)` and the written
  `Total(B)` both disagree with the computed marks, or when a confidently read row
  total contradicts the marks of that row while the written totals also disagree, the
  sheet is flagged `REVIEW` instead of being reported as valid.
- **The joint solver stops chasing a misread total.** The handwritten totals are only
  used as a tie-breaker when they agree with each other and with the bubble.

Audit result on the 168 sheets that were hand-audited mark by mark:

| metric | v23 baseline | v24 |
| --- | --- | --- |
| wrong mark cells | 296 of 4184 (7.1%) | 263 of 4184 (6.3%) |
| sheets with at least one wrong mark | 92 | 87 |
| false `OK` (sheet contradicts itself but is reported valid) | 17 | 6 |
| false `REVIEW` (sheet is consistent but is sent to review) | 22 | 34 |
| sheet status agreement | 76.8% (129/168) | 76.2% (128/168) |

The status agreement is flat on purpose: the stricter rules send 12 more audited sheets
to `REVIEW`, and that is the intended trade-off. Nothing is invented any more, and a
sheet that contradicts itself can no longer be reported as valid.

Runtime on the local CPU-only machine went from 9.6 s to 10.4 s per sheet because of the
extra server-model reads; the server model is only asked when the mobile read is weak.

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
OCR_Results/ocr_checkpoint_pilotfix_v24_audit.jsonl
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

## Performance

Expression fallback is the most expensive part of a sheet. It now caches results per cell, recognizes operand crops in batches, tries the strongest plus-shaped crosses first, and limits the low-score Part A server retry to the first pass. The reverse pass is used for conflict detection rather than repeating the full expression search.

On the local CPU-only validation machine, `840030_533056.bmp` dropped from about 24 s with the old sequential expression path to about 13 s; the v19 baseline on the same machine was about 10 s. Production runs with a CUDA-enabled PaddlePaddle runtime are faster.

## Validation snapshot

- Local audit of 604 sheets: 333 `OK` / 271 `REVIEW` with v24 (423 / 181 with the v23
  baseline). The change is deliberate: sheets whose written totals contradict the marks,
  and sheets where a mark would have to be invented, are now reported as `REVIEW`
  instead of being forced to a valid total.
- 168 sheets were hand-audited mark by mark; see the v24 section above for the
  before/after numbers.
- Check-folder statuses: 2 OK / 3 REVIEW with v24 (3 OK / 2 REVIEW with the v23
  baseline). The extra `REVIEW` is `840030_531595.bmp`, where the bubble total could
  only be reached by inventing several Part A and Part B marks.
- Ten-sheet ground-truth smoke run: 169/176 mark cells (96.0%) with two sheets sent to
  `REVIEW` because a mark would have had to be invented (baseline: 170/176 and one
  sheet in `REVIEW`).
- `840030_533056.bmp` reads the handwritten `3+2` expression in Q8 and validates OK.
- The supplied connected Q5 half-mark shape is read as `0.5`; the sheet remains REVIEW because its computed total does not match the bubble.
- Normal digit cells were checked to ensure the half-mark fallback does not turn them into false fractions.
- Synthetic checks confirm that a single Part B mark of `7` or `10` is stored as a total-only question and validates without a/b unreadable flags.

## Known limits

- Handwriting recognition is not guaranteed. Validation and human review remain required.
- The handwritten `Total(A)`, `Total(B)` and `TOTAL(A+B)` cells are still the weakest
  read on the sheet; a wrong total is reported as a warning or, when two totals
  contradict the marks at once, as `REVIEW`.
- Shape-level half-mark recognition is a weak fallback, not a general classifier.
- Single 10-mark detection requires the other row to be blank/dashed and the row total to agree; ambiguous rows remain in REVIEW.
- Totals and bubbles can disagree because of handwriting, scanning, or bubble-grid quality; those sheets stay in REVIEW.
- A full production run should be done in chunks and monitored, with the checkpoint kept on Google Drive.
