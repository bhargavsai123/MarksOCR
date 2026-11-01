# MarksOCR

Google Colab pipeline for reading handwritten mark sheets, validating totals against the printed bubble grid, and exporting Excel reports folder by folder.

The repository contains the Colab notebooks, this README, and the local test helpers under `dev/`.

## Contents

- `bruh.ipynb` — the Colab pipeline and batch runner (source of truth for the pipeline).
- `bruh_local.ipynb` — local (non-Colab) copy generated from `bruh.ipynb`.
- `requirements-local.txt` / `setup_local.sh` — local Python 3.12 environment.
- `dev/make_local_notebook.py` — regenerates `bruh_local.ipynb` from `bruh.ipynb`.
- `dev/test_local_notebook.py`, `dev/test_local_setup.py` — local regression tests.
- `README.md` — setup, usage, output layout, and validation notes.
- `Random_Images_Sample/` — 186 local test images (jpg/bmp/tif).

The v6/v7 notebooks and the old `dev/` helpers remain in git history.

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

Pipeline version: `glyph-grid-v26`

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

## Glyph-first front-end (v26, pipeline `glyph-grid-v26`)

The v25 pipeline erased a dilated band around every printed orange rule and then
assigned ink to the nearest cell centre.  Handwriting does not stay inside the
printed boxes: a digit written across a rule lost the pixels that crossed it, and
a digit tall enough to reach the next row was split into one fragment per row.
Each fragment was then cropped on its own, so the recogniser saw half a digit.

The v26 front-end works the other way round:

1. **Pen mask by colour and structure.**  A pixel is ink if it is darker than the
   local paper, or dark in absolute terms, or a saturated non-orange colour.
   Printed rules are removed by their *shape* (long straight orange runs), not by
   erasing a band.  A blue stroke on top of an orange rule survives because the
   blend is dark but not a long straight run.

2. **Whole-glyph segmentation.**  Connected components are joined into whole
   glyphs before anything is cropped.  Vertical join: two pieces of one digit
   separated by a rule (or a pen lift) are joined when they overlap
   horizontally and the gap is small, *unless* a detected row rule lies
   between two comparably sized blobs.  Horizontal join: two strokes of one
   digit (a 5, a 4, a 0 with a gap) are joined across a hairline gap.

3. **Ink-overlap assignment.**  Every glyph is given to the cell that holds the
   majority of its ink pixels.  A tall glyph keeps its `share` and `shares`
   (the fraction of the glyph in every cell) so the caller can see that it
   crossed a row without being cut.

4. **Glyph crops, not cell crops.**  The rendered view is the bounding box of
   the cell's glyphs plus margin, never the cell rectangle.  The crop contains
   the entire glyph even when it spans two rows.

5. **Shared-glyph ambiguity flag.**  When two digits in different rows touch
   and are merged into one glyph, the glyph stays whole for the row that owns
   it.  The other row gets a `part` crop of the ink it physically contains,
   and is marked `ambiguous=True` so the validation layer reports `REVIEW`
   instead of a clean blank.  A genuine tall digit's thin descender does *not*
   flag the next row (width < 0.45×row-height in the minority cell).

6. **Per-row fallback crop (`part` view).**  Every cell that owns a tall glyph
   also gets a crop restricted to its own row (`part` view).  The recogniser
   tries the whole-glyph crop first; if the reading is out of range (e.g. "43"
   for a 0–15 mark), the `part` crop is tried and typically reads the intended
   digit ("4").

Key behavioural changes:

| Scenario | v25 (band-erase) | v26 (glyph-first) |
|---|---|---|
| Digit written across a rule | Pixels on the rule lost; digit split | Entire stroke kept; whole glyph |
| Large digit reaching next row | Cut at rule; two half-crops | One tall glyph, full crop |
| Two digits in adjacent rows touch | One row empty, other gets both | Both rows get crops; second flagged `ambiguous` |
| Faint ink inside rule band | Erased | Recovered (structural rule removal + band protect) |

Validation logic (read_cell, arithmetic checks, Excel writer) is unchanged; it
receives the new view keys `color`, `binary`, `context`, `part`, `part_box`,
`per_glyph`, and legacy keys `box`, `large`, `overflow_x`, `overflow_y` for
compatibility with `_read_split_sum` and `_read_component_total`.

### Bubble grid (Total Marks) fit — v26

The OMR panel is **not** a fixed fraction of the Part B rectangle.  The
II-series forms use a taller, wider panel than the IV / I-Sem / BE forms, so the
old `B`-relative grid (x at `1.051, 1.115, 1.207, 1.279 × B.w`, row pitch
`0.21 × B.h`) sampled blank paper for the tens digit and the tens column for the
units.  Every II sheet therefore came back with a single digit: `Total Marks 5`
for a bubbled `50/51/52`, `4` for `46`, `0` for `14`.

`detect_bubbles` now fits the grid to the **printed orange rings**:

1. **Candidates.**  Connected orange components to the right of Part B, with a
   roughly square bounding box (10–60 px, aspect 0.7–1.4) — printed bubble
   rings and the printed header digits.
2. **Columns.**  Cluster the candidate x-centres and keep the first four that
   hold at least five rings.  A pen-filled bubble is still an orange ring
   fragment, so it survives the count.
3. **Rows.**  Accept a candidate y only when at least two of the other three
   columns have a ring at the same height.  Rows whose rings are hidden under
   pen are restored from the row pitch, and the edge rows are extended outward
   only while some column still has a ring there.
4. **Radius.**  The 75th-percentile ring half-width sets the fill disk
   (`r × 0.62`), so the disk covers the bubble interior and not the rule.

If the rings are too thin or fragmented — the JPEG BE sheets — `_ring_grid`
returns `None` and `detect_bubbles` falls back to the previous nominal + Hough
grid unchanged.  The IV layout still reads `38 / SI 13`; the BE JPEG still reads
`37`; only the II sheets change, and they change from one digit to two.

Verification across the 186 samples: all 36 `II__` sheets fit a full 4×10 grid,
and 35 of them now read a two-digit total (`II__22MTC04__840060_1035026` 5→**52**,
`II__22CSC03__840060_1041117` 5→**54**, `II__22EGC01__840060_1036013` 4→**45**,
`II__22PHY07__840060_1041602` 5→**51**, `II__22CEC01__840060_1039278` 0→**14**).
The one non-II change is a BE_B sheet whose second digit is now read from the
ring grid; its grand total is unchanged.  `python3 dev/test_bubble_grid.py`
(9 tests) execs the shipped notebook cell and asserts these cases.

### No-ground-truth workflow

Because the full 49 GB dataset has no ground-truth marks, the pipeline is
validated by diagnostics rather than an accuracy score:

1. **Run the sample diagnostics script** (`dev/run_sample.py`) on
   `Random_Images_Sample/`.  It produces:
   - `/tmp/glyph_diagnostics.csv` — one row per cell with glyph geometry,
     ambiguity flags, overflow metrics.
   - `/tmp/glyph_crops/contact_*.png` — visual contact sheets of every crop.
2. **Check the ambiguity rate.**  ~5 % of cells are `ambiguous` (touching
   digits).  If this spikes on a new batch, the glyph joiner may be too
   aggressive.
3. **Check the "large" cell count.**  ~9 % of ink cells have `large=True`
   (glyph taller than 1.25×row).  These are the cases the old pipeline cut.
4. **Verify crops visually.**  Open the contact sheets: every crop should show
   a complete digit, never a half-digit clipped at a rule.
5. **Run the pilot (5 sheets) in Colab** and inspect the per-view diagnostics
   printed by the pilot cell.  The `REVIEW` rate should be stable.
6. **Full batch.**  The notebook writes per-folder Excel reports; `Review
   Needed` worksheets list sheets with ambiguous cells or contradictory totals.

The front-end is testable in isolation (`python3 dev/test_glyph_frontend.py`,
13 tests) and does not require PaddleOCR or Colab.

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
DRIVE_FOLDER = Path('/content/drive/MyDrive/data2_glyph')
OUTPUT_DIR = Path('/content/drive/MyDrive/data2_glyph/OCR_Results')
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
OCR_Results/ocr_checkpoint_glyph_grid_v26.jsonl
```

If Colab disconnects, rerun the full-batch cell. Completed sheets load from the checkpoint, and missing folder reports are rebuilt without re-running OCR.

Useful switches:

```python
RESET_CHECKPOINT = False
FORCE_REBUILD_FOLDER_REPORTS = False
```

Set `RESET_CHECKPOINT = True` once only if you intentionally want to discard old checkpoint records and rerun every sheet.

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
