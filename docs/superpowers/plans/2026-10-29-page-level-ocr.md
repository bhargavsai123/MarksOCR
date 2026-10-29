# Page-Level OCR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace per-cell cropped OCR with a single PaddleOCR pipeline call over the entire mark sheet; map detected text boxes to answer cells by grid coverage and read each cell from the page-level text observations.

**Architecture:** Cell 6 loads one `PaddleOCR` pipeline (det+rec, PP-OCRv5_server on GPU or mobile on CPU). A new `read_grid()` runs `predict()` once on the deskewed page and returns per-box text/score/polygon. Boxes are assigned to cells by max coverage (>30%) using `cell_defs` geometry; multiple boxes in a cell are joined in reading order. `read_cell` gains a `grid_text` path that uses the page-level observations directly instead of rendered views; view rendering is skipped in batch mode (`analyze_sheet(render_views=False)`).

**Tech Stack:** Python, NumPy, OpenCV, PIL, PaddleOCR 3.x, Jupyter notebook. No new notebook cells.

**Spec:** Supersedes `docs/superpowers/specs/2026-10-28-crop-region-ownership-design.md`; new spec is this plan.

## Global Constraints

- All product code lives in cell 2 of `Mark Sheet OCR Pilot Fixes v6.ipynb`; model loading lives in cell 6.
- Edit the notebook through `json.load` / `json.dump` and validate with `ast.parse` before writing. No new notebook cells.
- PaddleOCR is not installed locally; all verification of OCR accuracy happens on Colab against the 10-sheet ground truth.
- Target: ≥96% cell accuracy (169/176 cells) on the 10 supplied ground-truth sheets.
- No git repository exists in the working directory. No commits; JSON rewrite + ast.parse verification.
- Checkpoint version bumped (old per-cell records not mixable with grid records).
- Both `render_cell` call sites (`analyze_sheet`, `extract_sheet`) updated together; `extract_sheet` is unused in grid mode but kept consistent.

## Review Focus

| Input / condition | Expected behavior | Pinned by task |
|---|---|---|
| Page smaller than detector default resolution (1389×650 sheets) | `limit_side_len=1600` keeps digits at native scale | Task 1 |
| Printed labels `Q1..Q15`, headers, bubble text | Coverage <30% → not assigned to any cell | Task 2 |
| Handwritten "75" split into two boxes | Joined in reading order → `parse_integer` reads 75 | Task 2 |
| Single recognition per box, no server fallback | Uncertain boxes drop to weak/None reading | Task 3 |
| Blank cell (`has_ink=False`) | Never OCR'd even if nearby label detected | Task 4 |
| Digit overlapping two cells | Assigned to cell with max coverage | Task 2 |
| No reverse second pass | One pass per cell; grid mode skips `_read_second_pass` | Task 5 |
| Total(A+B) grand total | Read from grid text; `expected` from bubble minus Part A | Task 5 |
| Old checkpoint format | Incompatible; `CHECKPOINT_PATH` version bumped | Task 6 |
| `extract_sheet` call site | Kept consistent with new `render_cell` signature | Task 6 |

## Tasks

### Task 1: Model loading — cell 6 loads PaddleOCR pipeline

**Files:** Modify cell 2 (`PaddleOCR` import, `load_ocr()` helper) and cell 6 (model loading).

**Interfaces:**
- Consumes: nothing (config from cell 6 globals `DEVICE`, `MOBILE_MODEL`, `SERVER_MODEL`)
- Produces: `load_ocr() -> LoadedModel` where `LoadedModel.pipeline` is a `PaddleOCR` instance; `LoadedModel.device`, `LoadedModel.recognizer` kept for compatibility.

- [ ] Step 1: Add `from paddleocr import PaddleOCR` to cell 2 imports; add `load_ocr(loaded)` that builds `PaddleOCR(ocr_version="PP-OCRv5", det_model_name="PP-OCRv5_server_det", rec_model_name="PP-OCRv5_server_rec", device=DEVICE, use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False, text_det_limit_side_len=1600)` on GPU, or `PP-OCRv5_mobile_det` + `PP-OCRv5_mobile_rec` on CPU.
- [ ] Step 2: Cell 6 calls `load_ocr(loaded)` after the existing model loading; keeps smoke test working with the pipeline.
- [ ] Step 3: Save notebook; `ast.parse` all code cells.
- [ ] Step 4: Commit snapshot.

### Task 2: Page-level detection + box-to-cell assignment

**Files:** Modify cell 2 — add `read_grid()`, `extract_boxes()`, `assign_to_cells()`.

**Interfaces:**
- `read_grid(loaded, arr) -> list[dict]`: runs `loaded.pipeline.predict(arr)` once; returns `[{box: [x1,y1,x2,y2], text: str, score: float}]`.
- `extract_boxes(result) -> list[dict]`: handles PaddleOCR result dict/list/generator forms; returns normalized boxes.
- `assign_to_cells(detections, cells, det) -> dict[str, list[dict]]`: for each cell key, list of `{text, score, coverage}` assigned by max coverage >0.30; multiple boxes joined in reading order (top-to-bottom, left-to-right).

- [ ] Step 1: Write offline test with synthetic page image + fake detections: two boxes in one cell joined, a printed label outside all cells dropped, a digit overlapping two cells goes to max-coverage cell.
- [ ] Step 2: Implement `read_grid`, `extract_boxes`, `assign_to_cells` in cell 2.
- [ ] Step 3: Run offline test → PASS; save notebook; `ast.parse`.
- [ ] Step 4: Commit snapshot.

### Task 3: `read_cell` grid_text path

**Files:** Modify cell 2 — `read_cell` signature gains `grid_text: list[dict] | None = None`.

**Interfaces:**
- `read_cell(loaded, entry, allowed, max_val, max_digits, reverse, allow_half, grid_text=None) -> Dict`
- When `grid_text` is non-empty: one observation per entry (joined text, max score), existing parsers (`parse_single_digit`, `parse_integer`, `parse_math_result`), no view loop, no server fallback, no split-sum/operand crops.
- When `grid_text` is empty/None: fall back to existing views path (kept for inspection cell / old mode).
- Returns same dict shape: `value, raw, candidates, confidence, score, strong_score`.

- [ ] Step 1: Write offline test: `read_cell` with `grid_text=[{text:"7", score:0.9}]` returns value 7; with two boxes `[{text:"7"},{text:"5"}]` joined → value 75; empty grid_text → BLANK.
- [ ] Step 2: Implement grid_text branch in `read_cell`.
- [ ] Step 3: Run offline test → PASS; save notebook; `ast.parse`.
- [ ] Step 4: Commit snapshot.

### Task 4: Question readers use grid_text

**Files:** Modify cell 2 — `_read_a_question`, `_read_b_question`, `read_part_a`, `read_part_b`.

**Interfaces:**
- `_read_a_question(loaded, views, q, reverse=False)`: passes `views[q].get("grid_text")` to `read_cell`; keeps `dash_only`/`faint_ink` prechecks; `has_ink` gate unchanged.
- `_read_b_question`: same for `ea`, `eb`, `et` entries; single-10-mark logic unchanged.
- `read_part_a`/`read_part_b`: unchanged signatures, iterate same.

- [ ] Step 1: Add grid_text passing to `_read_a_question` and `_read_b_question`.
- [ ] Step 2: Save notebook; `ast.parse`; run offline harness (no OCR available locally, so structural check only).
- [ ] Step 3: Commit snapshot.

### Task 5: Total readers use grid_text; skip reverse second pass

**Files:** Modify cell 2 — `_read_handwritten_total`, `_initial_record`, `_read_second_pass`, `extract_record`.

**Interfaces:**
- `_read_handwritten_total(loaded, entry, maxv, expected, allow_half, grid_text=None)`: grid_text path; `_read_total_union`/`_read_component_total` disabled in grid mode.
- `_initial_record(loaded, image_path)`: calls `read_grid` once; builds `grid_texts` dict; passes to question readers and total readers.
- `_read_second_pass`: no-op in grid mode (single pass only).
- `extract_record`: unchanged; validation flags unchanged.

- [ ] Step 1: Implement grid_text wiring in `_initial_record`; `_read_second_pass` returns early in grid mode.
- [ ] Step 2: Save notebook; `ast.parse`; structural check.
- [ ] Step 3: Commit snapshot.

### Task 6: `analyze_sheet` render_views flag + checkpoint bump

**Files:** Modify cell 2 — `analyze_sheet(path_or_image, render_views=True)`, checkpoint version.

**Interfaces:**
- `analyze_sheet(path_or_image, render_views=True)`: when `render_views=False`, skips `render_cell` per cell (views = None, grid_text populated separately); `image=det['arr']` kept for inspection.
- `CHECKPOINT_VERSION = "dynamic-grid-v1"` (bumped from v3).

- [ ] Step 1: Add `render_views` parameter to `analyze_sheet`; both call sites (`analyze_sheet` in `_initial_record`, `extract_sheet`) updated together.
- [ ] Step 2: Bump checkpoint version; rename `CHECKPOINT_PATH` suffix to `grid`.
- [ ] Step 3: Save notebook; `ast.parse` all code cells; verify tail intact.
- [ ] Step 4: Commit snapshot.

### Task 7: Offline harness + grid map/parse test

**Files:** Create `tools/validate_grid.py`.

**Interfaces:**
- `load_pipeline()` execs cell 2.
- `grid_map(ns, path)` → runs `analyze_sheet(render_views=False)` + `read_grid` + `assign_to_cells`; prints per-cell assigned texts.
- Tests: printed label dropped; cell with no ink → no grid_text; multi-box cell joined.

- [ ] Step 1: Write `tools/validate_grid.py` with offline tests (no OCR model needed for mapping tests; uses synthetic detections).
- [ ] Step 2: Run `python3 tools/validate_grid.py` → mapping tests PASS.
- [ ] Step 3: Commit new file.

### Task 8: Colab verification plan (documented, not executed here)

**Files:** None (doc in plan).

- [ ] Step 1: Upload notebook to Colab; run cell 6 (model loading) with GPU runtime.
- [ ] Step 2: Run full-batch cell on the 10 ground-truth sheets.
- [ ] Step 3: Compare cell accuracy vs 169/176 (96.0%) baseline.
- [ ] Step 4: If <96%, list failing cells by stem/question for the fix round.

**Ruling:** No server fallback, no view voting, no expression fallback — strict single-pass per box. If 96% is not met on Colab, report failing cells rather than adding fallbacks (fallbacks violate approach A).
