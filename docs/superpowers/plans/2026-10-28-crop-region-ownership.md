# Region-Based Crop Ownership Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make each mark-sheet cell's OCR crop cover the full printed cell plus a bounded margin, so digits written large are never cut off, and printed rule fragments can never become a crop.

**Architecture:** Crop geometry stops depending on component ownership. A new `cell_region()` computes a rectangle per cell from the printed grid, expanded by a margin and capped at the centre of the neighbouring cell, never clipping inside the cell itself. `render_cell()` paints every ink pixel inside that region (minus the printed grid, removed using the already-detected rule coordinates) instead of painting only the components assigned to the cell. Ownership keeps deciding blank/dash/faint only.

**Tech Stack:** Python, NumPy, OpenCV, PIL, Jupyter notebook. No OCR model needed for validation.

**Spec:** `/Users/bhargav/Downloads/DL_check/docs/superpowers/specs/2026-10-28-crop-region-ownership-design.md`

## Global Constraints

- All product code lives in **cell 2** of `Mark Sheet OCR Pilot Fixes v6.ipynb`. Do not add new notebook cells.
- Edit the notebook through `json.load` / `json.dump` and validate with `ast.parse` before writing. Raw text edits corrupt the JSON-escaped source.
- Do not change: `detect_layout`, `cell_defs`, `_components`, `assign_components`, `cell_owner_map`, `split_components_by_owner`, `read_bubble_marks`, `read_cell`, the OCR model loading, or any Part A/B total logic.
- `render_cell` must keep returning the same keys: `binary`, `gray`, `context`, `clip`, `dark`, `expanded`, `tall`, `tall_raw`, `tall_box`, `large`, `overflow_x`, `overflow_y`, `box`.
- Region constants: `pad_x = 0.18` of cell width, `pad_y = 0.55` of `ref_h`, rule band half-width `3` px.
- A region must always fully contain its own printed cell: clamp with `max(cell['x2'], min(bound, cap))`, never clip inside the cell.
- **Both** `render_cell` call sites must be updated together: `analyze_sheet` and `extract_sheet`.
- No git repository exists in the working directory. "Save" means rewrite the notebook JSON and re-verify; there are no commit steps. Review diffs are produced by exporting cell 2's source before and after each task and running `diff -u`.

## Review Focus

- **A cell whose neighbour is empty but whose digit fills the neighbour's space** — the digit must be fully visible, not clipped at the cell edge. Covered by Task 2 test `test_region_extends_into_empty_neighbour`.
- **Two filled neighbouring cells whose marks touch** — the crop may reach at most the neighbour's centre. Covered by Task 2 test `test_region_stops_at_neighbour_centre`.
- **A blank cell containing only a printed rule fragment** — must produce `None`, not a crop containing a line. Covered by Task 3 test `test_rule_only_region_renders_none`.
- **Handwriting crossing a printed rule** — must survive the rule removal. Covered by Task 3 test `test_pen_crossing_rule_survives`.
- **The rightmost question column next to the Total(B) column**, where `TOTAL_B.x1` overlaps the cell edge — the region must not shrink below the printed cell. Covered by Task 2 test `test_region_never_clips_inside_cell`.
- **`overflow_x` must still mean "ink leaves the cell"**, not "the region is wider than the cell" — otherwise `large` is never set and view selection silently changes. Covered by Task 3 test `test_overflow_flags_reflect_ink`.

---

## File Structure

| Path | Responsibility |
|------|----------------|
| `Mark Sheet OCR Pilot Fixes v6.ipynb` (cell 2) | All pipeline code: `cell_region`, `printed_rule_band`, `region_paint_mask`, `render_cell`, `is_rule_remnant` |
| `tools/validate_crops.py` | Standalone harness: execs cell 2, runs `analyze_sheet` on sample images, asserts crop invariants |
| `crop_debug/` | Regenerated visual output for human review (inspection only) |

`tools/validate_crops.py` is the only new file. It imports the pipeline by exec'ing cell 2's source, so it needs no notebook execution.

---

### Task 1: Validation harness

**Files:**
- Create: `tools/validate_crops.py`
- Test: `tools/validate_crops.py` (run directly)

**Interfaces:**
- Consumes: nothing (loads the notebook itself)
- Produces: `load_pipeline() -> dict`, `analyze(ns, path) -> dict`, `check_crops(result) -> list[str]`, and a `--check` mode that exits non-zero on invariant failure. Later tasks add assertions into this file.

- [ ] **Step 1: Write the failing harness**

```python
#!/usr/bin/env python3
"""Validate that per-cell OCR crops contain the whole cell and no bare rules."""
import argparse
import json
import sys

import numpy as np
from PIL import Image, ImageOps  # noqa: F401  (kept for later tasks)

NOTEBOOK = "Mark Sheet OCR Pilot Fixes v6.ipynb"
REFERENCE = ("Random_Images_Sample/"
             "BE_B.TECH VIII Sem (main) R22 -APRIL 2026 - Copy__22CAE11__751676.jpg")


def load_pipeline():
    with open(NOTEBOOK) as f:
        nb = json.load(f)
    src = "".join(nb["cells"][2]["source"])
    ns = {}
    exec(compile(src, "<pipeline>", "exec"), ns)
    return ns


def analyze(ns, path):
    return ns["analyze_sheet"](path)


def check_crops(result):
    """Return a list of human-readable failures; empty means pass."""
    failures = []
    for key, entry in result["views"].items():
        cell = entry["cell"]
        if not entry["views"]:
            continue
        box = entry["views"].get("box")
        if box is None:
            failures.append(f"{key}: missing box")
            continue
        xa, ya, xb, yb = box
        if xa > cell["x1"] or ya > cell["y1"] or xb < cell["x2"] or yb < cell["y2"]:
            failures.append(
                f"{key}: crop {box} does not contain cell "
                f"({cell['x1']},{cell['y1']},{cell['x2']},{cell['y2']})")
    return failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--image", default=REFERENCE)
    args = ap.parse_args()

    ns = load_pipeline()
    result = analyze(ns, args.image)
    failures = check_crops(result)
    for f in failures:
        print("FAIL", f)
    print(f"{len(result['views'])} cells checked, {len(failures)} failures")
    if args.check and failures:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `python3 tools/validate_crops.py --check`
Expected: non-zero exit, with `FAIL` lines for cells whose tight crop does not contain the printed cell — at minimum `Q15_a`, `Q13_a`, `Q6_total`, `Q15_total`, `Q4`, `Q9_a`, `Q13_b`.

- [ ] **Step 3: Save nothing**

The harness is standalone. Do not modify the notebook in this task.

---

### Task 2: `cell_region()` geometry

**Files:**
- Modify: `Mark Sheet OCR Pilot Fixes v6.ipynb`, cell 2, insert immediately before `def render_cell`
- Test: `tools/validate_crops.py` (new `test_region_*` functions)

**Interfaces:**
- Consumes: `cells` from `cell_defs(det)` after `assign_components`; each has `x1, x2, y1, y2, ref_h, group, key, row`
- Produces: `cell_region(cell: dict, cells: list, det: dict, pad_x: float = 0.18, pad_y: float = 0.55) -> tuple[int, int, int, int]` returning `(x1, y1, x2, y2)` in image pixels, clamped to the image.

**Ruling (recorded in ledger):** the cap is the **neighbour's centre**, not the boundary between the two cells. Printed cells are contiguous, so their shared boundary *is* the midpoint — capping there would erase the entire margin and leave the crop equal to the cell. The neighbour's centre keeps the margin (0.18 cell widths) while bounding contamination to less than half a neighbour's cell, as the spec requires.

- [ ] **Step 1: Write the failing geometry tests**

Add to `tools/validate_crops.py`:

```python
def test_region_extends_into_empty_neighbour(ns, result):
    cells = result["cells"]
    by_key = {(c["group"], c["key"], c.get("row")): c for c in cells}
    q1 = by_key[("A", "Q1", None)]
    x1, y1, x2, y2 = ns["cell_region"](q1, cells, result["det"])
    cw = q1["x2"] - q1["x1"]
    assert x2 > q1["x2"] + 0.10 * cw, f"Q1 right margin only {x2 - q1['x2']}"


def test_region_stops_at_neighbour_centre(ns, result):
    cells = result["cells"]
    by_key = {(c["group"], c["key"], c.get("row")): c for c in cells}
    q1 = by_key[("A", "Q1", None)]
    q2 = by_key[("A", "Q2", None)]
    x1, y1, x2, y2 = ns["cell_region"](q1, cells, result["det"])
    assert x2 <= (q2["x1"] + q2["x2"]) / 2 + 1


def test_region_never_clips_inside_cell(ns, result):
    cells = result["cells"]
    for c in cells:
        x1, y1, x2, y2 = ns["cell_region"](c, cells, result["det"])
        assert x1 <= c["x1"] and x2 >= c["x2"], c["key"]
        assert y1 <= c["y1"] and y2 >= c["y2"], c["key"]


def test_region_vertical_reach(ns, result):
    cells = result["cells"]
    by_key = {(c["group"], c["key"], c.get("row")): c for c in cells}
    q6a = by_key[("B", "Q6", "a")]
    x1, y1, x2, y2 = ns["cell_region"](q6a, cells, result["det"])
    assert y1 <= q6a["y1"] - 0.4 * q6a["ref_h"]
    assert y2 >= q6a["y2"] + 0.2 * q6a["ref_h"]
```

Wire them into `main()` after `check_crops`:

```python
    for fn in (test_region_extends_into_empty_neighbour,
               test_region_stops_at_neighbour_centre,
               test_region_never_clips_inside_cell,
               test_region_vertical_reach):
        try:
            fn(ns, result)
        except Exception as exc:
            failures.append(f"{fn.__name__}: {exc}")
```

- [ ] **Step 2: Run and confirm the new tests fail**

Run: `python3 tools/validate_crops.py --check`
Expected: `KeyError: 'cell_region'` reported for all four tests.

- [ ] **Step 3: Implement `cell_region` in cell 2**

Insert before `def render_cell`:

```python
def cell_region(cell, cells, det, pad_x=0.18, pad_y=0.55):
    """Rectangle an OCR crop may use: the printed cell plus a bounded margin.

    The margin may reach into a neighbour, but never past that neighbour's
    centre, and never inside the printed cell itself.
    """
    H, W = det['arr'].shape[:2]
    cw = max(1.0, cell['x2'] - cell['x1'])
    ch = max(1.0, cell.get('ref_h', cell['y2'] - cell['y1']))
    x1 = cell['x1'] - pad_x * cw
    x2 = cell['x2'] + pad_x * cw
    y1 = cell['y1'] - pad_y * ch
    y2 = cell['y2'] + pad_y * ch
    for o in cells:
        if o is cell:
            continue
        if min(cell['y2'], o['y2']) > max(cell['y1'], o['y1']):
            if o['x1'] >= cell['x2']:
                x2 = max(cell['x2'], min(x2, (o['x1'] + o['x2']) / 2.0))
            elif o['x2'] <= cell['x1']:
                x1 = min(cell['x1'], max(x1, (o['x1'] + o['x2']) / 2.0))
        if min(cell['x2'], o['x2']) > max(cell['x1'], o['x1']):
            if o['y1'] >= cell['y2']:
                y2 = max(cell['y2'], min(y2, (o['y1'] + o['y2']) / 2.0))
            elif o['y2'] <= cell['y1']:
                y1 = min(cell['y1'], max(y1, (o['y1'] + o['y2']) / 2.0))
    return (max(0, int(x1)), max(0, int(y1)), min(W, int(x2)), min(H, int(y2)))
```

- [ ] **Step 4: Run tests and confirm they pass**

Run: `python3 tools/validate_crops.py --check`
Expected: the four `test_region_*` tests pass. The `check_crops` failures remain — `render_cell` does not use the region yet.

- [ ] **Step 5: Save the notebook**

Export cell 2's source to a temp file, rewrite the notebook through `json.load`/`json.dump`, then run:
`python3 -c "import json,ast;nb=json.load(open('Mark Sheet OCR Pilot Fixes v6.ipynb'));[ast.parse(''.join(c['source'])) for c in nb['cells'] if c['cell_type']=='code'];print('ok')"`
Expected: `ok`

---

### Task 3: Rule band and region paint mask

**Files:**
- Modify: `Mark Sheet OCR Pilot Fixes v6.ipynb`, cell 2 — add `printed_rule_band`, `region_paint_mask`; rewrite the opening of `render_cell`; update both call sites
- Test: `tools/validate_crops.py`

**Interfaces:**
- Consumes: `det['A']['v']`, `det['A']['hlines']`, `det['B']['v']`, `det['B']['hlines']` (pixel coordinates), `det['mask']` (orange grid mask), `cell_region()`
- Produces:
  - `printed_rule_band(det: dict, half: int = 3) -> np.ndarray[bool]` — `True` on printed rule pixels
  - `region_paint_mask(arr, orange, band, region) -> np.ndarray[bool]` — shape `(y2-y1, x2-x1)`
  - `render_cell(cell, arr, orange, cells, det, target_h=64) -> tuple[np.ndarray | None, dict | None]`

- [ ] **Step 1: Write the failing tests**

Add to `tools/validate_crops.py`:

```python
def test_crop_contains_full_cell(ns, result):
    failures = check_crops(result)
    assert not failures, failures[:5]


def test_rule_only_region_renders_none(ns):
    """A grey printed rule on white must paint nothing."""
    arr = np.full((20, 20, 3), 255, np.uint8)
    arr[2:18, 9:11] = (150, 150, 150)          # neutral grey rule
    orange = np.zeros((20, 20), np.uint8)
    band = np.zeros((20, 20), bool)
    band[:, 9:12] = True
    paint = ns["region_paint_mask"](arr, orange, band, (0, 0, 20, 20))
    assert paint.sum() == 0, f"{paint.sum()} rule pixels survived"


def test_pen_crossing_rule_survives(ns):
    """A saturated pen stroke inside a rule band must be kept."""
    arr = np.full((20, 20, 3), 255, np.uint8)
    arr[2:18, 9:11] = (40, 40, 200)            # blue pen
    orange = np.zeros((20, 20), np.uint8)
    band = np.zeros((20, 20), bool)
    band[:, 9:12] = True
    paint = ns["region_paint_mask"](arr, orange, band, (0, 0, 20, 20))
    assert paint.sum() >= 20, f"pen crossing the rule was erased ({paint.sum()} px)"


def test_overflow_flags_reflect_ink(ns, result):
    """overflow_x must describe the ink, not the widened region."""
    entry = result["views"]["Q1"]
    assert entry["views"]["overflow_x"] is False, "Q1 ink is inside its cell"
```

- [ ] **Step 2: Run and confirm failure**

Run: `python3 tools/validate_crops.py --check`
Expected: `KeyError: 'printed_rule_band'`.

- [ ] **Step 3: Implement the two helpers in cell 2**

```python
def printed_rule_band(det, half=3):
    """True on the detected printed column/row rules, widened by `half` px."""
    H, W = det['arr'].shape[:2]
    band = np.zeros((H, W), bool)
    for g in ('A', 'B'):
        for x in det[g]['v']:
            xi = int(round(x))
            band[:, max(0, xi - half):min(W, xi + half + 1)] = True
        for y in det[g]['hlines']:
            yi = int(round(y))
            band[max(0, yi - half):min(H, yi + half + 1), :] = True
    return band


def region_paint_mask(arr, orange, band, region):
    """Ink pixels inside `region` that are not printed grid."""
    x1, y1, x2, y2 = region
    sub = arr[y1:y2, x1:x2]
    if sub.size == 0:
        return np.zeros((0, 0), bool)
    gray = cv2.cvtColor(sub, cv2.COLOR_RGB2GRAY)
    s = cv2.cvtColor(sub, cv2.COLOR_RGB2HSV)[:, :, 1]
    pen = s > 90                      # blue/violet pen is saturated
    rule = band[y1:y2, x1:x2] & ~pen  # neutral print inside a rule band
    paint = ((gray < 225) | pen) & (orange[y1:y2, x1:x2] == 0) & ~rule
    return paint
```

- [ ] **Step 4: Rewire `render_cell`**

Change the signature to `def render_cell(cell, arr, orange, cells, det, target_h=64):` and replace the body from the old first statement (`chosen = select_render_comps(cell)`) through `out[km] = crop[km]` with:

```python
    rx1, ry1, rx2, ry2 = cell_region(cell, cells, det)
    band = printed_rule_band(det)
    paint = region_paint_mask(arr, orange, band, (rx1, ry1, rx2, ry2))
    if not paint.any():
        return None, None
    crop = arr[ry1:ry2, rx1:rx2].copy()
    out = np.full_like(crop, 255)
    out[paint] = crop[paint]
    xa, ya, xb, yb = rx1, ry1, rx2, ry2
    mask = paint
    dil = cv2.dilate(paint.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
    pys, pxs = np.where(paint)
    x1, x2 = rx1 + int(pxs.min()), rx1 + int(pxs.max()) + 1
    y1, y2 = ry1 + int(pys.min()), ry1 + int(pys.max()) + 1
```

Everything after that point stays unchanged. `x1..y2` now describe the **ink** bounding box, so the `overflow_x`/`overflow_y` flags keep their original meaning and `large` is still set for tall totals.

Update **both** call sites:
- in `analyze_sheet`: `color, views = render_cell(c, det['arr'], orange, cells, det)`
- in `extract_sheet`: `color, views = render_cell(c, det['arr'], orange, cells, det)`

`lab` is no longer used by `render_cell`; leave `lab` and `ink` in the callers untouched (they are used by `assign_components`).

- [ ] **Step 5: Run tests**

Run: `python3 tools/validate_crops.py --check`
Expected: all tests pass, `0 failures`.

- [ ] **Step 6: Save the notebook**

Same export + JSON rewrite + `ast.parse` check as Task 2 Step 5.

---

### Task 4: Edge-spanning rule guard in `is_rule_remnant`

**Files:**
- Modify: `Mark Sheet OCR Pilot Fixes v6.ipynb`, cell 2, `is_rule_remnant`
- Test: `tools/validate_crops.py`

**Interfaces:**
- Consumes: `comp` dicts (`x, y, w, h, area`), `cell` dicts (`x1, x2, ref_h`)
- Produces: same signature `is_rule_remnant(comp, cell) -> bool`, stricter

- [ ] **Step 1: Write the failing test**

```python
def test_tall_thin_edge_fragment_rejected(ns, result):
    cells = result["cells"]
    by_key = {(c["group"], c["key"], c.get("row")): c for c in cells}
    cell = by_key[("B", "Q15", "a")]
    rh = cell["ref_h"]
    frag = dict(id=-1, x=int(cell["x1"]), y=int(cell["y1"]) + 2,
                w=4, h=int(0.8 * rh), area=4 * int(0.8 * rh),
                cx=float(cell["x1"]) + 2, cy=float(cell["y1"]) + 2)
    assert ns["is_rule_remnant"](frag, cell), "0.8-row tall 4px fragment accepted"
```

- [ ] **Step 2: Run and confirm failure**

Run: `python3 tools/validate_crops.py --check`
Expected: `AssertionError: 0.8-row tall 4px fragment accepted`.

- [ ] **Step 3: Implement the guard**

Replace the body of `is_rule_remnant` with:

```python
def is_rule_remnant(comp, cell):
    """True for a thin fragment of the printed cell rule.

    A printed rule fragment is thin, spans most of the row, and sits against
    the cell's left or right edge.  A handwritten ``1`` is also thin and tall,
    but it is written inside the cell rather than hugging the printed edge.
    """
    rh = max(1.0, cell.get('ref_h', cell['y2'] - cell['y1']))
    if comp['w'] > max(4.0, 0.12 * rh):
        return False
    margin = max(3.0, 0.12 * rh)
    left = comp['x'] - cell.get('x1', comp['x'])
    right = cell.get('x2', comp['x'] + comp['w']) - (comp['x'] + comp['w'])
    if left > margin and right > margin:
        return False
    return comp['h'] >= 0.60 * rh
```

- [ ] **Step 4: Run tests**

Run: `python3 tools/validate_crops.py --check`
Expected: all pass, `0 failures`.

- [ ] **Step 5: Save the notebook**

Same JSON rewrite + `ast.parse` check.

---

### Task 5: Sample sweep and visual review

**Files:**
- Modify: `tools/validate_crops.py` (add `--sweep`)
- Create: `crop_debug/` output (regenerated)

**Interfaces:**
- Consumes: `cell_region`, `render_cell`, `analyze_sheet`
- Produces: `sweep(ns, paths) -> list[str]` failures across images

- [ ] **Step 1: Add the sweep**

```python
def sweep(ns, paths):
    failures = []
    for path in paths:
        try:
            result = analyze(ns, path)
        except Exception as exc:
            failures.append(f"{path}: pipeline error {exc}")
            continue
        for msg in check_crops(result):
            failures.append(f"{path}: {msg}")
    return failures
```

In `main()`, add `--sweep N` (default 10) that globs `Random_Images_Sample/*.jpg`, takes the first N, and appends `sweep(...)` results to `failures`.

- [ ] **Step 2: Run the sweep**

Run: `python3 tools/validate_crops.py --check --sweep 10`
Expected: `0 failures`. Any failure names the image and cell — fix `cell_region`'s clamping, not the test.

- [ ] **Step 3: Regenerate the crop montages**

Re-run the `crop_debug` generator (montage per cell plus `_overview_all_cells.png` and `_page_boxes.png`) against the reference sheet.

- [ ] **Step 4: Visual check**

Open `crop_debug/_overview_all_cells.png` and confirm:
- `Q15_a`, `Q13_a`, `Q6_total`, `Q15_total` show their handwritten digit instead of a bare vertical line;
- `Q4`, `Q9_a`, `Q13_b` show the full digit body;
- `Q14_b` still shows a "4" and does not show Q15(b)'s "2";
- `Q1`–`Q5`, `TOTAL_A`, `Q12_a` look the same as before.

- [ ] **Step 5: Save the notebook and report**

Same JSON rewrite + `ast.parse` check. Report the before/after cell list.

---

## Self-Review

**Spec coverage:** Region geometry → Task 2. Region-based paint + rule removal by known line coordinates → Task 3. Blank-on-rule-only → Task 3 synthetic test. `is_rule_remnant` shape guard → Task 4. Bounded contamination → Task 2 `test_region_stops_at_neighbour_centre`. Overflow-flag semantics → Task 3 `test_overflow_flags_reflect_ink`. Validation on sample set → Task 5.

**Type consistency:** `cell_region(cell, cells, det, pad_x, pad_y) -> (x1, y1, x2, y2)` used identically in Tasks 2, 3, 5. `printed_rule_band(det, half=3)` and `region_paint_mask(arr, orange, band, region)` defined and consumed in Task 3. `render_cell` loses `lab` and gains `cells, det` in Task 3; both call sites updated in the same step.

**Known risk:** region-based painting means a cell with no assigned components can now render a crop. That is how `Q13_a` recovers its digit, but it decouples render output from `cell_has_ink`. `cell_has_ink` is unchanged and still governs `has_ink`/`dash_only`/`faint_ink`; only the crop is broader. If OCR then invents marks on genuinely blank cells, gate `render_cell` on `cell_has_ink(cell)` — note this in the Task 5 report rather than pre-emptively changing it.
