# Region-Based Crop Ownership — Design

**Date:** 2026-10-28
**Status:** Approved
**Notebook:** `Mark Sheet OCR Pilot Fixes v6.ipynb`

## Intent

Many students write marks larger than the printed cell, so part of the digit sits outside the grid. The current pipeline cuts those digits off before OCR sees them. In the worst cases the crop sent to OCR contains no handwriting at all — only a fragment of the printed rule.

The cause is that stroke ownership and crop geometry are the same thing today: `render_cell` builds the crop from the bounding box of the components assigned to the cell, so a mis-assigned or mis-filtered component *is* the crop.

This design separates the two. Ownership keeps deciding whether a cell is blank, a dash, or faint. Crop geometry comes from the printed grid.

## Goals

- Every cell's `color` crop covers the full printed cell plus a margin on all four sides.
- A digit spilling into a neighbour's space is visible up to the midpoint between the two cells — never cut earlier.
- No crop can consist only of a printed rule.
- Vertical reach is ±0.55 row heights, so tops and bottoms that leave the row are included.
- Contamination is bounded by construction: at most the near half of a neighbour's cell.

## Non-goals

- Zero contamination. That was the rejected alternative (splitting every blob at cell boundaries), which leaves large digits cut.
- Changing layout detection, bubble reading, the OCR model, value validation, or Part A/B totals.
- Changing `assign_components` scoring, `cell_owner_map`, or `split_components_by_owner`.

## Constraints

- Product code stays in cell 2 of the notebook.
- `read_cell` consumes `binary`, `gray`, `context`, `clip`, `dark`, `expanded`, `tall`, `large`, `overflow_x/y`, `box`; all keys and meanings are preserved.
- Dash→0 and blank detection behave as today for cells that are already correct.
- The notebook is edited through JSON, not raw text.

## Approaches considered

**A. Split every blob at cell boundaries.** Pixel ownership by rectangle, splitting allowed across questions. Zero contamination, but a digit written into a filled neighbour is still cut at the boundary — Q14(b)'s "4" would stay incomplete.

**B. Majority-ink ownership — dominant cell keeps the whole digit.** Score components by ink overlap rather than centroid. Big digits are never cut, but when two marks touch the losing cell goes blank: a half digit is traded for a missing one.

**C. Separate ownership from crop geometry — chosen.** Crop = printed cell + margin, clipped at the midpoint to neighbours but never inside the cell. Ownership no longer affects the crop.

C was chosen because it fixes both failure families (bare-rule crops and cut-off digits) with the smallest structural change, and because it degrades gracefully: contamination is capped at half a neighbour's cell rather than being unbounded.

## How it works

### 1. `cell_region(cell, cells, det, pad_x=0.18, pad_y=0.55)`

Returns `(x1, y1, x2, y2)`:

- start from the cell expanded by `0.18 · cell_width` horizontally and `0.55 · ref_h` vertically;
- for every cell that overlaps in the perpendicular axis, cap the expansion at the **neighbour's centre**. Printed cells are contiguous, so the boundary *between* two cells is exactly their shared edge — capping there would erase the whole margin and leave the crop equal to the cell. The neighbour's centre keeps the margin while bounding contamination to less than half a neighbour's cell;
- never clip inside the printed cell — the bound is always `max(cell.x2, min(bound, cap))` (and the mirror for the other three sides);
- clamp to the image.

The "never clip inside the cell" rule matters at the rightmost question column, where `TOTAL_B.x1` overlaps the cell edge.

### 2. `printed_rule_band(det, half=3)`

The layout detector already knows every rule coordinate (`A.v`, `A.hlines`, `B.v`, `B.hlines`). Build a boolean image that is `True` on a ±3 px band around each rule.

### 3. `region_paint_mask(arr, orange, band, region)`

Inside the region, paint pixels that are ink and not printed grid:

- `gray < 225` or saturation `> 90` (pen is saturated; pencil and print are not);
- outside the orange mask;
- inside a rule band, only if the pixel is clearly coloured pen (`s > 90`). This removes the neutral-dark core of a printed rule while letting a blue stroke that crosses the rule survive.

If the mask is empty the cell renders `None` — so a region holding only a rule is genuinely blank.

### 4. `is_rule_remnant` guard

A component is a rule remnant when it is thin (`w ≤ max(4, 0.12 · ref_h)`), sits within `max(3, 0.12 · ref_h)` px of the cell's left or right edge, and spans at least `0.60 · ref_h` of the row. This catches fragments that fall outside the detected line bands. A handwritten `1` is thin and tall but is written inside the cell, not against the printed edge.

### 5. What stays

`assign_components`, `split_components_by_owner`, `cell_owner_map`, `cell_has_ink`, `dash_only`, `faint_ink`, and the overflow/large flags are untouched. The `clip` view remains cell-only, so OCR still has a zero-contamination option when wider views disagree.

## Validation

`tools/validate_crops.py` execs cell 2 and asserts, on the reference sheet and on a 10-image sweep:

- every rendered crop contains its printed cell;
- a rule-only region renders `None`;
- pen crossing a rule survives;
- regions extend into an empty neighbour but stop at a filled neighbour's midpoint;
- regions never clip inside their own cell.

Visual confirmation via regenerated `crop_debug/` montages: `Q15_a`, `Q13_a`, `Q6_total`, `Q15_total` must show their digit; `Q14_b` must show "4" without Q15(b)'s "2"; `Q1`–`Q5`, `TOTAL_A`, `Q12_a` must be unchanged.

OCR-accuracy comparison requires PaddleOCR, which is not installed in this environment; it is left to the GPU environment.

## Known risk

Region-based painting means a cell with no assigned components can now render a crop. That is how `Q13_a` recovers its digit, but it decouples render output from `cell_has_ink`. If OCR starts inventing marks on genuinely blank cells, gate `render_cell` on `cell_has_ink(cell)`.
