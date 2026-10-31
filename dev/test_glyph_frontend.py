"""Tests for the glyph front-end.

Run with ``python3 dev/test_glyph_frontend.py`` (or pytest).

The regression tests use real sheets from ``Random_Images_Sample`` because the
failure being fixed only shows up on real handwriting: a digit written across
a printed rule, or tall enough to reach the next row.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

G = harness.load()
ROOT = harness.ROOT
SAMPLES = ROOT / "Random_Images_Sample"

IV_SHEET = SAMPLES / "IV__22AMC08__840030_616827.tif"
VIII_SHEET = SAMPLES / "BE_B.TECH VIII Sem (main) R22 -APRIL 2026 - Copy__22ADE31__751485.jpg"


def legacy_ink_mask(arr, orange):
    """The v6 mask, kept only as the regression reference.

    It erased every dark/coloured pixel inside the dilated orange rule band,
    which is exactly what cut pen strokes that crossed a table line.
    """
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    H, W = gray.shape
    k = max(11, (min(H, W) // 40) | 1)
    bg = cv2.medianBlur(gray, k)
    diff = bg.astype(np.int16) - gray.astype(np.int16)
    mad = float(np.median(np.abs(diff - np.median(diff))))
    dark = diff > max(12.0, 4.0 * mad)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    colored = (s > 70) & (v < 248)
    grid = cv2.dilate(orange, np.ones((3, 3), np.uint8), iterations=1) > 0
    ink = (dark | colored) & (~grid)
    ink = ink.astype(np.uint8)
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    return ink


# ----------------------------------------------------------------- synthetic

def _rule_image():
    """White page, one orange rule, one dark blue stroke written across it."""
    arr = np.full((80, 200, 3), 255, np.uint8)
    arr[38:42, 10:190] = (240, 120, 40)          # orange printed rule
    cv2.line(arr, (100, 12), (100, 66), (30, 40, 120), 5)   # pen across it
    return arr


def test_pen_mask_keeps_stroke_across_rule():
    arr = _rule_image()
    ink = G.pen_mask(arr)
    # the stroke survives where it crosses the rule
    assert ink[36:45, 99:102].sum() >= 12, "pen pixels on the rule were erased"
    # a column of the stroke above and below the rule is still connected
    col = ink[12:66, 98:103].sum(axis=1)
    assert col.min() >= 1, "stroke has a gap at the rule"


def test_pen_mask_drops_the_rule_itself():
    arr = _rule_image()
    ink = G.pen_mask(arr)
    assert ink[39:41, 20:80].sum() == 0, "printed rule was classified as ink"


def _blob(arr, x, y, w, h):
    cv2.rectangle(arr, (x, y), (x + w, y + h), (20, 30, 90), -1)


def test_glyph_joins_two_halves_broken_by_a_rule():
    arr = np.full((120, 200, 3), 255, np.uint8)
    # one digit written across a horizontal rule: two halves, 8 px apart
    # (the gap the removed rule leaves behind)
    _blob(arr, 60, 20, 26, 12)
    _blob(arr, 60, 40, 26, 12)
    ink = G.pen_mask(arr)
    det = {"arr": arr, "mask": np.zeros(arr.shape[:2], np.uint8),
           "A": {"v": [], "hlines": []}, "B": {"v": [], "hlines": [36]}}
    glyphs, _ = G.build_glyphs(ink, det, row_h=30)
    assert len(glyphs) == 1, f"halves stayed separate: {len(glyphs)} glyphs"


def test_glyph_keeps_stacked_digits_apart():
    arr = np.full((120, 200, 3), 255, np.uint8)
    # two full digits, one per row, comparable size, small gap
    _blob(arr, 60, 18, 26, 28)
    _blob(arr, 60, 52, 26, 28)
    ink = G.pen_mask(arr)
    det = {"arr": arr, "mask": np.zeros(arr.shape[:2], np.uint8),
           "A": {"v": [], "hlines": []}, "B": {"v": [], "hlines": [49]}}
    glyphs, _ = G.build_glyphs(ink, det, row_h=34)
    assert len(glyphs) == 2, f"two digits were merged into {len(glyphs)} glyph(s)"


def test_glyph_keeps_neighbouring_digits_apart():
    arr = np.full((80, 200, 3), 255, np.uint8)
    _blob(arr, 40, 20, 20, 34)   # "1"
    _blob(arr, 80, 20, 26, 34)   # "0" -> a genuine two-digit number
    ink = G.pen_mask(arr)
    det = {"arr": arr, "mask": np.zeros(arr.shape[:2], np.uint8),
           "A": {"v": [], "hlines": []}, "B": {"v": [], "hlines": []}}
    glyphs, _ = G.build_glyphs(ink, det, row_h=36)
    assert len(glyphs) == 2, "two digits of a number were merged"


def test_assignment_follows_ink_not_centre():
    arr = np.full((100, 200, 3), 255, np.uint8)
    # a tall glyph: 70% of its ink in the upper cell, 30% below the line
    _blob(arr, 60, 8, 26, 26)
    _blob(arr, 60, 36, 26, 12)
    ink = G.pen_mask(arr)
    det = {"arr": arr, "mask": np.zeros(arr.shape[:2], np.uint8),
           "A": {"v": [], "hlines": []}, "B": {"v": [], "hlines": []}}
    glyphs, glab = G.build_glyphs(ink, det, row_h=34)
    cells = [dict(group="B", key="Q6", row="a", x1=20, y1=4, x2=180, y2=38,
                  ref_h=34, kind="partB"),
             dict(group="B", key="Q6", row="b", x1=20, y1=38, x2=180, y2=72,
                  ref_h=34, kind="partB")]
    G.assign_glyphs(glyphs, glab, cells, det)
    a, b = cells
    assert len(a["comps"]) == 1 and not b["comps"], "tall glyph was split across rows"
    assert glyphs[0]["shares"][0] > 0.6 and glyphs[0]["shares"][1] > 0.1


# -------------------------------------------------------------------- real

def _glyphs_of(path):
    arr, det, views = G.extract_sheet(str(path))
    return arr, det, views


def test_crop_never_clips_a_glyph():
    for path in (IV_SHEET, VIII_SHEET):
        _arr, _det, views = _glyphs_of(path)
        for key, entry in views.items():
            v = entry.get("views")
            if not v:
                continue
            xa, ya, xb, yb = v["box"]
            for gx, gy, gw, gh in v["glyphs"]:
                assert xa <= gx and gx + gw <= xb, f"{key}: glyph clipped horizontally"
                assert ya <= gy and gy + gh <= yb, f"{key}: glyph clipped vertically"


def test_binary_view_is_proportional_to_box():
    """The expression splitter maps binary x positions back through ``box``."""
    _arr, _det, views = _glyphs_of(IV_SHEET)
    checked = 0
    for key, entry in views.items():
        v = entry.get("views") or {}
        binv, box = v.get("binary"), v.get("box")
        if binv is None or not box:
            continue
        bh, bw = binv.shape[:2]
        bx, by = box[2] - box[0], box[3] - box[1]
        assert abs((bw / bh) - (bx / by)) <= 0.06 * max(1.0, bx / by), (
            f"{key}: binary {bw}x{bh} does not match box {bx}x{by}")
        checked += 1
    assert checked > 10, f"only {checked} binary views checked"


def test_touching_digits_keep_whole_glyph_and_row_crop():
    """22AMC08 Q6 is a 4 in row a touching a 3 in row b.

    The two digits must not be cut at the rule (the whole glyph is what a
    tall digit needs), but row b must still get a crop of its 3 instead of a
    clean blank, and row a must have a row-restricted fallback for the 4.
    """
    _arr, det, views = _glyphs_of(IV_SHEET)
    a, b = views["Q6_a"], views["Q6_b"]
    row_h = a["cell"]["ref_h"]
    glyph = max(a["cell"]["comps"], key=lambda g: g["h"])
    assert glyph["h"] > 1.5 * row_h, (
        f"the merged glyph was cut at the row line (h={glyph['h']}, row={row_h})")
    # the crop is the whole glyph, not the cell
    assert a["views"]["box"][3] > a["cell"]["y2"]
    # row a can still try just its own part
    assert a["views"].get("part") is not None
    # row b is not blank: it holds the 3 that the merge took away
    assert b["has_ink"] and b["views"] and b["views"].get("shared")
    assert b["views"]["box"][3] <= b["cell"]["y2"] + 0.35 * row_h, (
        "row b's shared crop reaches into the next row")
    assert b["ambiguous"]


def test_ambiguity_is_not_flagged_for_tall_single_digits():
    """A tall digit's thin descender must not turn the next row into REVIEW."""
    flags = {k: e["ambiguous"] for k, e in _glyphs_of(IV_SHEET)[2].items()}
    # Q15_b is an ordinary mark of its own; the tall total below it must not
    # mark it ambiguous.
    assert flags["Q15_b"] is False, "a neighbour's tall total flagged Q15_b"


def test_no_cell_is_rendered_empty_when_it_has_ink():
    _arr, _det, views = _glyphs_of(IV_SHEET)
    for key, entry in views.items():
        if entry.get("has_ink"):
            assert entry.get("color") is not None, f"{key} has ink but no crop"
            assert entry["views"]["box"][2] > entry["views"]["box"][0]


def test_pen_mask_recovers_ink_the_old_mask_lost():
    """The old mask erased pen pixels inside the dilated rule band.

    Ground truth for "pen" is not available, so the test measures *coverage*
    of dark non-orange pixels inside the removed band: the old mask keeps
    fewer than half of them on the IV sheet, the new mask keeps most.  A few
    dark specks on the rules themselves are correctly dropped, which is why
    the target is coverage rather than zero loss.
    """
    for path, min_gain in ((IV_SHEET, 0.70), (VIII_SHEET, 0.60)):
        arr = np.array(G.ImageOps.exif_transpose(G.Image.open(str(path))).convert("RGB"))
        det = G.detect_layout(arr)
        old = legacy_ink_mask(det["arr"], det["mask"])
        new = G.pen_mask(det["arr"], det["mask"], G._grid_lines(det))
        hsv = cv2.cvtColor(det["arr"], cv2.COLOR_RGB2HSV)
        h, s, v = hsv[:, :, 0].astype(int), hsv[:, :, 1].astype(int), hsv[:, :, 2].astype(int)
        band = cv2.dilate(det["mask"], np.ones((5, 5), np.uint8)) > 0
        # Dark and not orange: orange is the printed rule, and a rule pixel is
        # not ink however dark the scan made it.
        orange = ((h <= 32) | (h >= 160)) & (s >= 60) & (v >= 40)
        pen_in_band = (v < 150) & band & ~orange
        n = int(pen_in_band.sum())
        assert n > 200, f"{path.name}: test premise broken, band holds only {n} dark px"
        cov_old = float((pen_in_band & (old > 0)).sum()) / n
        cov_new = float((pen_in_band & (new > 0)).sum()) / n
        # The new mask must never keep less of the band than the old one did.
        assert cov_new >= cov_old, (
            f"{path.name}: new mask keeps less band ink ({cov_old:.2f} -> {cov_new:.2f})")
        assert cov_new >= min_gain, (
            f"{path.name}: new mask keeps only {cov_new:.2f} of band ink (old {cov_old:.2f})")
        if path is IV_SHEET:
            # On this sheet the old mask lost over half of the ink that
            # crossed a rule, which is the regression this pipeline fixes.
            assert cov_old <= 0.55, (
                f"{path.name}: test premise broken, old mask already kept {cov_old:.2f}")


def test_sample_sheets_all_render():
    """Every sample must segment and produce crops without raising."""
    failures = []
    checked = 0
    for name in sorted(os.listdir(SAMPLES)):
        if not name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
            continue
        path = SAMPLES / name
        checked += 1
        try:
            arr, det, views = G.extract_sheet(str(path))
        except Exception as exc:                      # noqa: BLE001
            failures.append((name, repr(exc)))
            continue
        ink_cells = [k for k, e in views.items() if e.get("has_ink")]
        if not ink_cells:
            failures.append((name, "no cells with ink"))
        for key, entry in views.items():
            v = entry.get("views")
            if not v:
                continue
            xa, ya, xb, yb = v["box"]
            for gx, gy, gw, gh in v["glyphs"]:
                if not (xa <= gx and gx + gw <= xb and ya <= gy and gy + gh <= yb):
                    failures.append((name, f"{key}: clipped glyph"))
                    break
        if checked >= 40:
            break
    assert not failures, failures[:10]


def main():
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {name}: {exc}")
        except Exception as exc:                       # noqa: BLE001
            failed += 1
            print(f"ERROR {name}: {exc!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
