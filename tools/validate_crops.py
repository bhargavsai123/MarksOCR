#!/usr/bin/env python3
"""Validate that per-cell OCR crops contain the whole cell and no bare rules."""
import argparse
import glob
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


def test_tall_thin_edge_fragment_rejected(ns, result):
    cells = result["cells"]
    by_key = {(c["group"], c["key"], c.get("row")): c for c in cells}
    cell = by_key[("B", "Q15", "a")]
    rh = cell["ref_h"]
    frag = dict(id=-1, x=int(cell["x1"]), y=int(cell["y1"]) + 2,
                w=4, h=int(0.8 * rh), area=4 * int(0.8 * rh),
                cx=float(cell["x1"]) + 2, cy=float(cell["y1"]) + 2)
    assert ns["is_rule_remnant"](frag, cell), "0.8-row tall 4px fragment accepted"


def test_tall_thin_fragment_flush_with_edge_rejected(ns, result):
    """A tall thin fragment flush against the printed line is a remnant."""
    cells = result["cells"]
    by_key = {(c["group"], c["key"], c.get("row")): c for c in cells}
    cell = by_key[("B", "Q15", "a")]
    rh = cell["ref_h"]
    frag = dict(id=-1, x=int(cell["x1"]), y=int(cell["y1"]) + 2,
                w=4, h=int(0.8 * rh), area=4 * int(0.8 * rh),
                cx=float(cell["x1"]) + 2, cy=float(cell["y1"]) + 2)
    assert ns["is_rule_remnant"](frag, cell), "flush tall fragment not accepted"


def test_tall_thin_fragment_three_px_inside_edge_not_remnant(ns, result):
    """A tall thin '1' a few px inside the edge is not a rule remnant."""
    cells = result["cells"]
    by_key = {(c["group"], c["key"], c.get("row")): c for c in cells}
    cell = by_key[("B", "Q15", "a")]
    rh = cell["ref_h"]
    frag = dict(id=-1, x=int(cell["x1"]) + 3, y=int(cell["y1"]) + 2,
                w=4, h=int(0.8 * rh), area=4 * int(0.8 * rh),
                cx=float(cell["x1"]) + 5, cy=float(cell["y1"]) + 2)
    assert not ns["is_rule_remnant"](frag, cell), \
        "3px-inside tall fragment wrongly accepted"


def test_restrict_to_components_drops_outside_ink(ns):
    """Only ink inside the component bbox survives."""
    mask = np.zeros((20, 20), bool)
    mask[2:6, 2:6] = True
    mask[12:16, 12:16] = True
    comps = [dict(x=2, y=2, w=4, h=4)]
    out = ns["restrict_to_components"](mask, comps)
    assert out[2:6, 2:6].all(), "inside ink was dropped"
    assert not out[12:16, 12:16].any(), "outside ink was kept"
    assert ns["restrict_to_components"](mask, []) is mask, "empty comps must return mask"


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--image", default=REFERENCE)
    ap.add_argument("--sweep", type=int, default=10)
    args = ap.parse_args()

    ns = load_pipeline()

    failures = []
    if args.sweep:
        paths = sorted(glob.glob("Random_Images_Sample/*.jpg"))[:args.sweep]
        failures.extend(sweep(ns, paths))
        print(f"swept {len(paths)} images")

    result = analyze(ns, args.image)
    failures.extend(check_crops(result))
    for fn in (test_region_extends_into_empty_neighbour,
               test_region_stops_at_neighbour_centre,
               test_region_never_clips_inside_cell,
               test_region_vertical_reach,
               test_crop_contains_full_cell,
               test_rule_only_region_renders_none,
               test_pen_crossing_rule_survives,
               test_overflow_flags_reflect_ink,
               test_tall_thin_edge_fragment_rejected,
               test_tall_thin_fragment_flush_with_edge_rejected,
               test_tall_thin_fragment_three_px_inside_edge_not_remnant,
               test_restrict_to_components_drops_outside_ink):
        try:
            import inspect
            if len(inspect.signature(fn).parameters) == 1:
                fn(ns)
            else:
                fn(ns, result)
        except Exception as exc:
            failures.append(f"{fn.__name__}: {type(exc).__name__}: {exc}")
    for f in failures:
        print("FAIL", f)
    print(f"{len(result['views'])} cells checked, {len(failures)} failures")
    if args.check and failures:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
