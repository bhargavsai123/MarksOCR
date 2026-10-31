#!/usr/bin/env python3
"""Run the glyph front-end on Random_Images_Sample and dump diagnostics.

No ground truth required. Outputs:
  - /tmp/glyph_diagnostics.csv: one row per cell with glyph info
  - /tmp/glyph_summary.txt: per-sheet summary
  - /tmp/contact_<i>.png: crop contact sheets for first N sheets
"""
import os
import sys
import csv
import pathlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness

G = harness.load()
SAMPLES = harness.ROOT / "Random_Images_Sample"

names = sorted([n for n in os.listdir(SAMPLES)
                if n.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))])

print(f"Found {len(names)} images")

# Build a contact sheet for the first 5 sheets
from PIL import Image
import cv2
import numpy as np

os.makedirs("/tmp/glyph_crops", exist_ok=True)

with open("/tmp/glyph_diagnostics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["sheet", "cell", "y1", "y2", "x1", "x2", "ref_h",
                     "has_ink", "ambiguous", "ncomp", "n_glyphs", "glyphs",
                     "large", "spans_rows", "overflow_top", "overflow_bottom", "overflow_x",
                     "shared_ids", "dash_only", "faint_ink", "ink_px"])

    for idx, name in enumerate(names):
        path = SAMPLES / name
        try:
            arr, det, views = G.extract_sheet(str(path))
        except Exception as exc:
            print(f"  {name}: ERROR {exc}")
            writer.writerow([name, "ERROR", "", "", "", "", "",
                             "", "", "", "", "", "", "", "", "", "", "", "", "", "", str(exc)])
            continue

        n_cells = len(views)
        n_ink = sum(1 for e in views.values() if e.get("has_ink"))
        n_amb = sum(1 for e in views.values() if e.get("ambiguous"))
        n_glyphs = sum((e.get("views") or {}).get("n_glyphs", 0) for e in views.values())
        tall = sum(1 for e in views.values()
                   for g in e.get("cell", {}).get("comps", [])
                   if g.get("h", 0) > 1.25 * e["cell"].get("ref_h", 1))

        print(f"  [{idx+1}/{len(names)}] {name[:50]:50s}  cells={n_cells} ink={n_ink} amb={n_amb} glyphs={n_glyphs} tall={tall}")

        for key, entry in views.items():
            cell = entry.get("cell") or {}
            views_d = entry.get("views") or {}
            glyphs = cell.get("comps", [])
            writer.writerow([
                name, key,
                cell.get("y1", ""), cell.get("y2", ""),
                cell.get("x1", ""), cell.get("x2", ""),
                cell.get("ref_h", ""),
                entry.get("has_ink", False),
                entry.get("ambiguous", False),
                entry.get("ncomp", 0),
                views_d.get("n_glyphs", 0),
                ";".join(f"{g['x']},{g['y']},{g['w']},{g['h']}" for g in glyphs),
                views_d.get("large", False),
                views_d.get("spans_rows", False),
                views_d.get("overflow_top", 0),
                views_d.get("overflow_bottom", 0),
                views_d.get("overflow_x", 0),
                ";".join(map(str, entry.get("shared_ids") or [])),
                entry.get("dash_only", False),
                entry.get("faint_ink", False),
                entry.get("ink_px", 0),
            ])

        # Contact sheet for first 5
        if idx < 5:
            tiles = [(k, e["views"]["color"]) for k, e in views.items()
                     if e.get("has_ink") and e.get("views", {}).get("color") is not None]
            cols = 8
            cw, ch = 150, 110
            rows = (len(tiles) + cols - 1) // cols
            sheet_img = np.full((rows * ch, cols * cw, 3), 255, np.uint8)
            for i, (k, im) in enumerate(tiles):
                r, c = divmod(i, cols)
                h, w = im.shape[:2]
                sc = min((cw - 8) / w, (ch - 24) / h)
                im2 = cv2.resize(im, (max(1, int(w * sc)), max(1, int(h * sc))),
                                 interpolation=cv2.INTER_AREA)
                y = r * ch + 18
                x = c * cw + 4
                sheet_img[y:y + im2.shape[0], x:x + im2.shape[1]] = im2
                cv2.putText(sheet_img, k, (c * cw + 4, r * ch + 13),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 180), 1, cv2.LINE_AA)
            Image.fromarray(sheet_img).save(f"/tmp/glyph_crops/contact_{idx}_{name[:30]}.png")

print("\nDone. Diagnostics in /tmp/glyph_diagnostics.csv")
print("Contact sheets in /tmp/glyph_crops/")