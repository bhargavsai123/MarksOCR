#!/usr/bin/env python3
"""Regenerate per-cell crop montages into crop_debug/ for visual review."""
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

NOTEBOOK = "Mark Sheet OCR Pilot Fixes v6.ipynb"
REFERENCE = ("Random_Images_Sample/"
             "BE_B.TECH VIII Sem (main) R22 -APRIL 2026 - Copy__22CAE11__751676.jpg")
OUTDIR = "crop_debug"

with open(NOTEBOOK) as f:
    nb = json.load(f)
ns = {}
exec(compile("".join(nb["cells"][2]["source"]), "<pipeline>", "exec"), ns)

res = ns["analyze_sheet"](REFERENCE)
views, arr = res["views"], res["arr"]
os.makedirs(OUTDIR, exist_ok=True)


def selected_views(entry):
    v = entry.get("views") or {}
    large = bool(v.get("large"))
    if large:
        if v.get("overflow_x"):
            order = [("color", entry.get("color")), ("clip", v.get("clip")),
                     ("expanded", v.get("expanded")), ("context", v.get("context")),
                     ("dark", v.get("dark")), ("binary", v.get("binary")), ("gray", v.get("gray"))]
        else:
            order = [("color", entry.get("color")), ("binary", v.get("binary")),
                     ("gray", v.get("gray")), ("expanded", v.get("expanded")),
                     ("context", v.get("context")), ("dark", v.get("dark")), ("clip", v.get("clip"))]
    elif v.get("overflow_x"):
        order = [("color", entry.get("color")), ("clip", v.get("clip")), ("context", v.get("context"))]
    elif v.get("overflow_y"):
        order = [("color", entry.get("color")), ("context", v.get("context")),
                 ("dark", v.get("dark")), ("clip", v.get("clip"))]
    else:
        order = [("color", entry.get("color")), ("binary", v.get("binary")),
                 ("gray", v.get("gray")), ("context", v.get("context")),
                 ("dark", v.get("dark")), ("clip", v.get("clip"))]
    order = [(n, im) for n, im in order if im is not None][:7 if large else 3]
    return order, large


try:
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 12)
except Exception:
    font = ImageFont.load_default()


def montage(items, title):
    pad, labh = 6, 16
    ims = [im for _, im in items]
    if not ims:
        return None
    H = max(im.shape[0] for im in ims) + labh + 2 * pad
    W = sum(im.shape[1] + pad for im in ims) + pad
    canvas = Image.new("RGB", (W, H), (245, 245, 245))
    d = ImageDraw.Draw(canvas)
    x = pad
    for name, im in items:
        h, w = im.shape[:2]
        canvas.paste(Image.fromarray(im), (x, pad + labh))
        d.text((x, 2), name, fill=(0, 0, 0), font=font)
        d.rectangle([x - 1, pad + labh - 1, x + w, pad + labh + h], outline=(180, 180, 180))
        x += w + pad
    d.text((pad, pad + labh + max(im.shape[0] for im in ims) + 2), title, fill=(80, 80, 80), font=font)
    return canvas


rows = []
for k, entry in views.items():
    if not entry.get("views"):
        continue
    items, large = selected_views(entry)
    if not items:
        continue
    c = entry["cell"]
    title = (f"{k} | cell=({c['x1']},{c['y1']},{c['x2']},{c['y2']}) large={large} "
             f"ovx={entry['views'].get('overflow_x')} ovy={entry['views'].get('overflow_y')}")
    m = montage(items, title)
    if m is None:
        continue
    m.save(os.path.join(OUTDIR, f"{k}.png"))
    rows.append((k, m))

if rows:
    W = max(m.width for _, m in rows)
    H = sum(m.height + 8 for _, m in rows)
    ov = Image.new("RGB", (W, H), (255, 255, 255))
    y = 0
    for _, m in rows:
        ov.paste(m, (0, y))
        y += m.height + 8
    ov.save(os.path.join(OUTDIR, "_overview_all_cells.png"))

page = Image.fromarray(arr.copy())
d = ImageDraw.Draw(page)
for k, entry in views.items():
    c = entry["cell"]
    d.rectangle([c["x1"], c["y1"], c["x2"], c["y2"]], outline=(0, 120, 255), width=1)
    d.text((c["x1"] + 1, c["y1"] + 1), k, fill=(0, 80, 200), font=font)
    box = (entry.get("views") or {}).get("box")
    if box:
        d.rectangle(list(box), outline=(220, 0, 0), width=1)
page.save(os.path.join(OUTDIR, "_page_boxes.png"))
print(f"wrote {len(rows)} montages + overview + page boxes to {OUTDIR}/")
