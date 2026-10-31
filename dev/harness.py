"""Load the shipped v7 notebook for local testing.

The notebook is the deliverable, so the tests must exercise the same source
that ships.  ``load()`` executes the layout + glyph-front-end half of cell 2;
``load_full()`` executes the whole cell, including the bubble and analysis
sections.  There is no separate copy of the front-end to drift out of sync.
"""
import json
import pathlib
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Mark Sheet OCR v7 - glyph pipeline.ipynb"

HEADER = """
import os, re, json, time, math, glob, statistics
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional, Sequence
from itertools import product, combinations
from fractions import Fraction
import cv2
import numpy as np
from PIL import Image, ImageOps
PIPELINE_VERSION = "test"
"""


def notebook_source():
    nb = json.loads(NOTEBOOK.read_text())
    for cell in nb["cells"]:
        if cell["cell_type"] == "code" and "".join(cell["source"]).startswith(
                "# 2. Dynamic mark-sheet pipeline"):
            return "".join(cell["source"])
    raise RuntimeError("pipeline cell not found")


def load():
    src = notebook_source()
    start = src.index("# 2. Dynamic mark-sheet pipeline")
    end = src.index("# ============================ BUBBLE")
    code = HEADER + src[start:end]
    mod = types.ModuleType("glyph_frontend")
    mod.__file__ = str(NOTEBOOK)
    exec(compile(code, str(NOTEBOOK), "exec"), mod.__dict__)
    return mod


def load_full():
    """Exec the whole pipeline cell, including the bubble/analysis sections."""
    src = notebook_source()
    mod = types.ModuleType("mark_sheet_pipeline")
    mod.__file__ = str(NOTEBOOK)
    exec(compile(HEADER + src, str(NOTEBOOK), "exec"), mod.__dict__)
    return mod
