"""Regression tests for the ring-derived OMR bubble grid in the v7 notebook.

The notebook is the deliverable, so these tests exec the shipped pipeline cell
and call its own ``_ring_grid`` / ``read_bubble_marks``.

Run with:  python3 dev/test_bubble_grid.py
"""
import os
import sys

import numpy as np
from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import harness  # noqa: E402

P = harness.load_full()
SAMPLE = os.path.join(ROOT, 'Random_Images_Sample')


def load(name):
    arr = np.array(ImageOps.exif_transpose(
        Image.open(os.path.join(SAMPLE, name))).convert('RGB'))
    det = P.detect_layout(arr)
    return arr, det


def read_bubbles(name):
    arr, det = load(name)
    return P.read_bubble_marks(det['arr'], det)


def test_ii_grid_is_ten_uniform_rows():
    """The II forms use a taller panel than the B-relative nominal grid."""
    arr, det = load('II__22MTC04__840060_1035026.bmp')
    H, W = arr.shape[:2]
    g = P._ring_grid(det['B'], det['mask'], W, H)
    assert g is not None
    cols, rows, r = g
    assert len(cols) == 4 and len(rows) == 10
    d = np.diff(rows)
    assert d.min() > 0 and d.max() <= 1.35 * d.min()
    assert 0.5 <= (2 * r) / float(np.median(d)) <= 1.15


def test_ii_total_marks_two_digits():
    """Previously only the tens digit was read, giving Total Marks 5."""
    total, _, _ = read_bubbles('II__22MTC04__840060_1035026.bmp')
    assert total == 52, total


def test_ii_total_marks_second_sheet():
    total, _, _ = read_bubbles('II__22CSC03__840060_1041117.bmp')
    assert total == 54, total


def test_ii_total_marks_third_sheet():
    total, _, _ = read_bubbles('II__22EGC01__840060_1036013.bmp')
    assert total == 45, total


def test_ii_units_digit_preserved():
    """II__22PHY07 has tens=5 and units=1; the old grid dropped the units."""
    total, _, diag = read_bubbles('II__22PHY07__840060_1041602.bmp')
    assert diag['digits'][0] == 5 and diag['digits'][1] == 1, diag['digits']
    assert total == 51, (total, diag['digits'])


def test_iv_layout_unchanged():
    """The IV layout already worked; it must still read 38 / 13."""
    total, si, _ = read_bubbles('IV__22AMC08__840030_616827.tif')
    assert (total, si) == (38, 13), (total, si)


def test_jpeg_layout_falls_back_to_nominal():
    """BE sheets have thin rings; the ring fit must decline, not misread."""
    arr, det = load('BE_B.TECH VIII Sem (main) R22 -APRIL 2026 '
                    '- Copy__22ADE31__751485.jpg')
    H, W = arr.shape[:2]
    assert P._ring_grid(det['B'], det['mask'], W, H) is None
    total, si, _ = P.read_bubble_marks(det['arr'], det)
    assert total is not None and 0 <= total <= 70, total


def test_all_ii_sheets_fit():
    names = [n for n in os.listdir(SAMPLE) if n.startswith('II__')]
    assert len(names) >= 30
    for name in sorted(names):
        arr, det = load(name)
        H, W = arr.shape[:2]
        g = P._ring_grid(det['B'], det['mask'], W, H)
        assert g is not None, name
        cols, rows, r = g
        assert len(cols) == 4 and len(rows) == 10, name
        d = np.diff(rows)
        assert d.min() > 0 and d.max() <= 1.35 * d.min(), name


def test_every_ii_marks_line_is_two_digits():
    """No II sheet may come back with a lone digit (the old failure mode)."""
    names = sorted(n for n in os.listdir(SAMPLE) if n.startswith('II__'))
    singles = []
    for name in names:
        total, _, diag = read_bubbles(name)
        if total is not None and total < 10:
            singles.append((name, total, diag['digits']))
    # a genuine 0-9 total is possible; allow at most a couple
    assert len(singles) <= 3, singles


if __name__ == '__main__':
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith('test_') and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print('ok   %s' % name)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print('FAIL %s: %s' % (name, exc))
    print('\n%d/%d passed' % (len(tests) - failed, len(tests)))
    sys.exit(1 if failed else 0)
