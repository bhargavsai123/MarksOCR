#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3.12}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "error: $PYTHON not found. Install Python 3.12 (brew install python@3.12)" >&2
  echo "       or set PYTHON=/path/to/python3.12" >&2
  exit 1
fi

if [ ! -d .venv312 ]; then
  "$PYTHON" -m venv .venv312
fi

. .venv312/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-local.txt
python -m ipykernel install --user --name marksocr-local \
  --display-name "MarksOCR (local)"

python - <<'PY'
from importlib.metadata import version

import cv2
import numpy
import paddle
import paddleocr
from paddleocr import TextRecognition

print("Paddle:", paddle.__version__, "| CUDA compiled:", paddle.is_compiled_with_cuda())
print("PaddleOCR:", version("paddleocr"),
      "| OpenCV:", cv2.__version__, "| NumPy:", numpy.__version__)
PY

echo
echo "Environment ready."
echo "  Jupyter kernel: MarksOCR (local)"
echo "  VS Code interpreter: $(pwd)/.venv312/bin/python"
echo "  Next: open bruh_local.ipynb and run the cells in order."
