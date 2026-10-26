#!/bin/bash
# Environnement Python de lab8 : PyTorch pour ROCm, avec les seuls pilotes de la RX 9070 XT (gfx1201) ;
# demande uv. Même recette que lab3/py/setup.sh.
set -euo pipefail
cd "$(dirname "$0")"
uv venv --clear --python 3.13 .venv
override=$(mktemp)
echo "rocm[libraries,device-gfx1201]==7.14.1" > "$override"
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/rocm7.14 --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match --override "$override" \
  "torch==2.14.1+rocm7.14" numpy
rm -f "$override"
