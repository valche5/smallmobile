#!/bin/bash
# Environnement Python de l'entraînement.
#   ./setup.sh       PyTorch pour ROCm, avec les seuls pilotes de la RX 9070 (gfx1201) ; demande uv.
#   ./setup.sh cpu   PyTorch sur processeur (machine sans GPU, mini par exemple) ; venv et pip suffisent.
set -e
cd "$(dirname "$0")"
if [ "${1:-}" = cpu ]; then
  python3 -m venv --clear .venv
  .venv/bin/pip install --quiet --upgrade pip
  .venv/bin/pip install --quiet --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple torch numpy
  exit
fi
uv venv --clear --python 3.13 .venv
echo "rocm[libraries,device-gfx1201]==7.14.1" > /tmp/smallmobile-rocm-override.txt
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/rocm7.14 --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match --override /tmp/smallmobile-rocm-override.txt \
  "torch==2.14.1+rocm7.14" numpy
