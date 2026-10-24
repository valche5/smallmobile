#!/bin/bash
# Environnement Python de l'entraînement.
#   ./setup.sh       PyTorch pour ROCm, avec les seuls pilotes de la RX 9070 (gfx1201) ; demande uv.
#   ./setup.sh cpu   PyTorch sur processeur (machine de calcul sans GPU) ; venv et pip suffisent.
#   ./setup.sh cuda  PyTorch pour CUDA (machine à GPU NVIDIA, labtools/cloud.sh) ; venv et pip suffisent.
set -e
cd "$(dirname "$0")"
if [ "${1:-}" = cpu ] || [ "${1:-}" = cuda ]; then
  python3 -m venv --clear .venv
  .venv/bin/pip install --quiet --upgrade pip
  if [ "$1" = cpu ]; then
    .venv/bin/pip install --quiet --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple torch numpy
  else
    # les roues de PyTorch sur PyPI sont celles de CUDA
    .venv/bin/pip install --quiet torch numpy
  fi
  exit
fi
uv venv --clear --python 3.13 .venv
echo "rocm[libraries,device-gfx1201]==7.14.1" > /tmp/smallmobile-rocm-override.txt
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/rocm7.14 --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match --override /tmp/smallmobile-rocm-override.txt \
  "torch==2.14.1+rocm7.14" numpy
