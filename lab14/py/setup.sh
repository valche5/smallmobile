#!/bin/bash
# L'analyse n'a aucune dépendance Python externe ; l'apprentissage (train.py, loop.sh) demande PyTorch
# et NumPy : sur les machines vast.ai, `pip install torch==2.14.1 numpy` (lab14/journal.md).
# Interface conservée pour labtools/cloud.sh (cpu/cuda).
set -euo pipefail
cd "$(dirname "$0")"
timeout 60 python3 -m venv .venv
echo "lab14 : analyse prête (bibliothèque standard Python)."
