#!/bin/bash
# Le diagnostic et son analyse n'ont aucune dépendance Python externe.
# Interface conservée pour labtools/cloud.sh (cpu/cuda) ; pas d'entraînement dans lab6 à ce stade.
set -euo pipefail
cd "$(dirname "$0")"
timeout 60 python3 -m venv .venv
echo "lab6 : analyse prête (bibliothèque standard Python)."
