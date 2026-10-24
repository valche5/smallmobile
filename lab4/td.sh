#!/bin/bash
# Les cibles de valeur à court terme (lab4/journal.md) : depuis c130, PASSES pas d'apprentissage sur
# la fenêtre de deep7 (c099…c130, mêmes graines d'une variante à l'autre), un essai par part λ de la
# valeur de la recherche au tour suivant dans la cible (`train.py --td`, 0 : le témoin), puis la
# valeur de chacun sur les positions du plafond (py/ceiling.py) et sur les positions fixes
# (py/track.py).
#
#   lab4/td.sh 0 0.5
set -euo pipefail
cd "$(dirname "$0")/.."
PASSES=${PASSES:-3}
PY=../../lab3/py/.venv/bin/python
for td in "$@"; do
  out=lab4/runs/td-$td
  mkdir -p "$out"
  cp lab4/nets/c130.bin "$out/p0.bin"
  for i in $(seq 1 "$PASSES"); do
    [ -f "$out/p$i.bin" ] && continue
    (cd lab4/py && $PY train.py --cycles ../../lab3/data/deep7 --last c130 --init "../runs/td-$td/p$((i - 1)).bin" \
      --out "../runs/td-$td/p$i.bin" --lr 1.5e-4 --seed "$i" --td "$td") > "$out/p$i.log" 2>&1
    echo "td $td, passe $i : $(grep -E '^(entraînement|validation)' "$out/p$i.log" | tr '\n' ' ')"
  done
done
nets=$(for td in "$@"; do printf '../runs/td-%s/p%s.bin ' "$td" "$PASSES"; done)
(cd lab4/py && $PY ceiling.py ../runs/ceiling/c130.tsv --nets ../nets/c130.bin $nets)
D=$(for c in 126 127 128 129 130; do printf '../../lab3/data/deep7/c%s ' $c; done)
(cd lab4/py && $PY track.py --data $D --nets ../nets/c130.bin $nets)
