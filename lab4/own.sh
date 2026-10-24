#!/bin/bash
# Le poids de la tête des propriétaires (lab4/journal.md) : depuis c130 et son optimiseur, PASSES pas
# d'apprentissage sur la fenêtre de deep7 (c099…c130, mêmes graines d'un poids à l'autre), un essai
# par poids, puis le suivi de la valeur sur les parties de validation de c126–c130 (py/track.py).
#
#   lab4/own.sh 1.5 0.5 0
set -euo pipefail
cd "$(dirname "$0")/.."
PASSES=${PASSES:-3}
PY=../../lab3/py/.venv/bin/python
for w in "$@"; do
  out=lab4/runs/own-$w
  mkdir -p "$out"
  cp lab4/nets/c130.bin "$out/p0.bin"
  cp lab3/runs/deep7/c130.state.pt "$out/p0.state.pt"
  for i in $(seq 1 "$PASSES"); do
    [ -f "$out/p$i.bin" ] && continue
    (cd lab4/py && $PY train.py --cycles ../../lab3/data/deep7 --last c130 --init "../runs/own-$w/p$((i - 1)).bin" \
      --out "../runs/own-$w/p$i.bin" --lr 1e-4 --seed "$i" --own-weight "$w" \
      --state-in "../runs/own-$w/p$((i - 1)).state.pt" --state-out "../runs/own-$w/p$i.state.pt") > "$out/p$i.log" 2>&1
    echo "poids $w, passe $i : $(grep -E '^(entraînement|validation)' "$out/p$i.log" | tr '\n' ' ')"
  done
done
D=$(for c in 126 127 128 129 130; do printf '../../lab3/data/deep7/c%s ' $c; done)
(cd lab4/py && $PY track.py --data $D --nets ../nets/c130.bin $(for w in "$@"; do printf '../runs/own-%s/p%s.bin ' "$w" "$PASSES"; done))
