#!/bin/bash
# Efface du bucket les positions (*.pos) des cycles de la boucle à P joueurs sortis de sa fenêtre
# d'entraînement (data.outside : un cycle de marge) ; leurs parties restent (*.pos.games,
# *.pos.moves : de quoi les rejouer, et compter les positions jouées pour la taille de la fenêtre).
# Depuis mini, aucune machine allumée : une machine renverrait ses fichiers au bucket (--save).
#
#   lab3/prune.sh P
set -euo pipefail
cd "$(dirname "$0")/.."
P=${1:?le nombre de joueurs}
export AWS_PROFILE=${AWS_PROFILE:-smallai}
BUCKET=s3://smallai-lab-631245464763/lab3/data/p$P
# l'état lu en entier d'abord : avec pipefail, `grep -q` qui s'arrête au premier résultat ferait
# échouer `status` (écriture dans un tube fermé)
state=$(labtools/cloud.sh status 2>&1 || true)
if ! grep -q "aucune machine" <<< "$state"; then
  echo "une machine tourne : rien n'est effacé" >&2
  exit 1
fi
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
aws s3 sync --only-show-errors --exclude "*" --include "*.pos.games" "$BUCKET" "$tmp"
old=$(cd lab3/py && .venv/bin/python -c "from data import outside; print(' '.join(outside('$tmp')))")
echo "hors de la fenêtre : ${old:-aucun}"
for c in $old; do
  if grep -q '\.pos$' <<< "$(aws s3 ls "$BUCKET/$c/")"; then
    aws s3 rm --recursive --only-show-errors --exclude "*" --include "*.pos" "$BUCKET/$c"
    echo "$c : positions effacées"
  fi
done
echo "gardés : $(aws s3 ls --recursive --summarize "$BUCKET/" | tail -1)"
