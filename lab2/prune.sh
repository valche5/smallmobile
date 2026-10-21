#!/bin/bash
# Efface du bucket les positions (*.pos) des cycles de la boucle sortis de sa fenêtre d'entraînement
# (data.outside : un cycle de marge) ; leurs parties restent (*.pos.games, *.pos.moves : de quoi
# les rejouer, et compter les positions jouées pour la taille de la fenêtre). Depuis mini, aucune
# machine allumée : une machine renverrait ses fichiers au bucket (--save).
set -euo pipefail
cd "$(dirname "$0")/.."
export AWS_PROFILE=${AWS_PROFILE:-smallai}
BUCKET=s3://smallai-lab-631245464763/lab2/data/loop
if ! labtools/cloud.sh status 2>&1 | grep -q "aucune machine"; then
  echo "une machine tourne : rien n'est effacé" >&2
  exit 1
fi
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
aws s3 sync --only-show-errors --exclude "*" --include "*.pos.games" "$BUCKET" "$tmp"
old=$(cd lab2/py && .venv/bin/python -c "from data import outside; print(' '.join(outside('$tmp')))")
for c in $old; do
  if aws s3 ls "$BUCKET/$c/" | grep -q '\.pos$'; then
    aws s3 rm --recursive --only-show-errors --exclude "*" --include "*.pos" "$BUCKET/$c"
    echo "$c : positions effacées"
  fi
done
echo "gardés : $(aws s3 ls --recursive --summarize "$BUCKET/" | tail -1)"
