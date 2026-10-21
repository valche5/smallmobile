#!/bin/bash
# La boucle d'auto-apprentissage de lab2 (lab2/journal.md), à la manière de KataGo : à chaque cycle,
# des parties du dernier réseau (lab2/data/loop/cNNN), puis un entraînement depuis lui sur une
# fenêtre des derniers cycles (lab2/py/train.py), puis des arènes face à Difficile et Normale (mêmes
# donnes à chaque cycle).
#
#   lab2/loop.sh N        fait les cycles 1 à N ; une étape déjà faite est sautée (reprise)
#
# Réseaux : lab2/runs/loop/cNNN.bin (c000 : gen0, réseau neuf) ; arènes
# lab2/runs/loop/arena-cNNN-{hard,normal}.tsv ; journal lab2/runs/loop/log.txt (durée de chaque
# étape). Sur AWS : --save lab2/data/loop,lab2/runs/loop. Réglages (essais) : GAMES, DEALS, THREADS.
set -euo pipefail
cd "$(dirname "$0")/.."
LAST=${1:?le nombre de cycles}
GAMES=${GAMES:-25000}
DEALS=${DEALS:-920}
THREADS=${THREADS:+--threads $THREADS}
DATA=lab2/data/loop
OUT=lab2/runs/loop
TMP=lab2/data/tmp
LAB=./target/release/lab2
mkdir -p "$DATA" "$OUT" "$TMP"
[ -f "$OUT/c000.bin" ] || cp lab2/nets/gen0.bin "$OUT/c000.bin"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/log.txt"; }

for k in $(seq 1 "$LAST"); do
  c=$(printf c%03d "$k")
  prev=$(printf c%03d $((k - 1)))
  # des parties : écrites à part, puis déplacées (un cycle fini a ses fichiers ; --save peut
  # recréer un dossier vide)
  if ! compgen -G "$DATA/$c/*.pos" > /dev/null; then
    start=$(date +%s)
    rm -rf "${TMP:?}/$c"
    # shellcheck disable=SC2086
    if ! $LAB selfplay --agent "az:net=$OUT/$prev.bin,n=200" --games "$GAMES" --seed $((30000000 + k)) \
      $THREADS --out "$TMP/$c" > "$TMP/$c.txt" 2> "$TMP/$c.err"; then
      tail -5 "$TMP/$c.err"
      exit 1
    fi
    rm -rf "${DATA:?}/$c"
    mv "$TMP/$c" "$DATA/$c"
    log "$c parties : $(($(date +%s) - start)) s, $(cat "$TMP/$c.txt")"
    rm -f "$TMP/$c.txt" "$TMP/$c.err"
  fi
  if [ ! -f "$OUT/$c.bin" ]; then
    start=$(date +%s)
    # shellcheck disable=SC2086
    if ! (cd lab2/py && .venv/bin/python train.py --cycles ../data/loop --init "../runs/loop/$prev.bin" \
      --out "../runs/loop/$c.tmp.bin" $THREADS) > "$OUT/$c-train.log" 2>&1; then
      tail -20 "$OUT/$c-train.log"
      exit 1
    fi
    mv "$OUT/$c.tmp.bin" "$OUT/$c.bin"
    log "$c entraînement : $(($(date +%s) - start)) s"
    sed 's/^/    /' "$OUT/$c-train.log" | grep -v "écrit" | tee -a "$OUT/log.txt"
  fi
  for field in hard normal; do
    name=arena-$c-$field
    [ -f "$OUT/$name.tsv" ] && continue
    start=$(date +%s)
    seed=1002
    [ "$field" = normal ] && seed=1003
    # shellcheck disable=SC2086
    $LAB arena --hero "az:net=$OUT/$c.bin,n=200" --field "$field" --deals "$DEALS" --seed "$seed" $THREADS \
      --out "$OUT/$name.tmp" > "$OUT/$name.txt"
    mv "$OUT/$name.tmp" "$OUT/$name.tsv"
    log "$c face à $field : $(($(date +%s) - start)) s, $(python3 labtools/score.py "$OUT/$name.tsv" --role 0 | grep '^score')"
  done
done
