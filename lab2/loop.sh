#!/bin/bash
# La boucle d'auto-apprentissage de lab2 (lab2/journal.md), à la manière de KataGo. Pas k :
#   - en même temps : l'entraînement du réseau k depuis le réseau k−1, sur une fenêtre des derniers
#     cycles jusqu'au cycle k (lab2/py/train.py, TRAIN cœurs), et les parties du cycle k+1 jouées
#     par le réseau k−1 (PLAY cœurs) : un cycle de retard, comme les joueurs d'AlphaZero et de
#     KataGo pendant que le réseau apprend ;
#   - puis les arènes du réseau k face à Difficile et Normale (mêmes donnes à chaque pas).
# Les parties du cycle 1 (et du cycle 2) sont celles de c000, gen0 : le hasard.
#
#   lab2/loop.sh N        fait les pas 1 à N ; une étape déjà faite est sautée (reprise)
#
# Parties : lab2/data/loop/cNNN ; réseaux : lab2/runs/loop/cNNN.bin ; arènes
# lab2/runs/loop/arena-cNNN-{hard,normal}.tsv ; journal lab2/runs/loop/log.txt (durée de chaque
# étape). Sur AWS : --save lab2/data/loop,lab2/runs/loop. Réglages (essais sur mini) : GAMES,
# DEALS, PLAY, TRAIN.
set -euo pipefail
cd "$(dirname "$0")/.."
LAST=${1:?le nombre de pas}
GAMES=${GAMES:-25000}
DEALS=${DEALS:-460}
# cœurs des parties et de l'entraînement quand ils tournent ensemble (mesures du 21 octobre sur
# c7a.16xlarge : l'entraînement ne gagne presque plus rien au-delà de 16 cœurs)
PLAY=${PLAY:-48}
TRAIN=${TRAIN:-16}
DATA=lab2/data/loop
OUT=lab2/runs/loop
TMP=lab2/data/tmp
LAB=./target/release/lab2
mkdir -p "$DATA" "$OUT" "$TMP"
[ -f "$OUT/c000.bin" ] || cp lab2/nets/gen0.bin "$OUT/c000.bin"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/log.txt"; }
name() { printf c%03d "$1"; }

games() { # cycle, réseau, cœurs : écrites à part, puis déplacées. Un cycle fini a ses fichiers
  # .pos.games (--save peut recréer un dossier vide ; les .pos des cycles sortis de la fenêtre sont
  # effacés du bucket par lab2/prune.sh, les .games restent)
  local c=$1 net=$2 threads=$3 start
  compgen -G "$DATA/$c/*.pos.games" > /dev/null && return 0
  start=$(date +%s)
  rm -rf "${TMP:?}/$c"
  if ! $LAB selfplay --agent "az:net=$OUT/$net.bin,n=200" --games "$GAMES" --seed $((30000000 + 10#${c#c})) \
    --threads "$threads" --out "$TMP/$c" > "$TMP/$c.txt" 2> "$TMP/$c.err"; then
    tail -5 "$TMP/$c.err"
    return 1
  fi
  rm -rf "${DATA:?}/$c"
  mv "$TMP/$c" "$DATA/$c"
  log "$c parties ($net) : $(($(date +%s) - start)) s, $(cat "$TMP/$c.txt")"
  rm -f "$TMP/$c.txt" "$TMP/$c.err"
}

train() { # réseau, réseau de départ
  local c=$1 prev=$2 start
  [ -f "$OUT/$c.bin" ] && return 0
  start=$(date +%s)
  if ! (cd lab2/py && .venv/bin/python train.py --cycles ../data/loop --last "$c" --init "../runs/loop/$prev.bin" \
    --out "../runs/loop/$c.tmp.bin" --threads "$TRAIN") > "$OUT/$c-train.log" 2>&1; then
    tail -20 "$OUT/$c-train.log"
    return 1
  fi
  mv "$OUT/$c.tmp.bin" "$OUT/$c.bin"
  log "$c entraînement : $(($(date +%s) - start)) s"
  sed 's/^/    /' "$OUT/$c-train.log" | grep -v "écrit" | tee -a "$OUT/log.txt"
}

arenas() { # réseau
  local c=$1 field seed start
  for field in hard normal; do
    [ -f "$OUT/arena-$c-$field.tsv" ] && continue
    start=$(date +%s)
    seed=1002
    [ "$field" = normal ] && seed=1003
    $LAB arena --hero "az:net=$OUT/$c.bin,n=200" --field "$field" --deals "$DEALS" --seed "$seed" \
      --threads $((PLAY + TRAIN)) --out "$OUT/arena-$c-$field.tmp" > "$OUT/arena-$c-$field.txt"
    mv "$OUT/arena-$c-$field.tmp" "$OUT/arena-$c-$field.tsv"
    log "$c face à $field : $(($(date +%s) - start)) s, \
$(python3 labtools/score.py "$OUT/arena-$c-$field.tsv" --role 0 | grep '^score')"
  done
}

# en cas d'arrêt, les étapes lancées en même temps s'arrêtent aussi (par numéro de processus)
pids=()
trap 'for p in "${pids[@]}"; do pkill -P "$p" 2> /dev/null || true; kill "$p" 2> /dev/null || true; done' EXIT

for k in $(seq 1 "$LAST"); do
  c=$(name "$k")
  prev=$(name $((k - 1)))
  # les parties du cycle k (déjà jouées au pas précédent, sauf au premier pas ou à une reprise)
  games "$c" "$(name $((k > 2 ? k - 2 : 0)))" $((PLAY + TRAIN))
  # le réseau k et, en même temps, les parties du cycle k+1 par le réseau k−1
  pids=()
  train "$c" "$prev" &
  pids+=($!)
  if [ "$k" -lt "$LAST" ]; then
    games "$(name $((k + 1)))" "$prev" "$PLAY" &
    pids+=($!)
  fi
  for p in "${pids[@]}"; do
    wait "$p"
  done
  arenas "$c"
done
