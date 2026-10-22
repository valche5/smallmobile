#!/bin/bash
# La boucle d'auto-apprentissage de lab3 (lab3/journal.md), à la manière de KataGo, pour un réseau
# spécialisé dans un nombre de joueurs P : ses parties d'auto-jeu et ses arènes se jouent toutes
# à P joueurs. Pas k :
#   - en même temps : l'entraînement du réseau k depuis le réseau k−1, sur une fenêtre des derniers
#     cycles jusqu'au cycle k (lab3/py/train.py, TRAIN cœurs), et les parties du cycle k+1 jouées
#     par le réseau k−1 (PLAY cœurs) : un cycle de retard, comme les joueurs d'AlphaZero et de
#     KataGo pendant que le réseau apprend ;
#   - puis les arènes du réseau k face à Difficile et Normale, à P joueurs (mêmes donnes à chaque
#     pas).
# Le réseau de départ c000 est le généraliste de lab2 (lab3/nets/start.bin, son c060) ; les parties
# des cycles 1 et 2 sont les siennes.
#
#   lab3/loop.sh P N      fait les pas 1 à N du réseau à P joueurs ; une étape déjà faite est
#                         sautée (reprise)
#
# Parties : lab3/data/pP/cNNN ; réseaux : lab3/runs/pP/cNNN.bin ; arènes
# lab3/runs/pP/arena-cNNN-{hard,normal}.tsv ; journal lab3/runs/pP/log.txt (durée de chaque
# étape). Sur AWS : --save lab3/data/pP,lab3/runs/pP. Réglages (essais sur mini) : GAMES, DEALS,
# PLAY, TRAIN.
set -euo pipefail
cd "$(dirname "$0")/.."
P=${1:?le nombre de joueurs}
LAST=${2:?le nombre de pas}
GAMES=${GAMES:-25000}
DEALS=${DEALS:-460}
# cœurs des parties et de l'entraînement quand ils tournent ensemble, sur c7a.16xlarge : avec le
# moteur de lab3 (1,6 fois plus rapide que celui de lab2), les parties d'un réseau 128x4 n'ont plus
# besoin de 56 cœurs, et l'entraînement en profite (il ne gagne presque plus rien au-delà de 16)
PLAY=${PLAY:-48}
TRAIN=${TRAIN:-16}
DATA=lab3/data/p$P
OUT=lab3/runs/p$P
TMP=lab3/data/tmp
LAB=./target/release/lab3
mkdir -p "$DATA" "$OUT" "$TMP"
[ -f "$OUT/c000.bin" ] || cp lab3/nets/start.bin "$OUT/c000.bin"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/log.txt"; }
name() { printf c%03d "$1"; }

games() { # cycle, réseau, cœurs : écrites à part, puis déplacées. Un cycle fini a ses fichiers
  # .pos.games (--save peut recréer un dossier vide ; les .pos des cycles sortis de la fenêtre sont
  # effacés du bucket par lab3/prune.sh, les .games restent)
  local c=$1 net=$2 threads=$3 start
  compgen -G "$DATA/$c/*.pos.games" > /dev/null && return 0
  start=$(date +%s)
  rm -rf "${TMP:?}/p$P-$c"
  if ! $LAB selfplay --agent "az:net=$OUT/$net.bin,n=200" --players "$P" --games "$GAMES" \
    --seed $((30000000 + 10#${c#c})) --threads "$threads" --out "$TMP/p$P-$c" > "$TMP/p$P-$c.txt" \
    2> "$TMP/p$P-$c.err"; then
    tail -5 "$TMP/p$P-$c.err"
    return 1
  fi
  rm -rf "${DATA:?}/$c"
  mv "$TMP/p$P-$c" "$DATA/$c"
  log "$c parties ($net) : $(($(date +%s) - start)) s, $(cat "$TMP/p$P-$c.txt")"
  rm -f "$TMP/p$P-$c.txt" "$TMP/p$P-$c.err"
}

train() { # réseau, réseau de départ
  local c=$1 prev=$2 start
  [ -f "$OUT/$c.bin" ] && return 0
  start=$(date +%s)
  if ! (cd lab3/py && .venv/bin/python train.py --cycles "../data/p$P" --last "$c" --init "../runs/p$P/$prev.bin" \
    --out "../runs/p$P/$c.tmp.bin" --threads "$TRAIN") > "$OUT/$c-train.log" 2>&1; then
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
    $LAB arena --hero "az:net=$OUT/$c.bin,n=200" --field "$field" --players "$P" --deals "$DEALS" --seed "$seed" \
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
