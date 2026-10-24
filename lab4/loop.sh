#!/bin/bash
# La boucle d'auto-apprentissage de lab4, reprise de celle de lab3 (lab3/loop.sh), sur GPU et en
# étapes successives seulement, depuis la fenêtre et le réseau d'un autre essai. Pas k :
#   1. les parties du cycle k, jouées par le réseau k−1 (`lab4 selfplay`, recherches complètes à
#      SIMS simulations, objectif `win=WIN`) ;
#   2. l'apprentissage du réseau k depuis k−1 sur la fenêtre des derniers cycles (lab4/py/train.py,
#      en continu : optimiseur et moyenne des poids gardés d'un pas à l'autre, taux LR) ;
#   3. le duel du réseau k (objectif `win=WIN`) contre REF (objectif `win=REF_WIN`, 0,5 par défaut :
#      le réglage d'avant), n = 200, mêmes donnes à chaque pas (graine 1004), arrêté par le test
#      séquentiel : le gain de l'ensemble, objectif et apprentissage.
#
#   lab4/loop.sh N        les pas START à N de l'essai RUN
#
# Avant le premier pas : les cycles de la fenêtre liés dans lab4/data/RUN (au moins jusqu'au cycle
# START−1, et tous ceux d'avant pour la taille de la fenêtre), le réseau de départ et son état en
# lab4/runs/RUN/c(START−1).bin et .state.pt, la référence en lab4/runs/RUN/REF.bin.
#
# Parties : lab4/data/RUN/cNNN ; réseaux : lab4/runs/RUN/cNNN.bin ; journal lab4/runs/RUN/log.txt
# (durée de chaque étape). Python : celui de lab3 (lab3/py/.venv, mêmes dépendances).
set -euo pipefail
cd "$(dirname "$0")/.."
LAST=${1:?le dernier pas}
RUN=${RUN:?le nom de l’essai}
START=${START:?le premier pas}
P=${P:-2,3,4,5}
GAMES=${GAMES:-25000}
SIMS=${SIMS:-400}
# poids de la victoire dans la valeur que maximise la recherche (lab4/brain/src/search.rs,
# `Config::win_weight`) : selfplay et duels
WIN=${WIN:-0.5}
LR=${LR:-1e-4}
DEALS=${DEALS:-460}
REF=${REF:?la référence des duels}
REF_WIN=${REF_WIN:-0.5}
STOP=${STOP:-3}
THREADS=${THREADS:-$(nproc)}
SEED_BASE=${SEED_BASE:-30000060}
export LAB4_CONCURRENT=${LAB4_CONCURRENT:-16}
DATA=lab4/data/$RUN
OUT=lab4/runs/$RUN
TMP=lab4/data/tmp
LAB=./target/release/lab4
PY=../../lab3/py/.venv/bin/python
mkdir -p "$DATA" "$OUT" "$TMP"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/log.txt"; }
name() { printf c%03d "$1"; }

games() { # cycle, réseau : écrites à part, puis déplacées ; un cycle fini a ses .pos.games
  local c=$1 net=$2 start
  compgen -G "$DATA/$c/*.pos.games" > /dev/null && return 0
  start=$(date +%s)
  rm -rf "${TMP:?}/$RUN-$c"
  if ! $LAB selfplay --agent "az:net=$OUT/$net.bin,n=$SIMS,win=$WIN" --players "$P" --games "$GAMES" \
    --seed $((SEED_BASE + 10#${c#c})) --threads "$THREADS" --device vulkan --out "$TMP/$RUN-$c" \
    > "$TMP/$RUN-$c.txt" 2> "$TMP/$RUN-$c.err"; then
    tail -5 "$TMP/$RUN-$c.err"
    return 1
  fi
  rm -rf "${DATA:?}/$c"
  mv "$TMP/$RUN-$c" "$DATA/$c"
  log "$c parties ($net) : $(($(date +%s) - start)) s, $(cat "$TMP/$RUN-$c.txt")"
  rm -f "$TMP/$RUN-$c.txt" "$TMP/$RUN-$c.err"
}

train() { # réseau, réseau de départ
  local c=$1 prev=$2 start
  [ -f "$OUT/$c.bin" ] && return 0
  start=$(date +%s)
  if ! (cd lab4/py && $PY train.py --cycles "../data/$RUN" --last "$c" --init "../runs/$RUN/$prev.bin" \
    --out "../runs/$RUN/$c.tmp.bin" --lr "$LR" --state-in "../runs/$RUN/$prev.state.pt" \
    --state-out "../runs/$RUN/$c.state.pt") > "$OUT/$c-train.log" 2>&1; then
    tail -20 "$OUT/$c-train.log"
    return 1
  fi
  mv "$OUT/$c.tmp.bin" "$OUT/$c.bin"
  log "$c entraînement : $(($(date +%s) - start)) s"
  sed 's/^/    /' "$OUT/$c-train.log" | grep -v "écrit" | tee -a "$OUT/log.txt"
}

duel() { # réseau : contre REF et son objectif
  local c=$1 start f
  f=$OUT/duel-$c-$REF-w$REF_WIN
  [ -f "$f.tsv" ] && return 0
  start=$(date +%s)
  $LAB arena --hero "az:net=$OUT/$c.bin,n=200,win=$WIN" --field "az:net=$OUT/$REF.bin,n=200,win=$REF_WIN" --players "$P" \
    --deals "$DEALS" --seed 1004 --stop "$STOP" --threads "$THREADS" --device vulkan --out "$f.tmp" > "$f.txt"
  mv "$f.tmp" "$f.tsv"
  mv "$f.tmp.search.tsv" "$f.tsv.search.tsv"
  log "$c (win = $WIN) face à $REF (win = $REF_WIN) : $(($(date +%s) - start)) s, $(grep -E '^(arrêt|test séquentiel)' "$f.txt")"
}

log "essai $RUN, tailles $P, $GAMES parties, n = $SIMS, win = $WIN, taux $LR, $THREADS fils ; graines $SEED_BASE + cycle ; pas $START à $LAST ; duels contre $REF à win = $REF_WIN"
for k in $(seq "$START" "$LAST"); do
  c=$(name "$k")
  prev=$(name $((k - 1)))
  games "$c" "$prev"
  train "$c" "$prev"
  duel "$c"
done
