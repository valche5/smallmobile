#!/bin/bash
# La boucle d'auto-apprentissage de lab9 (phase 6, lab9/lab.md), sur le modèle de celles de lab3 à lab7,
# sur ce PC (Vulkan), en étapes successives depuis c000 (h1 par défaut). Au pas k :
#   1. les parties du cycle k, jouées par le réseau k−1 (`lab9 selfplay` : tous les joueurs cherchent
#      leurs plans, SEARCH ; plan joué tiré selon la politique améliorée aux EXPLORE premières manches) ;
#   2. l'apprentissage du réseau k depuis k−1 (lab9/py/train.py, AdamW neuf, STEPS lots au taux LR,
#      moyenne glissante EMA) sur les WINDOW derniers cycles, une part IMIT_SHARE de chaque lot tirée des
#      parties d'imitation IMIT (le réseau ne dérive pas tant que la fenêtre est petite) ;
#   3. le duel du réseau k contre c000, la recherche sur plans des deux côtés (DUEL_SEARCH), DEALS donnes
#      (graine 1004, les mêmes à chaque pas), arrêté par le test séquentiel (STOP points).
#
#   RUN=nom lab9/loop.sh N        jusqu'au pas N ; une étape déjà faite est sautée (reprise)
#
# Parties : lab9/data/RUN/cNNN ; réseaux et duels : lab9/runs/RUN ; journal lab9/runs/RUN/log.txt.
# Réglages : START (réseau de départ), GAMES (5 000), SEARCH (k=8,m=2), BETA (vide : celui de mctx),
# EXPLORE (2), THREADS (200, parties menées ensemble : le GPU veut de gros lots), STEPS (4 000), BATCH
# (256), LR (1e-4), EMA (0,998), WINDOW (4 cycles), IMIT (lab8/data/c130), IMIT_SHARE (0,25),
# DUEL_SEARCH (k=8,m=2), DEALS (460), DUEL_THREADS (150), STOP (3), SEED_BASE (90 000 000), PY (le python de PyTorch), LAB
# (le programme lab9 ; une copie figée si l'on compile pendant la boucle).
set -euo pipefail
cd "$(dirname "$0")/.."
# 11 fichiers par fil du selfplay (200 fils) : la limite de 1 024 d'un service systemd ne suffit pas
ulimit -n 65536
LAST=${1:?le dernier pas}
RUN=${RUN:?le nom de l’essai}
START=${START:-lab9/runs/h1/h1-150000.bin}
GAMES=${GAMES:-5000}
SEARCH=${SEARCH:-k=8,m=2}
BETA=${BETA:-}
EXPLORE=${EXPLORE:-2}
THREADS=${THREADS:-200}
STEPS=${STEPS:-4000}
BATCH=${BATCH:-256}
LR=${LR:-1e-4}
EMA=${EMA:-0.998}
WINDOW=${WINDOW:-4}
IMIT=${IMIT:-lab8/data/c130}
IMIT_SHARE=${IMIT_SHARE:-0.25}
DUEL_SEARCH=${DUEL_SEARCH:-k=8,m=2}
DEALS=${DEALS:-460}
# parties menées ensemble dans un duel : 20 (un fil par cœur) laissaient au GPU des lots de 9 lectures,
# 8,6 s par donne ; 100 fils : 3,2 s
DUEL_THREADS=${DUEL_THREADS:-150}
STOP=${STOP:-3}
SEED_BASE=${SEED_BASE:-90000000}
PY=${PY:-$PWD/lab3/py/.venv/bin/python}
DATA=lab9/data/$RUN
OUT=lab9/runs/$RUN
# le programme : une copie figée, pour qu'une compilation pendant la boucle ne la touche pas
LAB=${LAB:-./target/release/lab9}
mkdir -p "$DATA" "$OUT"
[ -f "$OUT/c000.bin" ] || cp "$START" "$OUT/c000.bin"
log() { echo "[$(date '+%d %H:%M:%S')] $*" | tee -a "$OUT/log.txt"; }
name() { printf c%03d "$1"; }

games() { # cycle, réseau : écrites à part, puis déplacées ; un cycle fini a son dossier
  local c=$1 net=$2 start
  [ -d "$DATA/$c" ] && return 0
  start=$(date +%s)
  rm -rf "$DATA/$c.tmp"
  if ! $LAB selfplay --net "$OUT/$net.bin" --games "$GAMES" --search "$SEARCH" ${BETA:+--beta "$BETA"} \
    --explore "$EXPLORE" --seed $((SEED_BASE + 10#${c#c})) --threads "$THREADS" --device vulkan \
    --out "$DATA/$c.tmp" > "$OUT/$c-games.txt" 2>&1; then
    tail -5 "$OUT/$c-games.txt"
    return 1
  fi
  mv "$DATA/$c.tmp" "$DATA/$c"
  log "$c parties ($net) : $(($(date +%s) - start)) s, $(grep '^selfplay' "$OUT/$c-games.txt") ; $(grep -E "^$GAMES/" "$OUT/$c-games.txt")"
}

train() { # réseau, réseau de départ, pas
  local c=$1 prev=$2 k=$3 start first dirs
  [ -f "$OUT/$c.bin" ] && return 0
  start=$(date +%s)
  first=$((k - WINDOW + 1))
  [ "$first" -lt 1 ] && first=1
  dirs=$(for i in $(seq "$first" "$k"); do printf "../../%s/%s," "$DATA" "$(name "$i")"; done)
  if ! (cd lab9/py && "$PY" train.py --data "${dirs%,}" --imitation "../../$IMIT" --imitation-share "$IMIT_SHARE" \
    --init "../../$OUT/$prev.bin" --steps "$STEPS" --batch "$BATCH" --lr "$LR" --ema "$EMA" --val-every "$STEPS" \
    --value-every 4 --out "../../$OUT/$c.tmp.bin") > "$OUT/$c-train.log" 2>&1; then
    tail -20 "$OUT/$c-train.log"
    return 1
  fi
  mv "$OUT/$c.tmp.bin" "$OUT/$c.bin"
  rm -f "$OUT/$c.tmp-"*.bin
  log "$c entraînement (cycles $(name "$first") à $c) : $(($(date +%s) - start)) s"
  grep -E "plans \(|validation" "$OUT/$c-train.log" | sed 's/^ */    /' | tee -a "$OUT/log.txt"
}

duel() { # réseau : contre c000, la même recherche des deux côtés
  local c=$1 start f=$OUT/duel-$1-c000
  [ -f "$f.tsv" ] && return 0
  start=$(date +%s)
  $LAB arena --hero "plans:net=$OUT/$c.bin,gpu=1,$DUEL_SEARCH" --field "plans:net=$OUT/c000.bin,gpu=1,$DUEL_SEARCH" \
    --deals "$DEALS" --seed 1004 --stop "$STOP" --threads "$DUEL_THREADS" --out "$f.tmp" > "$f.txt" 2> "$f.err"
  mv "$f.tmp" "$f.tsv"
  log "$c face à c000 : $(($(date +%s) - start)) s, $(grep -E '^(arrêt|test séquentiel)' "$f.txt")"
}

log "essai $RUN depuis $START : $GAMES parties par cycle, $SEARCH, beta ${BETA:-de mctx}, $EXPLORE manches tirées, $THREADS parties ensemble ; $STEPS lots de $BATCH, taux $LR, EMA $EMA, fenêtre $WINDOW cycles, imitation $IMIT pour $IMIT_SHARE ; duels $DUEL_SEARCH, $DEALS donnes, arrêt à $STOP points ; jusqu'à $(name "$LAST")"
for k in $(seq 1 "$LAST"); do
  c=$(name "$k")
  prev=$(name $((k - 1)))
  games "$c" "$prev"
  train "$c" "$prev" "$k"
  duel "$c"
done
