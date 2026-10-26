#!/bin/bash
# La boucle d'auto-apprentissage de lab8 (phase 2, lab8/lab.md), reprise de celles de lab3 et lab4,
# pour une machine à GPU NVIDIA (vast.ai, `--device cuda`), en étapes successives, depuis c130 et une
# fenêtre neuve. Avant le premier pas, c000 (c130) joue les cycles qui remplissent la fenêtre
# minimale (lab8/py/data.py, `filled`) ; puis, au pas k :
#   1. les parties du cycle k, jouées par le réseau k−1 (`lab8 selfplay`, recherches complètes à SIMS
#      simulations avec les options AGENT de la recherche, bruit de Gumbel aux NOISY premières
#      manches seulement, fins simulées des recherches rapides si FAST_ROLL=1) ;
#   2. l'apprentissage du réseau k depuis k−1 sur la fenêtre des derniers cycles (lab8/py/train.py,
#      AdamW neuf à chaque pas, taux maximal LR en un cycle) ; les positions des cycles sortis de la
#      fenêtre sont ensuite effacées (leurs `.games` et `.moves` restent) ;
#   3. le duel du réseau k contre c000, n = 200 des deux côtés, sans option de recherche (le réseau
#      seul change), mêmes donnes à chaque pas (graine 1004), arrêté par le test séquentiel.
#
#   RUN=nom lab8/loop.sh N        jusqu'au pas N ; une étape déjà faite est sautée (reprise)
#
# Après le dernier pas : la mesure finale, le dernier réseau contre c000 à n = 400 des deux côtés,
# FINAL donnes sans arrêt (graine 72101 : les mêmes donnes pour tous les bras), `final-cNNN.tsv`.
#
# Parties : lab8/data/RUN/cNNN ; réseaux et duels : lab8/runs/RUN ; journal lab8/runs/RUN/log.txt.
# Réglages : GAMES (25 000), SIMS (400), AGENT (options de la recherche du selfplay, par exemple
# `roll=1`), NOISY (manches avec bruit, `selfplay --noisy-rounds` ; toutes par défaut, comme lab3),
# FAST_ROLL (0), LUCK (0), LR (1,5e-4, le dernier réglage de lab3 qui a payé), DEALS (460), STOP (3), SEED_BASE
# (70 000 000), THREADS (tous), DEVICE (cuda), PY (le python de PyTorch, /venv/main/bin/python),
# FINAL (920, 0 : pas de mesure finale). Le remplissage de la fenêtre a ses propres FILL_GAMES,
# FILL_SIMS, FILL_AGENT, FILL_NOISY (par défaut ceux des pas) : un professeur coûteux (`AGENT=roll=1`)
# peut partir d'une fenêtre remplie par la recette ordinaire.
set -euo pipefail
cd "$(dirname "$0")/.."
LAST=${1:?le dernier pas}
RUN=${RUN:?le nom de l’essai}
P=${P:-2,3,4,5}
GAMES=${GAMES:-25000}
SIMS=${SIMS:-400}
AGENT=${AGENT:-}
NOISY=${NOISY:-1000}
FAST_ROLL=${FAST_ROLL:-0}
# LUCK=1 : la part du hasard de chaque résultat (`selfplay --luck`), retranchée de la cible de la
# valeur (`train.py --luck`)
LUCK=${LUCK:-0}
# VALUE_MIX : part de la valeur de la recherche des positions enregistrées dans la cible de la valeur
# (`train.py --value-mix` ; un professeur à fins simulées donne ainsi les deux cibles)
VALUE_MIX=${VALUE_MIX:-0}
LR=${LR:-1.5e-4}
FILL_GAMES=${FILL_GAMES:-$GAMES}
FILL_SIMS=${FILL_SIMS:-$SIMS}
FILL_AGENT=${FILL_AGENT-$AGENT}
FILL_NOISY=${FILL_NOISY:-$NOISY}
FINAL=${FINAL:-920}
DEALS=${DEALS:-460}
STOP=${STOP:-3}
SEED_BASE=${SEED_BASE:-70000000}
THREADS=${THREADS:-$(nproc)}
DEVICE=${DEVICE:-cuda}
PY=${PY:-/venv/main/bin/python}
# parties menées par fil dans le selfplay : 32 plutôt que 16 (lots du GPU deux fois plus gros, +16 %
# de positions par seconde sur une RTX 5080 en CUDA, 26 octobre). Pas dans les duels : l'arène
# partage l'anneau du GPU entre ses deux réseaux, 32 parties par fil n'y tiennent pas.
CONCURRENT=${CONCURRENT:-32}
DATA=lab8/data/$RUN
OUT=lab8/runs/$RUN
TMP=lab8/data/tmp
LAB=./target/release/lab8
mkdir -p "$DATA" "$OUT" "$TMP"
[ -f "$OUT/c000.bin" ] || cp lab8/nets/c130.bin "$OUT/c000.bin"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/log.txt"; }
name() { printf c%03d "$1"; }
python() { (cd lab8/py && "$PY" -c "$1"); }

games() { # cycle, réseau, parties, simulations, options, manches avec bruit : écrites à part, puis
  # déplacées ; un cycle fini a ses .pos.games
  local c=$1 net=$2 count=$3 sims=$4 agent=$5 noisy=$6 start
  compgen -G "$DATA/$c/*.pos.games" > /dev/null && return 0
  start=$(date +%s)
  rm -rf "${TMP:?}/$RUN-$c"
  if ! LAB8_CONCURRENT=$CONCURRENT $LAB selfplay --agent "az:net=$OUT/$net.bin,n=$sims${agent:+,$agent}" --players "$P" --games "$count" \
    --seed $((SEED_BASE + 10#${c#c})) --threads "$THREADS" --device "$DEVICE" --noisy-rounds "$noisy" \
    --fast-roll "$FAST_ROLL" --luck "$LUCK" --out "$TMP/$RUN-$c" > "$TMP/$RUN-$c.txt" 2> "$TMP/$RUN-$c.err"; then
    tail -5 "$TMP/$RUN-$c.err"
    return 1
  fi
  rm -rf "${DATA:?}/$c"
  mv "$TMP/$RUN-$c" "$DATA/$c"
  log "$c parties ($net) : $(($(date +%s) - start)) s, $(cat "$TMP/$RUN-$c.txt")"
  rm -f "$TMP/$RUN-$c.txt" "$TMP/$RUN-$c.err"
}

train() { # réseau, réseau de départ
  local c=$1 prev=$2 start old
  [ -f "$OUT/$c.bin" ] && return 0
  start=$(date +%s)
  local luck=()
  [ "$LUCK" = 1 ] && luck=(--luck)
  if ! (cd lab8/py && "$PY" train.py --cycles "../data/$RUN" --last "$c" --init "../runs/$RUN/$prev.bin" \
    --out "../runs/$RUN/$c.tmp.bin" --lr "$LR" --value-mix "$VALUE_MIX" "${luck[@]}") > "$OUT/$c-train.log" 2>&1; then
    tail -20 "$OUT/$c-train.log"
    return 1
  fi
  mv "$OUT/$c.tmp.bin" "$OUT/$c.bin"
  log "$c entraînement : $(($(date +%s) - start)) s"
  sed 's/^/    /' "$OUT/$c-train.log" | grep -v "écrit" | tee -a "$OUT/log.txt"
  # le disque de la machine : les positions des cycles sortis de la fenêtre
  for old in $(python "from data import outside; print(*outside('../data/$RUN'))"); do
    rm -rf "$DATA/$old"/*.pos.compact
  done
}

duel() { # réseau : contre c000, le réseau seul changé
  local c=$1 start f=$OUT/duel-$1-c000
  [ -f "$f.tsv" ] && return 0
  start=$(date +%s)
  $LAB arena --hero "az:net=$OUT/$c.bin,n=200" --field "az:net=$OUT/c000.bin,n=200" --players "$P" \
    --deals "$DEALS" --seed 1004 --stop "$STOP" --threads "$THREADS" --device "$DEVICE" --out "$f.tmp" > "$f.txt"
  mv "$f.tmp" "$f.tsv"
  log "$c face à c000 : $(($(date +%s) - start)) s, $(grep -E '^(arrêt|test séquentiel)' "$f.txt")"
}

log "essai $RUN : tailles $P, $GAMES parties, n = $SIMS${AGENT:+ ($AGENT)}, bruit aux $NOISY premières manches, fins des rapides $FAST_ROLL, hasard retranché $LUCK, valeur de la recherche $VALUE_MIX, taux $LR, $THREADS fils, $DEVICE ; remplissage $FILL_GAMES parties, n = $FILL_SIMS${FILL_AGENT:+ ($FILL_AGENT)}, bruit aux $FILL_NOISY premières manches ; graines $SEED_BASE + cycle ; jusqu'à $(name "$LAST")"
# le remplissage de la fenêtre par c000
first=1
games c001 c000 "$FILL_GAMES" "$FILL_SIMS" "$FILL_AGENT" "$FILL_NOISY"
while [ "$(python "from data import filled; print('oui' if filled('../data/$RUN', '$(name "$first")') else 'non')")" != oui ]; do
  first=$((first + 1))
  games "$(name "$first")" c000 "$FILL_GAMES" "$FILL_SIMS" "$FILL_AGENT" "$FILL_NOISY"
done
for k in $(seq "$first" "$LAST"); do
  c=$(name "$k")
  if [ "$k" -le "$first" ]; then prev=c000; else prev=$(name $((k - 1))); fi
  # le cycle k par le dernier réseau appris (celui du premier pas : déjà joué par c000)
  games "$c" "$prev" "$GAMES" "$SIMS" "$AGENT" "$NOISY"
  train "$c" "$prev"
  duel "$c"
done
# la mesure finale : le dernier réseau contre c000, n = 400 des deux côtés, sans arrêt
c=$(name "$LAST")
f=$OUT/final-$c
if [ "$FINAL" -gt 0 ] && [ ! -f "$f.tsv" ]; then
  start=$(date +%s)
  $LAB arena --hero "az:net=$OUT/$c.bin,n=400" --field "az:net=$OUT/c000.bin,n=400" --players "$P" \
    --deals "$FINAL" --seed 72101 --threads "$THREADS" --device "$DEVICE" --out "$f.tmp" > "$f.txt"
  mv "$f.tmp" "$f.tsv"
  log "mesure finale de $c face à c000 : $(($(date +%s) - start)) s, $(grep -E 'écart à l' "$f.txt")"
fi
