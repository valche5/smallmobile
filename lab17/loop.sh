#!/bin/bash
# La boucle d'auto-apprentissage de lab17 (lab17/lab.md, étape 3) : celle de lab11 (reprise de lab7 et
# lab3), sur le PC (RX 9070 XT, `--device vulkan`, PyTorch ROCm de lab3/py/.venv), depuis c130w (c130
# élargi aux extensions, `py/widen.py`). Les parties tirent leurs extensions au hasard
# (`LAB17_EXPANSIONS=mix`, `src/table.rs` : un quart sans extension). Avant le premier pas, c000
# (c130w) joue les cycles qui remplissent la fenêtre minimale (lab17/py/data.py, `filled`) ; puis, au
# pas k :
#   1. les parties du cycle k, jouées par le réseau k−1 (`lab17 selfplay`, recherches à SIMS
#      simulations, options AGENT ; bruit de Gumbel aux NOISY premières manches) ;
#   2. l'apprentissage du réseau k depuis k−1 sur la fenêtre des derniers cycles (lab17/py/train.py) ;
#      les positions des cycles sortis de la fenêtre sont ensuite effacées ;
#   3. le duel du réseau k contre c000, n = 200 des deux côtés, **avec les trois extensions**, mêmes
#      donnes à chaque pas (graine 1704), arrêté par le test séquentiel ; tous les BASE_EVERY pas, le
#      même duel **sans extension** (graine 1705) : le jeu de base ne doit pas reculer.
#
#   RUN=nom lab17/loop.sh N        jusqu'au pas N ; une étape déjà faite est sautée (reprise)
#
# Arrêt : le fichier lab17/runs/RUN/stop arrête la boucle avant le pas suivant ; tuée en cours de
# route, elle refait à la reprise l'étape interrompue (parties, apprentissage ou duel), rien d'autre.
#
# Parties : lab17/data/RUN/cNNN ; réseaux et duels : lab17/runs/RUN ; journal lab17/runs/RUN/log.txt.
# Réglages : GAMES (25 000), SIMS (400), AGENT (options de la recherche), NOISY (manches avec bruit,
# toutes par défaut), LR (1,5e-4), DEALS (460), STOP (3), BASE_EVERY (5), SEED_BASE (170 000 000),
# THREADS (tous), DEVICE (vulkan), PY (lab3/py/.venv), CONCURRENT (16), FINAL (0 : pas de mesure
# finale ; les mesures finales sont faites à part, lab.md étape 4), et ceux du remplissage
# (FILL_GAMES, FILL_SIMS, FILL_AGENT, FILL_NOISY), FAST_ROLL, LUCK, VALUE_MIX comme dans lab11.
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
FINAL=${FINAL:-0}
DEALS=${DEALS:-460}
STOP=${STOP:-3}
SEED_BASE=${SEED_BASE:-170000000}
BASE_EVERY=${BASE_EVERY:-5}
THREADS=${THREADS:-$(nproc)}
DEVICE=${DEVICE:-vulkan}
PY=${PY:-$PWD/lab3/py/.venv/bin/python}
# parties menées par fil dans le selfplay (16 sur le PC, comme lab4 ; 32 sur RTX 5080 en CUDA dans
# lab7). Pas dans les duels : l'arène partage l'anneau du GPU entre ses deux réseaux.
CONCURRENT=${CONCURRENT:-16}
DATA=lab17/data/$RUN
OUT=lab17/runs/$RUN
TMP=lab17/data/tmp
LAB=./target/release/lab17
mkdir -p "$DATA" "$OUT" "$TMP"
[ -f "$OUT/c000.bin" ] || cp lab17/nets/c130w.bin "$OUT/c000.bin"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/log.txt"; }
name() { printf c%03d "$1"; }
python() { (cd lab17/py && "$PY" -c "$1"); }

games() { # cycle, réseau, parties, simulations, options, manches avec bruit : écrites à part, puis
  # déplacées ; un cycle fini a ses .pos.games
  local c=$1 net=$2 count=$3 sims=$4 agent=$5 noisy=$6 start
  compgen -G "$DATA/$c/*.pos.games" > /dev/null && return 0
  start=$(date +%s)
  rm -rf "${TMP:?}/$RUN-$c"
  if ! LAB17_EXPANSIONS=mix LAB17_CONCURRENT=$CONCURRENT $LAB selfplay --agent "az:net=$OUT/$net.bin,n=$sims${agent:+,$agent}" --players "$P" --games "$count" \
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
  if ! (cd lab17/py && "$PY" train.py --cycles "../data/$RUN" --last "$c" --init "../runs/$RUN/$prev.bin" \
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

duel() { # réseau, extensions (all, none), graine : contre c000, le réseau seul changé
  local c=$1 x=$2 seed=$3 start f=$OUT/duel-$1-c000-$2
  [ -f "$f.tsv" ] && return 0
  start=$(date +%s)
  LAB17_EXPANSIONS=$x $LAB arena --hero "az:net=$OUT/$c.bin,n=200" --field "az:net=$OUT/c000.bin,n=200" --players "$P" \
    --deals "$DEALS" --seed "$seed" --stop "$STOP" --threads "$THREADS" --device "$DEVICE" --out "$f.tmp" > "$f.txt" 2> /dev/null
  mv "$f.tmp" "$f.tsv"
  log "$c face à c000 ($x) : $(($(date +%s) - start)) s, $(grep -E "écart à l'attendu" "$f.txt" | sed 's/.*écart/écart/'), $(grep -E '^(arrêt|test séquentiel)' "$f.txt")"
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
  # arrêt propre entre deux pas : `touch lab17/runs/RUN/stop` (lab17/p1.sh stop) ; relancer reprend ici
  if [ -f "$OUT/stop" ]; then
    rm -f "$OUT/stop"
    log "arrêt demandé, avant $c"
    exit 0
  fi
  if [ "$k" -le "$first" ]; then prev=c000; else prev=$(name $((k - 1))); fi
  # le cycle k par le dernier réseau appris (celui du premier pas : déjà joué par c000)
  games "$c" "$prev" "$GAMES" "$SIMS" "$AGENT" "$NOISY"
  train "$c" "$prev"
  duel "$c" all 1704
  if [ $((k % BASE_EVERY)) = 0 ]; then duel "$c" none 1705; fi
done
# la mesure finale : le dernier réseau contre c000, n = 400 des deux côtés, sans arrêt
c=$(name "$LAST")
f=$OUT/final-$c
if [ "$FINAL" -gt 0 ] && [ ! -f "$f.tsv" ]; then
  start=$(date +%s)
  LAB17_EXPANSIONS=all $LAB arena --hero "az:net=$OUT/$c.bin,n=400" --field "az:net=$OUT/c000.bin,n=400" --players "$P" \
    --deals "$FINAL" --seed 172101 --threads "$THREADS" --device "$DEVICE" --out "$f.tmp" > "$f.txt"
  mv "$f.tmp" "$f.tsv"
  log "mesure finale de $c face à c000 : $(($(date +%s) - start)) s, $(grep -E 'écart à l' "$f.txt")"
fi
