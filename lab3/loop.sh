#!/bin/bash
# La boucle d'auto-apprentissage de lab3 (lab3/journal.md), à la manière de KataGo, pour un réseau
# pour P joueurs (un nombre, ou une liste comme 2,3,4,5 pour un généraliste). Pas k :
#   - en même temps : l'entraînement du réseau k depuis le réseau k−1, sur une fenêtre des derniers
#     cycles jusqu'au cycle k (lab3/py/train.py, TRAIN cœurs), et les parties du cycle k+1 jouées
#     par le réseau k−1 (PLAY cœurs) : un cycle de retard, comme les joueurs d'AlphaZero et de
#     KataGo pendant que le réseau apprend ;
#   - puis, tous les ARENA_EVERY pas (chaque pas sur GPU, tous les 4 sur CPU) et au dernier, le duel
#     du réseau k contre REF (c000 par défaut) aux tailles P, mêmes donnes à chaque fois (graine
#     1004), arrêté par le test séquentiel ; sur GPU avec DEVICE. Les
#     arènes face aux IA du jeu (Difficile, Normale) ne mesurent plus les progrès à ce niveau
#     (lab2/journal.md) et prenaient la moitié d'un pas.
# Le réseau de départ c000 est le généraliste de lab2 (lab3/nets/start.bin, son c060). Il joue
# d'abord, avec tous les cœurs, les cycles 1 à F qui remplissent la fenêtre minimale
# (lab3/py/data.py : `filled`, F = 4 à 4 joueurs) ; le premier pas est F (réseau cF entraîné depuis
# c000, sur toute la fenêtre), et c000 joue aussi le cycle F+1.
#
#   lab3/loop.sh P N      remplit la fenêtre, puis fait les pas F à N du réseau à P joueurs ; une
#                         étape déjà faite est sautée (reprise)
#
# Parties : lab3/data/pP/cNNN ; réseaux : lab3/runs/pP/cNNN.bin ; duels
# lab3/runs/pP/duel-cNNN-REF.tsv ; journal lab3/runs/pP/log.txt (durée de chaque
# étape). Sur AWS : --save lab3/data/pP,lab3/runs/pP. Réglages (essais sur mini) : GAMES, DEALS,
# PLAY, TRAIN ; DEVICE=vulkan : le réseau des parties sur GPU (`lab3 selfplay --device vulkan`).
# L'entraînement Python prend le GPU quand il y en a un, indépendamment de DEVICE.
# La fenêtre de l'apprentissage est lue sous forme compacte (lab3/py/data.py, `pack` : un fichier
# `.pos.packed.npz` fait une fois à côté de chaque `.pos`), en RAM.
# Sur GPU : étapes successives par défaut (PARALLEL=0), THREADS fils de jeu (tous les cœurs par
# défaut) ; les parties du cycle k+1 y sont jouées par le réseau k, qui vient d'apprendre. TRAIN règle seulement les opérations CPU de PyTorch ; aucun cœur ne lui est réservé.
# Sur CPU : PARALLEL=1 conserve le recouvrement historique avec PLAY et TRAIN cœurs.
# SEED_BASE=30000060 : continuer les graines après les 60 cycles de lab2 (défaut : 30000000).
# Pour une liste, les chemins utilisent des tirets (p2-3-4-5) : une virgule sépare les options
# dans la notation d'un agent `az:net=…,n=200` et ne peut donc pas faire partie de son chemin.
set -euo pipefail
cd "$(dirname "$0")/.."
P=${1:?le nombre de joueurs}
GROUP=${P//,/-}
LAST=${2:?le nombre de pas}
GAMES=${GAMES:-25000}
# le duel de mesure (`arenas`) : contre REF, n = 200 des deux côtés, DEALS donnes au plus, arrêté
# dès que le test séquentiel conclut entre un gain de STOP points et rien (labtools/README.md)
DEALS=${DEALS:-460}
REF=${REF:-c000}
STOP=${STOP:-3}
# sur GPU, un duel prend moins d'une minute (lab3 arena --device vulkan) : à chaque pas ; sur CPU,
# ~15 min : tous les 4 pas
case "$DEVICE" in
  cpu) ARENA_EVERY=${ARENA_EVERY:-4} ;;
  *) ARENA_EVERY=${ARENA_EVERY:-1} ;;
esac
SEED_BASE=${SEED_BASE:-30000000}
DEVICE=${DEVICE:-cpu}
case "$DEVICE" in
  cpu) PARALLEL=${PARALLEL:-1} ;;
  *) PARALLEL=${PARALLEL:-0} ;;
esac
if [ "$PARALLEL" = 1 ]; then
  # Recette CPU sur c7a.16xlarge : générations et apprentissage se partagent les cœurs.
  PLAY=${PLAY:-48}
  TRAIN=${TRAIN:-16}
  THREADS=$((PLAY + TRAIN))
else
  PLAY=${THREADS:-$(nproc)}
  THREADS=$PLAY
  TRAIN=${TRAIN:-4}
fi
# RUN : le nom des dossiers de parties et de réseaux (un essai à part, voir START)
RUN=${RUN:-p$GROUP}
DATA=lab3/data/$RUN
OUT=lab3/runs/$RUN
TMP=lab3/data/tmp
LAB=./target/release/lab3
mkdir -p "$DATA" "$OUT" "$TMP"
[ -f "$OUT/c000.bin" ] || cp lab3/nets/start.bin "$OUT/c000.bin"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/log.txt"; }
name() { printf c%03d "$1"; }

filled() { # cycle : oui si la fenêtre minimale est remplie, non sinon
  (cd lab3/py && .venv/bin/python -c "from data import filled; print('oui' if filled('../data/$RUN', '$1') else 'non')")
}

player() { # le réseau qui joue le cycle k. En parallèle : c000 jusqu'au premier pas compris, puis
  # k−2 (le cycle k se joue pendant l'apprentissage de k−1). En successif : le dernier appris, k−1.
  if [ "$PARALLEL" = 1 ]; then
    if [ "$1" -le $((first + 1)) ]; then echo c000; else name $(($1 - 2)); fi
  else
    if [ "$1" -le "$first" ]; then echo c000; else name $(($1 - 1)); fi
  fi
}

games() { # cycle, réseau, cœurs : écrites à part, puis déplacées. Un cycle fini a ses fichiers
  # .pos.games (--save peut recréer un dossier vide ; les .pos des cycles sortis de la fenêtre sont
  # effacés du bucket par lab3/prune.sh, les .games restent)
  local c=$1 net=$2 threads=$3 start
  compgen -G "$DATA/$c/*.pos.games" > /dev/null && return 0
  start=$(date +%s)
  rm -rf "${TMP:?}/$RUN-$c"
  if ! $LAB selfplay --agent "az:net=$OUT/$net.bin,n=200" --players "$P" --games "$GAMES" \
    --seed $((SEED_BASE + 10#${c#c})) --threads "$threads" --device "$DEVICE" --out "$TMP/$RUN-$c" \
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
  if ! (cd lab3/py && .venv/bin/python train.py --cycles "../data/$RUN" --last "$c" --init "../runs/$RUN/$prev.bin" \
    --out "../runs/$RUN/$c.tmp.bin" --threads "$TRAIN") > "$OUT/$c-train.log" 2>&1; then
    tail -20 "$OUT/$c-train.log"
    return 1
  fi
  mv "$OUT/$c.tmp.bin" "$OUT/$c.bin"
  log "$c entraînement : $(($(date +%s) - start)) s"
  sed 's/^/    /' "$OUT/$c-train.log" | grep -v "écrit" | tee -a "$OUT/log.txt"
}

arenas() { # réseau, pas : son duel contre REF, tous les ARENA_EVERY pas et au dernier
  local c=$1 k=$2 start
  [ $((k % ARENA_EVERY)) -eq 0 ] || [ "$k" -eq "$LAST" ] || return 0
  [ -f "$OUT/duel-$c-$REF.tsv" ] && return 0
  start=$(date +%s)
  $LAB arena --hero "az:net=$OUT/$c.bin,n=200" --field "az:net=$OUT/$REF.bin,n=200" --players "$P" \
    --deals "$DEALS" --seed 1004 --stop "$STOP" --threads "$THREADS" --device "$DEVICE" --out "$OUT/duel-$c-$REF.tmp" \
    > "$OUT/duel-$c-$REF.txt"
  mv "$OUT/duel-$c-$REF.tmp" "$OUT/duel-$c-$REF.tsv"
  log "$c face à $REF : $(($(date +%s) - start)) s, $(grep -E '^(arrêt|test séquentiel)' "$OUT/duel-$c-$REF.txt")"
}

# en cas d'arrêt, les étapes lancées en même temps s'arrêtent aussi (par numéro de processus)
pids=()
trap 'for p in "${pids[@]}"; do pkill -P "$p" 2> /dev/null || true; kill "$p" 2> /dev/null || true; done' EXIT

# le remplissage de la fenêtre par c000
log "tailles $P, $DEVICE, étapes $([ "$PARALLEL" = 1 ] && echo parallèles || echo successives), $THREADS fils ; graines $SEED_BASE + cycle ; jusqu'à $(name "$LAST")"
first=1
games c001 c000 "$THREADS"
while :; do
  # une erreur de Python arrête la boucle (set -e)
  full=$(filled "$(name "$first")")
  [ "$full" = oui ] && break
  first=$((first + 1))
  games "$(name "$first")" c000 "$THREADS"
done

# START : commencer au pas START (un essai qui reprend la fenêtre d'un autre : ses cycles liés dans
# DATA, le réseau de départ en OUT/c(START−1).bin)
for k in $(seq "$((${START:-0} > first ? ${START:-0} : first))" "$LAST"); do
  c=$(name "$k")
  # le départ de l'apprentissage : le dernier réseau appris (en parallèle, c'est aussi celui qui
  # joue le cycle k+1 pendant ce temps)
  if [ "$k" -le "$first" ]; then prev=c000; else prev=$(name $((k - 1))); fi
  # les parties du cycle k (déjà jouées au pas précédent, sauf à une reprise)
  games "$c" "$(player "$k")" "$THREADS"
  if [ "$PARALLEL" = 1 ]; then
    # Recouvrement historique : le réseau k et les parties du cycle k+1 par le réseau d'avant.
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
  else
    # Un seul utilisateur du GPU : les parties du cycle suivant seront jouées au tour suivant.
    train "$c" "$prev"
  fi
  arenas "$c" "$k"
done
