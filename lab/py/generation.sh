#!/bin/bash
# Une génération d'auto-apprentissage :
#   1. la recherche guidée par le réseau N joue contre elle-même → lab/data/genN ;
#   2. un nouveau réseau, parti du réseau N, apprend sur les dernières générations (fenêtre
#      glissante) et un peu de parties de l'IA Normal → lab/nets/genN+1 ;
#   3. il affronte l'IA Normal et l'IA Difficile pour suivre les progrès (lab/runs/genN+1-arena.txt).
#
#   lab/py/generation.sh N
# Réglages (variables d'environnement) : GAMES, WINDOW, SEARCH (options de la recherche), EPOCHS,
# NORMAL (part des parties de l'IA Normal), DEALS.
set -euo pipefail
cd "$(dirname "$0")/../.."
N=$1
NEXT=$((N + 1))
GAMES=${GAMES:-6000}
WINDOW=${WINDOW:-3}
SEARCH=${SEARCH:-n=200,w=0,roll=1,c=2.5}
EPOCHS=${EPOCHS:-4}
NORMAL=${NORMAL:-0.1}
DEALS=${DEALS:-100}
LAB=./target/release/lab

if [ ! -f "lab/data/gen$N/.done" ]; then
  echo "== génération $N : $GAMES parties ($SEARCH)"
  timeout 36000 $LAB selfplay --agent "az:net=lab/nets/gen$N.bin,$SEARCH,noise=0.25" \
    --games "$GAMES" --out "lab/data/gen$N" --seed "$((N * 1000003))" \
    --full-share 0.25 --fast 50 > "lab/runs/gen$N.log" 2>&1
  touch "lab/data/gen$N/.done"
fi

DATA=()
for ((g = N; g > N - WINDOW && g >= 0; g--)); do DATA+=("../data/gen$g"); done
DATA+=("../data/normal@$NORMAL")
echo "== apprentissage de gen$NEXT sur ${DATA[*]}"
(cd lab/py && timeout 36000 .venv/bin/python train.py --data "${DATA[@]}" --init "../nets/gen$N.pt" \
  --out "../nets/gen$NEXT.bin" --epochs "$EPOCHS" --lr 1e-3 2>&1 | grep -v rocSHMEM) | tee "lab/runs/gen$NEXT-train.log"

for field in normal hard; do
  echo "== gen$NEXT contre $field"
  timeout 36000 $LAB arena --hero "az:net=lab/nets/gen$NEXT.bin,$SEARCH" --field $field --deals "$DEALS" --seed 77 \
    2>/dev/null | tee -a "lab/runs/gen$NEXT-arena.txt"
done
