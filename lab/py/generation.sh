#!/bin/bash
# Une génération d'auto-apprentissage :
#   1. le réseau de la génération N joue contre lui-même (recherche guidée) → lab/data/genN ;
#   2. un nouveau réseau apprend sur les dernières générations (fenêtre glissante) → lab/nets/genN+1 ;
#   3. il affronte l'IA Difficile pour suivre les progrès (lab/runs/genN+1-arena.txt).
#
#   lab/py/generation.sh N [parties] [fenêtre] [itérations]
set -euo pipefail
cd "$(dirname "$0")/../.."
N=$1
GAMES=${2:-3000}
WINDOW=${3:-3}
ITER=${4:-200}
NEXT=$((N + 1))
LAB=./target/release/lab
NET=lab/nets/gen$N.bin

if [ ! -s "lab/data/gen$N/.done" ]; then
  echo "== génération $N : $GAMES parties ($ITER itérations, 25 % de recherches complètes)"
  timeout 36000 $LAB selfplay --agent "az:net=$NET,n=$ITER,c=1.5,noise=0.25,alpha=0.3" \
    --games "$GAMES" --out "lab/data/gen$N" --seed "$((N * 1000003))" \
    --full-share 0.25 --fast "$((ITER / 4))" > "lab/runs/gen$N.log" 2>&1
  echo ok > "lab/data/gen$N/.done"
fi

DATA=()
for ((g = N; g > N - WINDOW && g >= 0; g--)); do DATA+=("lab/data/gen$g"); done
echo "== apprentissage de gen$NEXT sur ${DATA[*]}"
INIT=()
[ -f "lab/nets/gen$N.pt" ] && INIT=(--init "lab/nets/gen$N.pt")
(cd lab/py && timeout 36000 .venv/bin/python train.py --data "${DATA[@]/#/../../}" --out "../nets/gen$NEXT.bin" \
  "${INIT[@]}" --epochs "${EPOCHS:-4}" 2>&1 | grep -v rocSHMEM) | tee "lab/runs/gen$NEXT-train.log"

echo "== gen$NEXT contre l'IA Difficile"
timeout 36000 $LAB arena --hero "az:net=lab/nets/gen$NEXT.bin,n=$ITER" --field hard --deals "${DEALS:-100}" 2>/dev/null \
  | tee "lab/runs/gen$NEXT-arena.txt"
