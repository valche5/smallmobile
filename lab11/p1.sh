#!/bin/bash
# Pilotage de la boucle p1 de lab11 (lab11/lab.md) sur le PC :
#
#   lab11/p1.sh status   où en est la boucle (processus, dernières lignes du journal)
#   lab11/p1.sh stop     arrêt propre : la boucle s'arrête avant le pas suivant (jusqu'à ~2 h)
#   lab11/p1.sh kill     arrêt immédiat : l'étape en cours (parties, apprentissage ou duel) est perdue
#                        et refaite à la reprise
#   lab11/p1.sh start    reprise, détachée du terminal, avec les réglages choisis au lancement
#                        (professeur et taille des cycles, lus dans lab11/runs/start-p1.log)
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=lab11/runs/p1
LAST=40

# les processus de la boucle (et du lancement automatique) : leurs groupes, pour tuer aussi les
# programmes qu'ils ont lancés
groups() {
  pgrep -f '^/bin/bash (lab11/loop\.sh|lab11/runs/start-p1\.sh)' | xargs -r -n1 ps -o pgid= -p | sort -u
}

case ${1:-status} in
  status)
    if [ -n "$(groups)" ]; then echo "en cours"; else echo "arrêtée"; fi
    [ -f "$OUT/stop" ] && echo "arrêt demandé (avant le pas suivant)"
    tail -n 8 "$OUT/log.txt" 2> /dev/null | grep -v '^    ' || true
    ;;
  stop)
    mkdir -p "$OUT"
    touch "$OUT/stop"
    echo "arrêt demandé : la boucle s'arrêtera avant le pas suivant"
    ;;
  kill)
    for g in $(groups); do kill -TERM -- "-$g"; done
    rm -f "$OUT/stop"
    echo "boucle arrêtée"
    ;;
  start)
    [ -z "$(groups)" ] || { echo "la boucle tourne déjà"; exit 1; }
    agent=$(sed -n 's/.*professeur : n=400,//p' lab11/runs/start-p1.log | tail -1)
    games=$(sed -n 's/.*cycles de \([0-9]*\) parties.*/\1/p' lab11/runs/start-p1.log | tail -1)
    [ -n "$agent" ] && [ -n "$games" ] || { echo "réglages introuvables dans lab11/runs/start-p1.log"; exit 1; }
    rm -f "$OUT/stop"
    echo "reprise : professeur n=400,$agent, cycles de $games parties, jusqu'à c0$LAST"
    RUN=p1 SIMS=400 AGENT=$agent VALUE_MIX=0.5 GAMES=$games FILL_SIMS=400 FILL_AGENT= FILL_GAMES=25000 \
      setsid nohup lab11/loop.sh "$LAST" >> "$OUT/loop.out" 2>&1 < /dev/null &
    ;;
  *)
    sed -n '2,10p' "$0"
    exit 1
    ;;
esac
