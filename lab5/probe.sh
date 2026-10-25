#!/bin/bash
# Une seule étape, bornée dans le temps, sur la machine de calcul. Voir lab.md.
# RUN=pilot STAGE=positions LIMIT=900 lab5/probe.sh
# Même configuration à chaque étape ; le binaire valide le manifeste avant de reprendre.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=${RUN:?nom du lot sous lab5/runs}
STAGE=${STAGE:?positions, select ou confirm}
LIMIT=${LIMIT:?délai maximal en secondes, estimé avant chaque étape}
[[ "$RUN" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "RUN invalide" >&2; exit 2; }
[[ "$STAGE" =~ ^(positions|select|confirm)$ ]] || { echo "STAGE invalide" >&2; exit 2; }
[[ "$LIMIT" =~ ^[1-9][0-9]*$ ]] || { echo "LIMIT invalide" >&2; exit 2; }
if [[ $(hostname -s) == mini ]]; then
  echo "Les campagnes se lancent sur une machine de calcul, pas sur mini." >&2
  exit 2
fi
OUT=lab5/runs/$RUN
mkdir -p "$OUT"
timeout 15 sha256sum "${BIN:-target/release/lab5}" "${NET:-lab5/nets/c130.bin}" > "$OUT/binary-$STAGE.sha256"
timeout 15 git rev-parse HEAD > "$OUT/commit-$STAGE.txt" 2>/dev/null || true
timeout 15 uname -a > "$OUT/machine-$STAGE.txt"
timeout --kill-after=30s "${LIMIT}s" "${BIN:-target/release/lab5}" probe \
  --stage "$STAGE" --out "$OUT" --net "${NET:-lab5/nets/c130.bin}" \
  --players "${PLAYERS:-2,4}" --positions "${POSITIONS:-1}" --candidates "${CANDIDATES:-6}" \
  --select "${SELECT:-32}" --confirm "${CONFIRM:-128}" --sims "${SIMS:-400}" \
  --seed "${SEED:-51001}" --threads "${THREADS:-$(nproc)}" --concurrent "${CONCURRENT:-16}" \
  --device "${DEVICE:-cpu}" 2>&1 | tee -a "$OUT/$STAGE.log"
if [[ "$STAGE" == confirm ]]; then
  timeout 60 python3 lab5/py/probe.py "$OUT" | tee "$OUT/report.txt"
fi
