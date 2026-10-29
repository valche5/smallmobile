#!/bin/bash
# Machines louées sur vast.ai (un GPU NVIDIA et un gros processeur), pilotées depuis mini, pour le
# labo `LAB` : son programme est compilé sur mini (Vulkan, glslc 2026.4) puis envoyé ; la machine ne
# compile rien. Une machine à la fois.
#
#   labtools/vast.sh find                        offres conformes, les moins chères d'abord
#   LAB=labN labtools/vast.sh up OFFRE           loue, attend SSH, vérifie le GPU au repos, envoie le labo,
#                                                installe cuBLAS et NVRTC, compare le GPU au processeur
#                                                (Vulkan et CUDA, `labN check-gpu`)
#   LAB=labN labtools/vast.sh push               recompile sur mini, renvoie binaire, scripts et poids
#   labtools/vast.sh run NOM LIMITE 'commande'   calcul détaché depuis la racine du dépôt, LIMITE en s ;
#                                                rend la main aussitôt
#   labtools/vast.sh wait NOM [SECONDES]         suit le calcul jusqu'à sa fin, au plus SECONDES (600) ;
#                                                code 3 s'il tourne encore (relancer wait)
#   labtools/vast.sh get CHEMIN...               rapatrie vers les mêmes chemins du dépôt
#   labtools/vast.sh ssh [commande]              shell, ou commande courte (30 s)
#   labtools/vast.sh down                        rapatrie les journaux, détruit, vérifie, dépense
#   labtools/vast.sh status                      machines louées et crédit
#
# Chaque étape a un délai court et s'arrête net en cas d'échec, sa cause dans le journal :
# labtools/state/vast/vast.log, horodaté (commandes, réponses de vast, état de la machine, sorties).
# Si `up` échoue, il note les derniers messages de la machine (`vastai logs`) puis la détruit
# (KEEP=1 la garde pour diagnostic). Sur la machine, un calcul écrit vast-jobs/NOM.log (sa sortie,
# et toutes les 30 s la charge, le GPU et la mémoire) et vast-jobs/NOM.done (code, durée) ;
# `wait` et `down` les recopient dans labtools/state/vast/jobs/.
#
# SSH : connexion directe, délai de 10 s, connexion morte détectée en 30 s, trois essais si la
# connexion échoue. Les calculs tournent détachés (setsid) : une coupure ne les arrête pas.
#
# Réglages (variables d'environnement) : VAST_NAME (une machine de plus à côté, état séparé :
# labtools/state/NOM), IMAGE, LAB, NET (réseau du test Vulkan, labN/nets/c130.bin), MAX_PRICE ($/h, 0,6), DISK (Go, 16), KEEP, VAST (le
# client vastai), GLSLC_DIR (shaderc 2026.4 extrait, `usr/bin/glslc`), CUDA_PIP (cuBLAS et NVRTC
# installés par pip : ceux de CUDA 13.0, comme PyTorch cu130 ; plus récents, leur PTX est refusé par
# le pilote 595 des hôtes : CUDA_ERROR_UNSUPPORTED_PTX_VERSION, 26 octobre).
set -euo pipefail
cd "$(dirname "$0")/.."

VAST=${VAST:-$(command -v vastai || echo lab4/py/.venv/bin/vastai)}
IMAGE=${IMAGE:-vastai/base-image:stock-ubuntu26.04-2026-10-23}
# certains hôtes laissent authorized_keys avec des droits que sshd refuse
ONSTART='chown -R root:root /root/.ssh; chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys'
GLSLC_DIR=${GLSLC_DIR:-$HOME/.local/opt/shaderc-2026.4}
TARGET=${CARGO_TARGET_DIR:-$HOME/.cache/vast-target}
MAX_PRICE=${MAX_PRICE:-0.6}
CUDA_PIP=${CUDA_PIP:-nvidia-cublas==13.0.2.14 nvidia-cuda-nvrtc==13.0.88}
DISK=${DISK:-16}
KEEP=${KEEP:-}
LAB=${LAB:-}
ST=labtools/state/${VAST_NAME:-vast}
REMOTE=/root/sm
mkdir -p "$ST/jobs"
LOG=$ST/vast.log

say() { echo "[vast $(date +%H:%M:%S)] $*" | tee -a "$LOG" >&2; }
note() { sed "s/^/[vast $(date +%H:%M:%S)]   /" >> "$LOG"; }
die() { say "ERREUR : $*"; exit 1; }
need_lab() { [[ "$LAB" =~ ^lab[0-9]+$ ]] && [ -d "$LAB" ] || die "LAB=labN : le labo à envoyer"; }
st() { cat "$ST/$1" 2>/dev/null || true; }
need_up() { [ -n "$(st id)" ] || die "aucune machine louée (up)"; }
json() { python3 -I -c "$(printf 'import json, sys\nd = json.load(sys.stdin)\n%s' "$1")"; }

SSHOPT=(-o ConnectTimeout=10 -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -o BatchMode=yes
  -o LogLevel=ERROR -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$ST/known_hosts")
# rsh SECONDES commande… : réessaie si la connexion échoue (code 255), pas si la commande échoue
rsh() {
  local t=$1 i code
  shift
  for i in 1 2 3 4 5; do
    code=0
    timeout --kill-after=5 "$t" ssh "${SSHOPT[@]}" -p "$(st port)" "root@$(st host)" "$@" || code=$?
    [ "$code" -ne 255 ] && return "$code"
    say "ssh : connexion impossible (essai $i/5)"
    sleep $((i * 5))
  done
  return 255
}
# rget SECONDES SOURCE DESTINATION [options] : rsync depuis la machine, trois essais
rget() {
  local t=$1 src=$2 dst=$3 i
  shift 3
  for i in 1 2 3; do
    timeout "$t" rsync -az "$@" -e "ssh ${SSHOPT[*]} -p $(st port)" "root@$(st host):$src" "$dst" && return 0
    say "rsync : $src non copié (essai $i/3)"
    sleep $((i * 5))
  done
  return 1
}
rsync_e() { echo "ssh ${SSHOPT[*]} -p $(st port)"; }

info() { timeout 30 "$VAST" show instance "$(st id)" --raw 2>/dev/null; }

# destroy ID : détruit le contrat et attend qu'il disparaisse de la liste (60 s), sinon s'arrête
destroy() {
  local id=$1 i
  timeout 60 "$VAST" destroy instance "$id" -y 2>&1 | note || true
  for i in $(seq 12); do
    timeout 30 "$VAST" show instances --raw | json "print(any(x['id'] == $id for x in d))" | grep -q False && break
    sleep 5
  done
  timeout 30 "$VAST" show instances --raw | json "assert not any(x['id'] == $id for x in d)" ||
    die "la machine $id existe encore : vérifier sur cloud.vast.ai"
}

find_cmd() {
  say "recherche d'offres (au plus $MAX_PRICE \$/h)"
  timeout 60 "$VAST" search offers "cpu_cores_effective>=24 num_gpus=1 reliability>0.985 rentable=true \
verified=true dph_total<$MAX_PRICE inet_down>300 cpu_ghz>3.6 gpu_ram>=11" -o dph_total --limit 400 --raw |
    json "$(cat <<'PY'
import re
cpu = re.compile(r"Ryzen 9 (7|9)|Threadripper (PRO )?(7|9)|i9-1[34]|EPYC 9", re.I)
gpu = re.compile(r"50[789]0|40[789]0|3090|PRO 4500", re.I)
print(f"{'offre':>9} {'$/h':>6}  {'GPU':<13} {'fils':>4}  {'processeur':<30} {'RAM':>4}  {'fiab.':>5}  lieu")
for o in d:
    if cpu.search(o["cpu_name"]) and gpu.search(o["gpu_name"]):
        print(f"{o['id']:>9} {o['dph_total']:6.3f}  {o['gpu_name']:<13} {o['cpu_cores_effective']:4.0f}  "
              f"{o['cpu_name'][:30]:<30} {o['cpu_ram'] / 1024:3.0f}G  {o['reliability2']:.3f}  {o['geolocation']}")
PY
)"
}

build() {
  need_lab
  [ -x "$GLSLC_DIR/usr/bin/glslc" ] || die "glslc 2026.4 attendu dans $GLSLC_DIR (GLSLC_DIR=…)"
  say "compilation de $LAB (Vulkan) dans $TARGET"
  if ! PATH=$GLSLC_DIR/usr/bin:$PATH LD_LIBRARY_PATH=$GLSLC_DIR/usr/lib/x86_64-linux-gnu CARGO_TARGET_DIR=$TARGET \
    timeout 1500 nice cargo build --release -p "smallmobile-$LAB" > "$ST/build.txt" 2>&1; then
    tail -20 "$ST/build.txt" | note
    die "compilation échouée (journal : $LOG)"
  fi
  tail -1 "$ST/build.txt" | note
}

push_cmd() {
  need_up
  build
  local bin=$TARGET/release/$LAB
  say "envoi de $LAB/ et du binaire ($(sha256sum "$bin" | cut -c1-12))"
  rsh 30 "mkdir -p $REMOTE/target/release $REMOTE/vast-jobs"
  timeout 300 rsync -az --delete -e "$(rsync_e)" --exclude runs --exclude data --exclude .venv \
    --exclude __pycache__ "$LAB/" "root@$(st host):$REMOTE/$LAB/" || die "envoi de $LAB/ échoué"
  timeout 120 rsync -az -e "$(rsync_e)" "$bin" "root@$(st host):$REMOTE/target/release/$LAB" ||
    die "envoi du binaire échoué"
  { git rev-parse HEAD; git status --short | head -20; } 2>/dev/null |
    rsh 30 "cat > $REMOTE/vast-commit.txt"
  say "envoyé : $(rsh 30 "du -sh $REMOTE | cut -f1")"
  touch "$ST/pushed"
}

up_cmd() {
  need_lab
  local offer=${1:?numéro d\'offre (find)} out id s i
  [ -z "$(st id)" ] || die "machine $(st id) déjà louée (down d'abord)"
  build
  say "location de l'offre $offer ($IMAGE)"
  out=$(timeout 60 "$VAST" create instance "$offer" --image "$IMAGE" --disk "$DISK" --ssh --direct \
    --label "$LAB" --onstart-cmd "$ONSTART" --raw 2>&1) || true
  echo "$out" | note
  id=$(echo "$out" | json 'print(d["new_contract"] if d.get("success") else "")' 2>/dev/null) || true
  if [ -z "$id" ]; then
    # vast peut refuser la location et créer quand même le contrat (29 octobre : 58280590, resté
    # « loading » 4 h) : le détruire, sinon il peut démarrer plus tard et être facturé
    id=$(echo "$out" | json 'print(d.get("new_contract") or "")' 2>/dev/null) || true
    if [ -n "$id" ]; then
      say "location refusée, mais contrat $id créé : destruction"
      destroy "$id"
    fi
    die "location refusée"
  fi
  echo "$id" > "$ST/id"
  date +%s > "$ST/start"
  trap up_failed EXIT
  # 1. démarrage (5 min)
  local last=""
  for i in $(seq 30); do
    s=$(info | json 'print(d.get("actual_status"), str(d.get("status_msg") or "")[:70].replace("\n", " "))') || s="?"
    [ "$s" != "$last" ] && say "machine $id : $s"
    last=$s
    [[ $s == running* ]] && break
    sleep 10
  done
  [[ $s == running* ]] || die "pas démarrée en 5 min"
  info | json '
p = (d.get("ports") or {}).get("22/tcp") or []
print(d["public_ipaddr"].strip() if p else d["ssh_host"])
print(p[0]["HostPort"] if p else d["ssh_port"])
print(d["dph_total"])' > "$ST/conn"
  sed -n 1p "$ST/conn" > "$ST/host"
  sed -n 2p "$ST/conn" > "$ST/port"
  sed -n 3p "$ST/conn" > "$ST/dph"
  say "adresse $(st host):$(st port), $(LC_ALL=C printf %.3f "$(st dph)") \$/h"
  # 2. SSH (2 min)
  for i in $(seq 12); do
    timeout 15 ssh "${SSHOPT[@]}" -p "$(st port)" "root@$(st host)" true 2>/dev/null && break
    sleep 10
  done
  rsh 20 true || die "SSH ne répond pas en 2 min"
  # l'image continue de démarrer après `running` et relance son sshd : attendre trois connexions
  # réussies d'affilée, à 10 s d'intervalle (3 min au plus)
  local ok=0
  for i in $(seq 18); do
    sleep 10
    if timeout 15 ssh "${SSHOPT[@]}" -p "$(st port)" "root@$(st host)" true 2>/dev/null; then
      ok=$((ok + 1))
    else
      ok=0
      say "SSH coupé pendant le démarrage de l'image, nouvelle attente"
    fi
    [ "$ok" -ge 3 ] && break
  done
  [ "$ok" -ge 3 ] || die "SSH instable pendant 3 min"
  say "SSH prêt et stable"
  # 3. l'image finit son démarrage ; un GPU ouvert trop tôt a déjà bloqué des hôtes
  say "attente de 2 min avant de toucher au GPU (démarrage de l'image)"
  sleep 120
  rsh 30 'nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.total --format=csv,noheader;
    echo "$(nproc) fils, $(grep -m1 "model name" /proc/cpuinfo | cut -d: -f2-)";
    free -g | awk "/Mem/{print \$2\" Go de mémoire\"}"; df -h / | awk "NR==2{print \$4\" libres sur le disque\"}"' |
    tee "$ST/machine" | note
  say "$(paste -sd ';' "$ST/machine")"
  # occupé au repos à trois lectures : sur trois hôtes, le signe d'un GPU bloqué (Vulkan figé ensuite)
  local busy
  busy=$(rsh 30 'for i in 1 2 3; do nvidia-smi --query-gpu=utilization.gpu,power.draw,power.limit \
    --format=csv,noheader,nounits; sleep 3; done' | tee -a "$LOG" |
    awk -F', ' '$1 >= 90 { n++ } END { print n + 0 }')
  [ "$busy" -lt 3 ] || die "GPU occupé au repos : hôte défectueux"
  # 4. le labo ; cuBLAS et NVRTC pour `--device cuda` (paquets pip, sans PyTorch), chargés par les
  # calculs de `run` (cuda.env)
  push_cmd
  rsh 300 "py=/venv/main/bin/python; [ -x \$py ] || py=python3
    \$py -m pip install -q --break-system-packages $CUDA_PIP > /dev/null 2>&1 || true"'
    lib=$(dirname "$(find / -xdev -name "libcublas.so.1[0-9]" 2>/dev/null | head -1)")
    [ "$lib" != . ] && echo "export LD_LIBRARY_PATH=$lib:\${LD_LIBRARY_PATH:-}" > /root/sm/cuda.env; cat /root/sm/cuda.env' | note || true
  # 5. le GPU comparé au processeur, par moteur, détaché (60 s) : un pilote bloqué ne bloque pas mini
  local dev
  for dev in vulkan cuda; do
    rsh 20 "cd $REMOTE || exit 1; [ -f cuda.env ] && . ./cuda.env; setsid nohup timeout -s KILL 60 target/release/$LAB check-gpu \
      --net ${NET:-$LAB/nets/c130.bin} --device $dev > vast-jobs/check-$dev.txt 2>&1 < /dev/null & echo \$! > vast-jobs/check.pid"
    for i in $(seq 13); do
      sleep 5
      rsh 20 "kill -0 \$(cat $REMOTE/vast-jobs/check.pid) 2> /dev/null" || break
    done
    rsh 20 "cat $REMOTE/vast-jobs/check-$dev.txt" | tee "$ST/check-$dev" | note
    say "$dev : $(paste -sd ';' "$ST/check-$dev")"
  done
  # juste : victoire à moins de 0,01 du processeur
  local good=""
  for dev in cuda vulkan; do
    awk '/écart maximal/ { for (i = 1; i < NF; i++) if ($i == "victoire") ok = $(i + 1) < 0.01 } END { exit !ok }' \
      "$ST/check-$dev" && good="$good $dev"
  done
  [ -n "$good" ] || die "aucun moteur GPU juste (journal : $LOG)"
  say "moteurs justes :$good"
  trap - EXIT
  say "machine $id prête"
}
up_failed() {
  local code=$?
  trap - EXIT
  [ "$code" -eq 0 ] && return
  say "échec de up (code $code$([ "$code" -eq 124 ] && echo " : délai dépassé")) ; derniers messages de la machine"
  timeout 30 "$VAST" logs "$(st id)" --tail 60 2>&1 | note || true
  if [ -n "$KEEP" ]; then
    say "KEEP=1 : machine $(st id) gardée pour diagnostic (down pour la détruire)"
  else
    down_cmd
  fi
  exit "$code"
}

# le lanceur, sur la machine : NOM LIMITE, commande dans vast-jobs/NOM.cmd
JOB='cd /root/sm
[ -f cuda.env ] && . ./cuda.env
J=vast-jobs/$1
rm -f "$J.done"
echo "== début $(date "+%F %T"), limite $2 s : $(cat "$J.cmd")" >> "$J.log"
(while sleep 30; do
  echo "== état $(date +%T) : charge $(cut -d" " -f1-3 /proc/loadavg), GPU $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader), mémoire $(free -g | awk "/Mem/{print \$3\"/\"\$2\" Go\"}")"
done >> "$J.log") &
mon=$!
t=$SECONDS
timeout --kill-after=30 "$2" bash "$J.cmd" >> "$J.log" 2>&1
code=$?
kill $mon
echo "== fin $(date "+%F %T") : code $code en $((SECONDS - t)) s" >> "$J.log"
echo "$code $((SECONDS - t))" > "$J.done"'

run_cmd() {
  need_up
  local name=${1:?NOM} limit=${2:?LIMITE en secondes} cmd=${3:?commande}
  [[ "$name" =~ ^[A-Za-z0-9_-]+$ ]] || die "NOM invalide"
  [[ "$limit" =~ ^[1-9][0-9]*$ ]] || die "LIMITE invalide"
  rsh 20 "[ ! -f $REMOTE/vast-jobs/$name.log ] || [ -f $REMOTE/vast-jobs/$name.done ]" ||
    die "$name tourne déjà (wait $name)"
  rsh 20 "mkdir -p $REMOTE/vast-jobs && cat > $REMOTE/vast-jobs/$name.cmd" <<< "$cmd"
  rsh 20 "cat > $REMOTE/vast-jobs/job.sh" <<< "$JOB"
  # sans ps ni pgrep : ils peuvent rester bloqués sur un processus qui tient le GPU
  rsh 20 "cd $REMOTE || exit 1; setsid nohup bash vast-jobs/job.sh $name $limit > /dev/null 2>&1 < /dev/null &
    echo \$! > vast-jobs/$name.pid; sleep 1; kill -0 \$(cat vast-jobs/$name.pid) 2> /dev/null || [ -f vast-jobs/$name.done ]"
  say "calcul $name lancé (au plus $limit s) : $cmd"
}

wait_cmd() {
  need_up
  local name=${1:?NOM} max=${2:-600} seen=0 n t=$SECONDS done=""
  local J=$REMOTE/vast-jobs/$name
  rsh 20 "[ -f $J.log ]" || die "aucun calcul $name"
  while :; do
    if ! n=$(rsh 20 "wc -l < $J.log"); then
      info | json 'print(d.get("actual_status"))' | grep -q running || die "machine disparue"
      die "machine injoignable"
    fi
    if [ "$n" -gt "$seen" ]; then
      rsh 20 "tail -n +$((seen + 1)) $J.log | tail -15" | tee -a "$LOG"
      seen=$n
    fi
    done=$(rsh 20 "cat $J.done 2>/dev/null" || true)
    [ -n "$done" ] && break
    [ $((SECONDS - t)) -ge "$max" ] && break
    sleep 15
  done
  rget 60 "$J.log" "$ST/jobs/$(st id)-$name.log" || true
  if [ -z "$done" ]; then
    say "$name tourne encore après $max s (wait $name pour reprendre)"
    return 3
  fi
  say "$name terminé : code ${done% *} en ${done#* } s"
  return "${done% *}"
}

get_cmd() {
  need_up
  local p
  for p in "$@"; do
    say "rapatriement de $p"
    rget 600 "$REMOTE/./$p" . --relative || die "rapatriement de $p échoué"
  done
}

ssh_cmd() {
  need_up
  if [ $# -eq 0 ]; then
    ssh "${SSHOPT[@]}" -p "$(st port)" -t "root@$(st host)"
  else
    rsh 30 "$*"
  fi
}

down_cmd() {
  local id cost
  id=$(st id)
  [ -n "$id" ] || { say "aucune machine louée"; status_cmd; return; }
  if [ -f "$ST/pushed" ]; then
    rget 60 "$REMOTE/vast-jobs/" "$ST/jobs/$id/" || true
  fi
  say "destruction de la machine $id"
  destroy "$id"
  if [ -n "$(st dph)" ]; then
    cost=$(python3 -I -c "print(f'{$(st dph) * ($(date +%s) - $(st start)) / 3600:.2f}')")
    say "machine $id détruite après $((($(date +%s) - $(st start)) / 60)) min, environ $cost \$"
  else
    say "machine $id détruite"
  fi
  rm -f "$ST"/{id,start,conn,host,port,dph,machine,vulkan,check-vulkan,check-cuda,pushed}
  status_cmd
}

status_cmd() {
  timeout 30 "$VAST" show instances --raw | json "$(cat <<'PY'
for x in d:
    print(f"machine {x['id']} : {x['actual_status']}, {x['gpu_name']}, {x['dph_total']:.3f} $/h")
print("machines louées :", len(d))
PY
)"
  timeout 30 "$VAST" show user --raw | json "print(f\"crédit restant : {d['credit']:.2f} \$\")"
}

case "${1:-}" in
  find) find_cmd ;;
  up) shift; up_cmd "$@" ;;
  push) need_lab; push_cmd ;;
  run) shift; run_cmd "$@" ;;
  wait) shift; wait_cmd "$@" ;;
  get) shift; get_cmd "$@" ;;
  ssh) shift; ssh_cmd "$@" ;;
  down) down_cmd ;;
  status) status_cmd ;;
  *) sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
