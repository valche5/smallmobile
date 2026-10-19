#!/bin/bash
# Machines de calcul AWS (EC2, eu-north-1), pilotées depuis mini qui ne calcule pas.
#
#   lab/tools/cloud.sh job [--type T] [--disk GO] [--data D1,D2] [--get CHEMIN,...] -- 'commande'
#       tout d'un coup : up, data, run, get, down (la machine est détruite même en cas d'erreur)
#
#   lab/tools/cloud.sh up [--type T] [--disk GO]   lance une machine, installe, compile le labo
#   lab/tools/cloud.sh push                        renvoie le code et les poids, recompile
#   lab/tools/cloud.sh data D1 D2...               copie lab/data/Di depuis le bucket (gratuit)
#   lab/tools/cloud.sh run 'commande'              calcul depuis la racine du dépôt ; attend la fin
#   lab/tools/cloud.sh wait                        se rattache au calcul en cours
#   lab/tools/cloud.sh get CHEMIN...               rapatrie des fichiers ou dossiers vers mini
#   lab/tools/cloud.sh ssh [commande]              shell sur la machine
#   lab/tools/cloud.sh down                        détruit la machine, vérifie qu'il ne reste rien
#   lab/tools/cloud.sh status                      machines, disques, adresses et dépense du mois
#   lab/tools/cloud.sh guard                       (re)pose les alarmes AWS de la machine
#   lab/tools/cloud.sh init                        une fois : clé SSH et groupe de sécurité
#
# Garde-fous sur la machine (installés au démarrage, indépendants de mini) : extinction au bout de
# MAX_HOURS heures, et après IDLE_MIN minutes sans calcul (charge < 1) ; une machine qui s'éteint
# est détruite, son disque avec. Pas d'adresse IP fixe : l'adresse publique part avec la machine.
# Hors de la machine (alarmes AWS, `guard`) : destruction si le processeur reste sous 1 % pendant
# IDLE_MIN minutes, ou si elle ne répond plus aux vérifications d'AWS pendant 10 min.
# Réglages (variables d'environnement) : TYPE, DISK, MAX_HOURS, IDLE_MIN, MAX_PRICE ($/h), SPOT=1
# (Spot uniquement, plafond de 1 $/h et durée maximale de 12 h).
set -euo pipefail
cd "$(dirname "$0")/../.."

export AWS_PROFILE=${AWS_PROFILE:-smallai}
export AWS_REGION=eu-north-1
BUCKET=smallai-lab-631245464763
TYPE=${TYPE:-c7a.16xlarge}
DISK=${DISK:-100}
MAX_HOURS=${MAX_HOURS:-12}
IDLE_MIN=${IDLE_MIN:-20}
MAX_PRICE=${MAX_PRICE:-1}
SPOT=${SPOT:-1}
SPOT_MAX=1.00
KEY=$HOME/.ssh/smallai_ed25519
KEYNAME=smallai
SG_NAME=smallai-ssh
PROFILE_NAME=smallai-worker
STATE=lab/runs/cloud
mkdir -p "$STATE"

say() { echo "[cloud $(date +%H:%M:%S)] $*" >&2; }
die() { say "ERREUR : $*"; exit 1; }

state() { cat "$STATE/$1" 2>/dev/null || true; }
IP() { state ip; }
ssh_opts() {
  echo -i "$KEY" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$STATE/known_hosts" \
    -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -o LogLevel=ERROR
}
on() {
  [ -n "$(IP)" ] || die "aucune machine (lab/tools/cloud.sh up)"
  # shellcheck disable=SC2046
  ssh $(ssh_opts) "admin@$(IP)" "$@"
}

price() { # prix horaire à la demande (ou Spot actuel) du type $1, en $
  if [ "$SPOT" = 1 ]; then
    aws ec2 describe-spot-price-history --instance-types "$1" --product-descriptions Linux/UNIX \
      --start-time "$(date -u +%FT%TZ)" --query 'SpotPriceHistory[].SpotPrice' --output text |
      tr '\t' '\n' | sort -g | tail -n 1
    return
  fi
  aws pricing get-products --region us-east-1 --service-code AmazonEC2 --filters \
    Type=TERM_MATCH,Field=instanceType,Value="$1" Type=TERM_MATCH,Field=regionCode,Value=$AWS_REGION \
    Type=TERM_MATCH,Field=operatingSystem,Value=Linux Type=TERM_MATCH,Field=tenancy,Value=Shared \
    Type=TERM_MATCH,Field=preInstalledSw,Value=NA Type=TERM_MATCH,Field=capacitystatus,Value=Used \
    --query 'PriceList[0]' --output text |
    python3 -c "import json,sys;t=json.load(sys.stdin)['terms']['OnDemand'];print(next(p['pricePerUnit']['USD'] for v in t.values() for p in v['priceDimensions'].values()))"
}

sg_id() {
  aws ec2 describe-security-groups --filters Name=group-name,Values=$SG_NAME \
    --query 'SecurityGroups[0].GroupId' --output text
}

cmd_init() {
  if [ ! -f "$KEY" ]; then
    ssh-keygen -q -t ed25519 -N '' -C smallai -f "$KEY"
  fi
  if ! aws ec2 describe-key-pairs --key-names $KEYNAME >/dev/null 2>&1; then
    aws ec2 import-key-pair --key-name $KEYNAME --public-key-material "fileb://$KEY.pub" >/dev/null
    say "clé SSH $KEYNAME importée"
  fi
  if [ "$(sg_id)" = None ]; then
    aws ec2 create-security-group --group-name $SG_NAME \
      --description "SSH depuis mini vers les machines de calcul" \
      --tag-specifications 'ResourceType=security-group,Tags=[{Key=Project,Value=smallai}]' >/dev/null
    say "groupe de sécurité $SG_NAME créé"
  fi
}

open_ssh() { # SSH ouvert à la seule adresse actuelle de mini
  local sg me old
  sg=$(sg_id)
  me="$(curl -s -m 10 https://checkip.amazonaws.com)/32"
  [ "$me" != /32 ] || die "adresse publique de mini introuvable"
  old=$(aws ec2 describe-security-groups --group-ids "$sg" \
    --query 'SecurityGroups[0].IpPermissions[?FromPort==`22`].IpRanges[].CidrIp' --output text)
  for cidr in $old; do
    [ "$cidr" = "$me" ] || aws ec2 revoke-security-group-ingress --group-id "$sg" \
      --protocol tcp --port 22 --cidr "$cidr" >/dev/null
  done
  [[ " $old " == *" $me "* ]] || aws ec2 authorize-security-group-ingress --group-id "$sg" \
    --protocol tcp --port 22 --cidr "$me" >/dev/null
}

user_data() {
  cat <<EOF
#!/bin/bash
# Garde-fous d'abord : extinction (donc destruction) au bout de $MAX_HOURS h, ou après $IDLE_MIN min
# sans calcul. Service système et échéance sur disque : ils survivent à un redémarrage.
echo \$(( \$(date +%s) + $MAX_HOURS * 3600 )) > /etc/smallai-deadline
systemd-run --unit smallai-deadline --on-active=${MAX_HOURS}h /usr/bin/systemctl poweroff
cat > /usr/local/bin/smallai-guard <<'GUARD'
#!/bin/bash
idle=0
while true; do
  if [ "\$(date +%s)" -ge "\$(cat /etc/smallai-deadline)" ]; then systemctl poweroff; fi
  if [ "\$idle" -ge $IDLE_MIN ]; then systemctl poweroff; fi
  sleep 60
  if awk '{exit !(\$1 < 1)}' /proc/loadavg; then idle=\$((idle + 1)); else idle=0; fi
  echo "\$idle" > /run/smallai-idle-minutes
done
GUARD
chmod +x /usr/local/bin/smallai-guard
cat > /etc/systemd/system/smallai-guard.service <<'UNIT'
[Unit]
Description=SmallAI : extinction après l'échéance ou sans calcul
[Service]
ExecStart=/usr/local/bin/smallai-guard
Restart=always
[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now smallai-guard

export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -y -q build-essential pkg-config python3-venv rsync curl unzip
curl -sS https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o /tmp/awscli.zip
unzip -q /tmp/awscli.zip -d /tmp && /tmp/aws/install
sudo -u admin bash -c 'curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal'
touch /var/lib/smallai-ready
EOF
}

cmd_up() {
  [ "$SPOT" = 1 ] || die "Spot obligatoire : le tarif à la demande est désactivé"
  awk -v p="$MAX_PRICE" 'BEGIN {exit !(p > 0 && p <= 1)}' || die "MAX_PRICE doit être compris entre 0 et 1 $/h"
  [[ "$MAX_HOURS" =~ ^[0-9]+$ ]] && [ "$MAX_HOURS" -ge 1 ] && [ "$MAX_HOURS" -le 12 ] || die "MAX_HOURS doit être compris entre 1 et 12 heures"
  [ -z "$(state id)" ] || die "une machine existe déjà ($(state id)) : lab/tools/cloud.sh down"
  cmd_init
  local p
  p=$(price "$TYPE")
  awk -v p="$p" -v m="$MAX_PRICE" 'BEGIN{exit !(p <= m)}' ||
    die "$TYPE coûte $p \$/h, plus que MAX_PRICE=$MAX_PRICE"
  open_ssh
  local ami rootdev market=() id=""
  ami=$(aws ssm get-parameter --name /aws/service/debian/release/13/latest/amd64 \
    --query Parameter.Value --output text)
  rootdev=$(aws ec2 describe-images --image-ids "$ami" --query 'Images[0].RootDeviceName' --output text)
  if [ "$SPOT" = 1 ]; then
    market=(--instance-market-options "MarketType=spot,SpotOptions={MaxPrice=$SPOT_MAX,SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}")
  fi
  user_data > "$STATE/user-data.sh"
  say "lancement d'une $TYPE ($p \$/h, disque $DISK Go, au plus $MAX_HOURS h)"
  # Une zone après l'autre si la première n'a plus de machines de ce type.
  for subnet in $(aws ec2 describe-subnets --filters Name=default-for-az,Values=true \
    --query 'Subnets[].SubnetId' --output text); do
    if id=$(aws ec2 run-instances --image-id "$ami" --instance-type "$TYPE" --count 1 \
      --subnet-id "$subnet" --key-name $KEYNAME --security-group-ids "$(sg_id)" \
      --iam-instance-profile Name=$PROFILE_NAME --instance-initiated-shutdown-behavior terminate \
      --metadata-options HttpTokens=required "${market[@]}" \
      --block-device-mappings "[{\"DeviceName\":\"$rootdev\",\"Ebs\":{\"VolumeSize\":$DISK,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
      --user-data "file://$STATE/user-data.sh" \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Project,Value=smallai},{Key=Name,Value=smallai-$TYPE}]" \
      "ResourceType=volume,Tags=[{Key=Project,Value=smallai}]" \
      --query 'Instances[0].InstanceId' --output text 2> "$STATE/run-error"); then
      break
    fi
    grep -q -E 'InsufficientInstanceCapacity|Unsupported' "$STATE/run-error" || die "$(cat "$STATE/run-error")"
    say "pas de place dans $subnet, zone suivante"
    id=""
  done
  [ -n "$id" ] || die "aucune zone n'a de $TYPE disponible"
  echo "$id" > "$STATE/id"
  echo "$TYPE" > "$STATE/type"
  echo "$p" > "$STATE/price"
  echo "$DISK" > "$STATE/disk"
  date +%s > "$STATE/start"
  rm -f "$STATE/known_hosts"
  aws ec2 wait instance-running --instance-ids "$id"
  aws ec2 describe-instances --instance-ids "$id" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text > "$STATE/ip"
  cmd_guard
  say "machine $id à $(IP), installation"
  local t
  for t in $(seq 60); do on true 2>/dev/null && break; sleep 5; done
  for t in $(seq 120); do on test -f /var/lib/smallai-ready && break; sleep 5; done
  on test -f /var/lib/smallai-ready || die "installation non terminée après 10 min (lab/tools/cloud.sh ssh)"
  cmd_push
  say "prête : $(on nproc) cœurs, $(on free -g | awk '/Mem/{print $2}') Go de mémoire"
}

cmd_guard() { # garde-fous hors de la machine : alarmes AWS qui la détruisent
  local id terminate
  id=$(state id)
  [ -n "$id" ] || die "aucune machine"
  terminate=arn:aws:automate:$AWS_REGION:ec2:terminate
  # Moins d'un cœur sur 64 occupé (1 %) pendant IDLE_MIN minutes (mesures toutes les 5 min).
  aws cloudwatch put-metric-alarm --alarm-name "smallai-$id-idle" \
    --alarm-description "détruit $id après $IDLE_MIN min sans calcul" \
    --namespace AWS/EC2 --metric-name CPUUtilization --dimensions Name=InstanceId,Value="$id" \
    --statistic Maximum --period 300 --evaluation-periods $(((IDLE_MIN + 4) / 5)) --threshold 1 \
    --comparison-operator LessThanThreshold --treat-missing-data missing --alarm-actions "$terminate"
  # Machine qui ne répond plus aux vérifications d'AWS pendant 10 min.
  aws cloudwatch put-metric-alarm --alarm-name "smallai-$id-dead" \
    --alarm-description "détruit $id si elle ne répond plus" \
    --namespace AWS/EC2 --metric-name StatusCheckFailed --dimensions Name=InstanceId,Value="$id" \
    --statistic Maximum --period 60 --evaluation-periods 10 --threshold 1 \
    --comparison-operator GreaterThanOrEqualToThreshold --treat-missing-data missing --alarm-actions "$terminate"
}

clean_alarms() { # alarmes des machines qui n'existent plus
  local alarm id st
  for alarm in $(aws cloudwatch describe-alarms --alarm-name-prefix smallai- \
    --query 'MetricAlarms[].AlarmName' --output text); do
    id=$(echo "$alarm" | sed -E 's/^smallai-(i-[0-9a-f]+)-.*/\1/')
    st=$(aws ec2 describe-instances --instance-ids "$id" \
      --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null || echo gone)
    case $st in pending | running | stopping | stopped | shutting-down) ;;
      *) aws cloudwatch delete-alarms --alarm-names "$alarm" ;; esac
  done
}

cmd_push() {
  say "envoi du code et des poids"
  tar -czf - --exclude=lab/py/.venv --exclude=__pycache__ \
    Cargo.toml Cargo.lock rustfmt.toml kernel brain lab/Cargo.toml lab/src lab/tests lab/py lab/nets lab/runs/*.sh |
    on 'mkdir -p smallmobile && tar -xzf - -C smallmobile'
  say "compilation du labo et environnement Python"
  on 'cd smallmobile && mkdir -p lab/runs lab/data &&
      (lab/py/setup.sh cpu > lab/runs/setup-py.log 2>&1 & py=$!;
       ~/.cargo/bin/cargo build --release -q -p smallmobile-lab 2>&1 | tail -20;
       wait $py || { tail -20 lab/runs/setup-py.log; exit 1; })'
}

cmd_data() {
  local d
  for d in "$@"; do
    say "données $d depuis le bucket"
    on "aws s3 sync --only-show-errors --region $AWS_REGION s3://$BUCKET/data/$d smallmobile/lab/data/$d"
  done
}

cmd_run() {
  [ $# -ge 1 ] || die "commande manquante"
  printf 'cd ~/smallmobile\n%s\n' "$*" | on 'cat > job.sh'
  on 'rm -f job.exit job.log; setsid nohup bash -c "bash job.sh > job.log 2>&1; echo \$? > job.exit" \
      > /dev/null 2>&1 < /dev/null &'
  say "calcul lancé : $*"
  cmd_wait
}

cmd_wait() { # affiche le journal du calcul au fil de l'eau, rend son code de sortie
  local off=0 code
  while true; do
    # Le code de sortie d'abord : s'il est là, le journal lu ensuite est complet.
    code=$(on 'cat job.exit 2>/dev/null; true') || { sleep 30; continue; }
    on "s=\$(stat -c %s job.log 2>/dev/null || echo 0); echo \$s; tail -c +$((off + 1)) job.log 2>/dev/null | head -c \$((s - $off))" \
      > "$STATE/chunk" || { sleep 30; continue; }
    off=$(head -n 1 "$STATE/chunk")
    tail -n +2 "$STATE/chunk"
    if [ -n "$code" ]; then
      say "calcul terminé (code $code)"
      return "$code"
    fi
    sleep 20
  done
}

cmd_get() {
  local p
  for p in "$@"; do
    say "rapatriement de $p"
    rsync -azR -e "ssh $(ssh_opts)" "admin@$(IP):smallmobile/./$p" .
  done
}

cmd_ssh() {
  [ -n "$(IP)" ] || die "aucune machine"
  # shellcheck disable=SC2046
  ssh -t $(ssh_opts) "admin@$(IP)" "$@"
}

cost_of_run() { # coût estimé de la machine actuelle : calcul + disque + adresse IPv4
  local start end
  start=$(state start)
  end=$(date +%s)
  awk -v s="$((end - start))" -v p="$(state price)" -v d="$(state disk)" 'BEGIN {
    h = s / 3600; if (s < 60) h = 60 / 3600
    c = h * p + d * 0.0836 / 730 * h + 0.005 * h
    printf "%d min, environ %.2f $ (machine %.2f, disque %.3f, adresse %.3f)\n", s / 60, c, h * p, d * 0.0836 / 730 * h, 0.005 * h }'
}

cmd_down() {
  local id
  id=$(state id)
  if [ -n "$id" ]; then
    say "destruction de $id"
    aws ec2 terminate-instances --instance-ids "$id" >/dev/null
    aws ec2 wait instance-terminated --instance-ids "$id"
    say "détruite après $(cost_of_run)"
    rm -f "$STATE"/{id,ip,type,price,disk,start,known_hosts}
  fi
  cmd_check
}

cmd_check() { # rien ne doit rester qui coûte : machines, disques, adresses fixes, instantanés, alarmes
  local left=0 x
  clean_alarms
  x=$(aws cloudwatch describe-alarms --alarm-name-prefix smallai- \
    --query 'MetricAlarms[].[AlarmName,StateValue]' --output text)
  [ -z "$x" ] || { say "alarmes (machine en vie) :"; echo "$x"; }
  x=$(aws ec2 describe-instances --filters Name=instance-state-name,Values=pending,running,stopping,stopped,shutting-down \
    --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name]' --output text)
  [ -z "$x" ] || { say "machines encore là :"; echo "$x"; left=1; }
  x=$(aws ec2 describe-volumes --query 'Volumes[].[VolumeId,Size,State]' --output text)
  [ -z "$x" ] || { say "disques encore là :"; echo "$x"; left=1; }
  x=$(aws ec2 describe-addresses --query 'Addresses[].PublicIp' --output text)
  [ -z "$x" ] || { say "adresses IP fixes encore là :"; echo "$x"; left=1; }
  x=$(aws ec2 describe-snapshots --owner-ids self --query 'Snapshots[].SnapshotId' --output text)
  [ -z "$x" ] || { say "instantanés encore là :"; echo "$x"; left=1; }
  [ "$left" = 1 ] || say "aucune machine, aucun disque, aucune adresse fixe à $AWS_REGION"
  return "$left"
}

cmd_status() {
  if [ -n "$(state id)" ]; then say "machine $(state id) ($(state type)) à $(IP) depuis $(cost_of_run)"; fi
  cmd_check || true
  local month
  month=$(date +%Y-%m-01)
  aws ce get-cost-and-usage --region us-east-1 --time-period "Start=$month,End=$(date -d tomorrow +%F)" \
    --granularity MONTHLY --metrics UnblendedCost --group-by Type=DIMENSION,Key=SERVICE \
    --query 'ResultsByTime[0].Groups[].[Keys[0],Metrics.UnblendedCost.Amount]' --output text |
    awk '{a=$NF; $NF=""; if (a > 0.001) printf "  %-60s %8.3f $\n", $0, a; t += a}
      END {printf "dépense du mois (avant crédits, connue avec quelques heures de retard) : %.2f $\n", t}'
  aws freetier get-account-plan-state --region us-east-1 \
    --query 'accountPlanRemainingCredits.amount' --output text | sed 's/^/crédits restants : /; s/$/ $/'
}

cmd_job() {
  local data="" get=""
  while [ $# -gt 0 ] && [ "$1" != -- ]; do
    case $1 in
      --type) TYPE=$2 ;;
      --disk) DISK=$2 ;;
      --data) data=$2 ;;
      --get) get=$2 ;;
      *) die "option inconnue : $1" ;;
    esac
    shift 2
  done
  [ "${1:-}" = -- ] || die "usage : job [options] -- 'commande'"
  shift
  trap 'cmd_down' EXIT
  cmd_up
  [ -z "$data" ] || cmd_data ${data//,/ }
  local code=0
  cmd_run "$*" || code=$?
  [ -z "$get" ] || cmd_get ${get//,/ }
  return "$code"
}

case "${1:-}" in
  init | up | guard | push | data | run | wait | get | ssh | down | status | job)
    c=$1
    shift
    if [ "$c" = up ]; then
      while [ $# -gt 0 ]; do
        case $1 in --type) TYPE=$2 ;; --disk) DISK=$2 ;; *) die "option inconnue : $1" ;; esac
        shift 2
      done
    fi
    "cmd_$c" "$@"
    ;;
  check) cmd_check ;;
  *) sed -n '2,27p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
