"""Entraîne le réseau sur les parties de la boucle (`lab3/loop.sh`), puis l'exporte pour Rust.

    python train.py --cycles ../data/p4 --last c005 --init ../runs/p4/c004.bin --out ../runs/p4/c005.bin

Un pas de la boucle : le cycle `--last` (un sous-dossier de `--cycles`) vient d'être joué.
L'entraînement repart du réseau précédent `--init`, sur une fenêtre des derniers cycles jusqu'à
`--last`, fenêtre qui grandit moins vite que le total des parties (comme KataGo : les données anciennes en
sortent), et chaque nouvelle position y sert en moyenne `USES` fois. Les positions sont tirées en
proportion de leur poids (surprise de la recherche, comme KataGo).

Cibles : la politique (politique améliorée de la recherche à la racine, Gumbel AlphaZero), la
victoire (part de chaque joueur à la fin de la partie) et, en appoint, les pièces gagnées d'ici la
fin et à qui appartient chaque région au début du prochain tour du joueur (comme la propriété des
cases de KataGo : des dizaines d'étiquettes réelles par position, là où la victoire n'en donne
qu'une par partie). Les positions sont vues du joueur qui agit.

AdamW, taux 5e-4 en un cycle (montée puis descente) à chaque pas de la boucle, lots de 512 ; calcul
en bf16 (`torch.autocast`, poids et optimiseur en 32 bits, comme AlphaZero sur TPU) : 1,3 fois plus
rapide sur les processeurs qui ont AVX-512 BF16, 2,7 fois sur GPU (RX 9070 XT, 22 octobre).
Validation en 32 bits. Sans `torch.compile` : sur c7a.16xlarge, 16 cœurs, bf16, 3 % gagnés pour 9
à 18 s de compilation par cycle (mesure du 21 octobre).
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import ILLEGAL, OFF_MAP, OWNERS, chosen, in_validation, load as load_data
from model import GAIN_SCALE, export, load as load_net, unpack_adj

LR = 5e-4
WEIGHT_DECAY = 1e-4
BATCH = 512
# passages moyens sur chaque nouvelle position
USES = 4
# poids des pièces gagnées et des propriétaires dans la perte (KataGo : 1,5 pour la propriété)
GAIN_WEIGHT = 0.25
OWN_WEIGHT = 1.5
# part des parties gardées pour la validation (`in_validation`)
VAL_SHARE = 0.05
def batches(recs, idx, size, device):
    for start in range(0, len(idx), size):
        sel = np.sort(idx[start : start + size])
        recs_sel = recs[sel]
        # champs copiés à part : un enregistrement n'aligne pas ses nombres sur 4 octets
        r = {name: np.ascontiguousarray(recs_sel[name]) for name in recs_sel.dtype.names}
        policy = torch.from_numpy(r["policy"].astype(np.int32)).to(device)
        legal = policy != ILLEGAL
        policy = torch.where(legal, policy, 0).float()
        yield dict(
            node=torch.from_numpy(r["node"]).to(device),
            glob=torch.from_numpy(r["glob"]).to(device),
            adj=unpack_adj(torch.from_numpy(r["adj"].astype(np.int64)).to(device)),
            regions=torch.from_numpy(r["regions"].astype(np.int64)).to(device),
            players=torch.from_numpy(r["players"].astype(np.int64)).to(device),
            legal=legal,
            policy=policy / policy.sum(1, keepdim=True).clamp(min=1),
            win=torch.from_numpy(r["win"]).to(device),
            gain=torch.from_numpy(r["gain"]).to(device) / GAIN_SCALE,
            owner=torch.from_numpy(r["owner"].astype(np.int64)).to(device),
        )


METRICS = ("politique", "valeur", "pièces", "propriétaires", "précision", "brier")


def losses(net, b):
    """Pertes et mesures (`METRICS`) : politique, valeur, pièces, propriétaires, précision de la
    politique, brier."""
    logits, value, own = net(b["node"], b["glob"], b["adj"], b["regions"])
    logits = logits.masked_fill(~b["legal"], -1e9)
    pol = -(b["policy"] * F.log_softmax(logits, 1)).sum(1).mean()
    present = torch.arange(5, device=value.device)[None] < b["players"][:, None]
    vlog = value[:, :5].masked_fill(~present, -1e9)
    val = -(b["win"] * F.log_softmax(vlog, 1)).sum(1).mean()
    gain = (((value[:, 5:] - b["gain"]) ** 2) * present).sum(1).mean()
    owners = F.cross_entropy(own.reshape(-1, OWNERS), b["owner"].reshape(-1), ignore_index=OFF_MAP)
    # précision : le coup préféré de la recherche est-il le préféré du réseau ?
    acc = (logits.argmax(1) == b["policy"].argmax(1)).float().mean()
    # la victoire prédite, comparée à la cible (erreur quadratique, « Brier »)
    brier = ((torch.softmax(vlog, 1) - b["win"]) ** 2).sum(1).mean()
    return pol, val, gain, owners, acc, brier


def learn(net, opt, sched, b):
    """Un lot d'entraînement (en bf16) ; renvoie ses mesures (`METRICS`)."""
    with torch.autocast(b["node"].device.type, dtype=torch.bfloat16):
        out = losses(net, b)
        pol, val, gain, owners = out[:4]
        loss = pol + val + GAIN_WEIGHT * gain + OWN_WEIGHT * owners
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
    opt.step()
    sched.step()
    return [x.item() for x in out]


def validate(net, recs, idx, device):
    """Les mesures (`METRICS`) sur les positions `idx`, en 32 bits, pondérées par position."""
    net.eval()
    vs, vc = np.zeros(len(METRICS)), 0
    with torch.no_grad():
        for b in batches(recs, idx, 2048, device):
            vs += np.array([x.item() for x in losses(net, b)]) * len(b["players"])
            vc += len(b["players"])
    net.train()
    return vs / max(vc, 1)


def show(values):
    return " ".join(f"{name} {v:.1%}" if name == "précision" else f"{name} {v:.3f}" for name, v in zip(METRICS, values))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", required=True, help="les cycles de la boucle, un sous-dossier chacun")
    ap.add_argument("--last", required=True, help="le cycle qui vient d'être joué (son dossier)")
    ap.add_argument("--init", required=True, help="le réseau qui a joué le dernier cycle (.bin)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, help="cœurs de PyTorch")
    ap.add_argument("--seed", type=int, default=0, help="ordre des lots")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.threads:
        torch.set_num_threads(args.threads)
    start = time.time()
    dirs, new = chosen(args.cycles, args.last)
    recs, keys = load_data(dirs)
    rng = np.random.default_rng(args.seed)
    # validation sur des parties entières, choisies par une règle fixe (`in_validation`) : aucune
    # position d'une partie vue à l'entraînement, ni par ce réseau ni par les suivants
    is_val = in_validation(keys, VAL_SHARE)
    val_idx, train_idx = np.flatnonzero(is_val), np.flatnonzero(~is_val)
    # au moins 20 lots : le taux en un cycle (OneCycleLR) demande une montée d'au moins un lot
    steps = max(20, math.ceil(USES * new / BATCH))
    print(
        f"fenêtre {dirs[0].name}…{dirs[-1].name} : {len(recs)} positions de {len(np.unique(keys))} parties "
        f"({len(val_idx)} positions pour la validation) ; {new} nouvelles, {steps} lots ; "
        f"chargement {time.time() - start:.0f} s",
        flush=True,
    )

    net = load_net(args.init).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)
    # tirage avec remise, en proportion des poids
    weight = recs["weight"][train_idx].astype(np.float64)
    order = rng.choice(train_idx, size=steps * BATCH, p=weight / weight.sum())
    start = time.time()
    net.train()
    sums = np.zeros(len(METRICS))
    for b in batches(recs, order, BATCH, device):
        sums += learn(net, opt, sched, b)
    train_secs = time.time() - start

    start = time.time()
    val = validate(net, recs, val_idx, device)
    print(f"entraînement ({train_secs:.0f} s) : {show(sums / steps)}")
    print(f"validation ({time.time() - start:.0f} s) : {show(val)}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    export(net, out)
    print(f"écrit : {out}")


if __name__ == "__main__":
    main()
