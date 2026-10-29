"""Entraîne le réseau sur les parties de la boucle (`lab11/loop.sh`), puis l'exporte pour Rust.

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

AdamW, taux 5e-4 (`--lr`) en un cycle (montée puis descente) à chaque pas de la boucle, lots de 512 ; calcul
en bf16 (`torch.autocast`, poids et optimiseur en 32 bits, comme AlphaZero sur TPU) : 1,3 fois plus
rapide sur les processeurs qui ont AVX-512 BF16, 2,7 fois sur GPU (RX 9070 XT, 22 octobre).
Validation en 32 bits. Sans `torch.compile` : sur c7a.16xlarge, 16 cœurs, bf16, 3 % gagnés pour 9
à 18 s de compilation par cycle (mesure du 21 octobre).

En continu (`--state-out`, piste du plateau, lab3/journal.md) : l'optimiseur, le réseau qui apprend et
une moyenne de ses poids passent d'un pas à l'autre (`--state-in`), le taux reste constant (montée
de `WARMUP` lots au premier pas seulement), et le réseau exporté, qui joue, est la moyenne
(exponentielle, `EMA` par lot, tous les `EMA_EVERY` lots), comme les réseaux du selfplay de KataGo. Sans cela, chaque pas
repart d'un AdamW neuf et remonte au taux maximal, ce qui écarte un réseau mûr du point où le pas
précédent l'avait posé.
"""

import argparse
import copy
import math
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import ILLEGAL, OFF_MAP, OWNERS, Packed, chosen, in_validation, load_packed
from model import GAIN_SCALE, export, load as load_net, unpack_adj

LR = 5e-4
WEIGHT_DECAY = 1e-4
BATCH = 512
# passages moyens sur chaque nouvelle position
USES = 4
# poids des pièces gagnées et des propriétaires dans la perte (KataGo : 1,5 pour la propriété)
GAIN_WEIGHT = 0.25
OWN_WEIGHT = 1.5
# part des parties gardées pour la validation (`in_validation`), et de celles qui servent à la
# mesurer : un sous-ensemble fixe des premières (mêmes parties d'un pas à l'autre, ~40 000 positions
# pour une fenêtre de 6,8 millions ; la validation entière prenait jusqu'à 45 s)
VAL_SHARE = 0.05
VAL_MEASURE = 0.006
# lots préparés d'avance (`batches`), par `WORKERS` fils : quand la fenêtre dépasse le cache du
# système (24 millions de positions, ~27 Go, le 24 octobre), un lot attend ses pages du disque ;
# plusieurs fils en demandent plusieurs à la fois. Lot de 512 sur la fenêtre de c110 : 22,8 ms avec
# un fil, 8,7 avec 4, 9,2 avec 8, 23 avec 16 (le GPU en calcule un en ~12 ms)
WORKERS = 4
AHEAD = 8
# en continu : moyenne des poids sur ~5 000 lots (les trois quarts d'un pas de 25 000 parties),
# mise à jour tous les `EMA_EVERY` lots (à chaque lot, elle prenait 36 s des 120 d'un pas sur le
# GPU), et montée du taux au premier pas
EMA = 0.9998
EMA_EVERY = 8
WARMUP = 500
def fields(recs, sel):
    """Les champs pleins des positions `sel` : d'une fenêtre compacte (`Packed`), ou d'un tableau
    d'enregistrements."""
    if isinstance(recs, Packed):
        return recs.take(sel)
    recs_sel = recs[sel]
    # Copie obligatoire : le pas d'un enregistrement (6210 octets) n'aligne pas les f32.
    # Avec une seule position, NumPy juge la vue contiguë et ascontiguousarray la garde,
    # mais PyTorch refuse encore son pas de 6210 octets. copy remet aussi ce pas à plat.
    return {name: recs_sel[name].copy(order="C") for name in recs_sel.dtype.names}


def batches(recs, idx, size, device):
    """Les lots des positions `idx`, de `size` positions, sur `device` ; chacun préparé d'avance
    dans d'autres fils (`AHEAD` lots, `WORKERS` fils), pendant que le GPU calcule les précédents."""
    starts = range(0, len(idx), size)
    with ThreadPoolExecutor(WORKERS) as pool:
        ahead = deque()
        for start in starts:
            ahead.append(pool.submit(fields, recs, np.sort(idx[start : start + size])))
            if len(ahead) <= AHEAD:
                continue
            yield tensors(ahead.popleft().result(), device)
        while ahead:
            yield tensors(ahead.popleft().result(), device)


def tensors(r, device):
    """Un lot (`fields`) en tenseurs, cibles normalisées."""
    policy = torch.from_numpy(r["policy"].astype(np.int32)).to(device)
    legal = policy != ILLEGAL
    policy = torch.where(legal, policy, 0).float()
    return dict(
        node=torch.from_numpy(r["node"]).to(device),
        glob=torch.from_numpy(r["glob"]).to(device),
        adj=unpack_adj(torch.from_numpy(r["adj"].astype(np.int64)).to(device)),
        regions=torch.from_numpy(r["regions"].astype(np.int64)).to(device),
        players=torch.from_numpy(r["players"].astype(np.int64)).to(device),
        legal=legal,
        policy=policy / policy.sum(1, keepdim=True).clamp(min=1),
        win=torch.from_numpy(r["win"]).to(device),
        # la valeur de la recherche (`data.VALUES`) ; le résultat sans elle (enregistrements bruts)
        value=torch.from_numpy(r["value"] if "value" in r else r["win"]).to(device),
        gain=torch.from_numpy(r["gain"]).to(device) / GAIN_SCALE,
        owner=torch.from_numpy(r["owner"].astype(np.int64)).to(device),
        # la part du hasard dans `win` (`selfplay --luck`) ; 0 sans elle
        luck=torch.from_numpy(r["luck"] if "luck" in r else np.zeros_like(r["win"])).to(device),
    )


METRICS = ("politique", "valeur", "pièces", "propriétaires", "précision", "brier")


def losses(net, b, mix=0.0, luck=False):
    """Pertes et mesures (`METRICS`) : politique, valeur, pièces, propriétaires, précision de la
    politique, brier. La cible de la valeur : le résultat de la partie, mêlé pour une part `mix` à la
    valeur de la recherche (`--value-mix`), ou (`--luck`) sans la part du hasard (lab11/src/luck.rs :
    même espérance, de somme 1, parfois hors de [0, 1] ; le gradient de l'entropie croisée, p − cible,
    reste celui d'une cible souple) ; le brier compare toujours au résultat."""
    logits, value, own = net(b["node"], b["glob"], b["adj"], b["regions"])
    logits = logits.masked_fill(~b["legal"], -1e9)
    pol = -(b["policy"] * F.log_softmax(logits, 1)).sum(1).mean()
    present = torch.arange(5, device=value.device)[None] < b["players"][:, None]
    vlog = value[:, :5].masked_fill(~present, -1e9)
    target = b["win"] if mix == 0 else (1 - mix) * b["win"] + mix * b["value"]
    if luck:
        target = target - b["luck"]
    val = -(target * F.log_softmax(vlog, 1)).sum(1).mean()
    gain = (((value[:, 5:] - b["gain"]) ** 2) * present).sum(1).mean()
    owners = F.cross_entropy(own.reshape(-1, OWNERS), b["owner"].reshape(-1), ignore_index=OFF_MAP)
    # précision : le coup préféré de la recherche est-il le préféré du réseau ?
    acc = (logits.argmax(1) == b["policy"].argmax(1)).float().mean()
    # la victoire prédite, comparée à la cible (erreur quadratique, « Brier »)
    brier = ((torch.softmax(vlog, 1) - b["win"]) ** 2).sum(1).mean()
    return pol, val, gain, owners, acc, brier


def learn(net, opt, sched, b, mix=0.0, luck=False):
    """Un lot d'entraînement (en bf16) ; renvoie ses mesures (`METRICS`)."""
    with torch.autocast(b["node"].device.type, dtype=torch.bfloat16):
        out = losses(net, b, mix, luck)
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
    ap.add_argument("--lr", type=float, default=LR, help="taux maximal du cycle (le taux constant en continu)")
    ap.add_argument("--value-mix", type=float, default=0.0,
                    help="part de la valeur de la recherche dans la cible de la valeur (0 : le résultat seul)")
    ap.add_argument("--luck", action="store_true",
                    help="cible de la valeur sans la part du hasard (selfplay --luck)")
    ap.add_argument("--state-in", help="en continu : l'état du pas précédent, s'il existe")
    ap.add_argument("--state-out", help="en continu : l'état écrit pour le pas suivant")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.threads:
        torch.set_num_threads(args.threads)
    start = time.time()
    dirs, new = chosen(args.cycles, args.last)
    recs, keys = load_packed(dirs)
    rng = np.random.default_rng(args.seed)
    # validation sur des parties entières, choisies par une règle fixe (`in_validation`) : aucune
    # position d'une partie vue à l'entraînement, ni par ce réseau ni par les suivants
    is_val = in_validation(keys, VAL_SHARE)
    val_idx, train_idx = np.flatnonzero(in_validation(keys, VAL_MEASURE)), np.flatnonzero(~is_val)
    # au moins 20 lots : le taux en un cycle (OneCycleLR) demande une montée d'au moins un lot
    steps = max(20, math.ceil(USES * new / BATCH))
    print(
        f"fenêtre {dirs[0].name}…{dirs[-1].name} : {len(recs)} positions de {len(np.unique(keys))} parties "
        f"({len(val_idx)} positions pour la validation) ; {new} nouvelles, {steps} lots ; "
        f"chargement {time.time() - start:.0f} s ; {device}",
        flush=True,
    )

    net = load_net(args.init).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    ema = None
    if args.state_out:
        state = None
        if args.state_in and Path(args.state_in).exists():
            state = torch.load(args.state_in, map_location=device)
            net.load_state_dict(state["net"])
            opt.load_state_dict(state["opt"])
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] = args.lr
        ema = state["ema"] if state else [p.detach().clone() for p in net.parameters()]
        print(f"en continu : {'état repris' if state else 'premier pas'}, taux {args.lr:g}", flush=True)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 1.0 if state else min(1.0, (i + 1) / WARMUP))
    else:
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    # tirage avec remise, en proportion des poids
    weight = recs["weight"][train_idx].astype(np.float64)
    order = rng.choice(train_idx, size=steps * BATCH, p=weight / weight.sum())
    start = time.time()
    net.train()
    sums = np.zeros(len(METRICS))
    params = list(net.parameters())
    for i, b in enumerate(batches(recs, order, BATCH, device)):
        sums += learn(net, opt, sched, b, args.value_mix, args.luck)
        if ema and i % EMA_EVERY == EMA_EVERY - 1:
            with torch.no_grad():
                torch._foreach_lerp_(ema, params, 1 - EMA**EMA_EVERY)
    train_secs = time.time() - start

    played = net
    if ema:
        tmp = Path(args.state_out + ".tmp")
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "ema": ema}, tmp)
        os.replace(tmp, args.state_out)
        played = copy.deepcopy(net)
        with torch.no_grad():
            for p, e in zip(played.parameters(), ema):
                p.copy_(e)
    start = time.time()
    val = validate(played, recs, val_idx, device)
    print(f"entraînement ({train_secs:.0f} s) : {show(sums / steps)}")
    print(f"validation ({time.time() - start:.0f} s) : {show(val)}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    export(played, out)
    print(f"écrit : {out}")


if __name__ == "__main__":
    main()
