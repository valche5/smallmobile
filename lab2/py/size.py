"""Quelle taille de réseau ? Une taille entraînée depuis zéro sur des parties de la boucle, avec la
recette de train.py (bf16, positions tirées selon leur poids, taux en un cycle), pour comparer ses
erreurs de validation à celles d'autres tailles sur les mêmes parties.

    python size.py --data ../data/loop/c019 ../data/loop/c020 --size 96x6 --passes 2 --out ../runs/size/96x6.bin

`--size LARGEURxCOUCHES` : un réseau neuf ; ou `--init réseau.bin` pour continuer un réseau déjà
entraîné (sa taille vient du fichier ; ses mesures de départ sont affichées). `--share` : ne garder
qu'une part des positions, tirée au hasard mais la même d'un processus à l'autre (plus de parties
dans la même mémoire). Après chaque passage sur les données : temps par lot et mesures de
validation. Une taille par processus : lab2/runs/size.sh en lance plusieurs en même temps.
"""

import argparse
import math
import time

import numpy as np
import torch

from data import in_validation, load as load_data
from model import Net, export, load as load_net
from train import BATCH, LR, VAL_SHARE, WEIGHT_DECAY, batches, learn, show, validate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    size = ap.add_mutually_exclusive_group(required=True)
    size.add_argument("--size", help="LARGEURxCOUCHES, par exemple 96x6 : un réseau neuf")
    size.add_argument("--init", help="un réseau déjà entraîné (.bin), à continuer")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--share", type=float, default=1.0, help="part des positions gardées")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int)
    args = ap.parse_args()

    torch.manual_seed(0)
    if args.threads:
        torch.set_num_threads(args.threads)
    pick = np.random.default_rng(1)
    recs, keys = load_data(args.data, keep=(lambda k: pick.random(len(k)) < args.share) if args.share < 1 else None)
    is_val = in_validation(keys, VAL_SHARE)
    val_idx, train_idx = np.flatnonzero(is_val), np.flatnonzero(~is_val)
    if args.init:
        net = load_net(args.init)
        args.size = f"{net.hidden}x{len(net.layers)} (suite de {args.init})"
    else:
        net = Net(*(int(x) for x in args.size.split("x")))
    params = sum(p.numel() for p in net.parameters())
    per_pass = math.ceil(len(train_idx) / BATCH)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=args.passes * per_pass, pct_start=0.1)
    weight = recs["weight"][train_idx].astype(np.float64)
    rng = np.random.default_rng(0)
    print(
        f"{args.size} : {params} paramètres ; {len(train_idx)} positions d'entraînement, {len(val_idx)} de validation, "
        f"{len(np.unique(keys))} parties",
        flush=True,
    )
    if args.init:
        print(f"{args.size}, au départ : {show(validate(net, recs, val_idx, 'cpu'))}", flush=True)
    for k in range(args.passes):
        order = rng.choice(train_idx, size=per_pass * BATCH, p=weight / weight.sum())
        start = time.time()
        for b in batches(recs, order, BATCH, "cpu"):
            learn(net, opt, sched, b)
        ms = 1000 * (time.time() - start) / per_pass
        print(f"{args.size}, passage {k + 1}/{args.passes} ({ms:.0f} ms par lot) : {show(validate(net, recs, val_idx, 'cpu'))}",
              flush=True)
    export(net, args.out)


if __name__ == "__main__":
    main()
