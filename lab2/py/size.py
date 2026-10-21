"""Quelle taille de réseau ? Une taille entraînée depuis zéro sur des parties de la boucle, avec la
recette de train.py (bf16, positions tirées selon leur poids, taux en un cycle), pour comparer ses
erreurs de validation à celles d'autres tailles sur les mêmes parties.

    python size.py --data ../data/loop/c019 ../data/loop/c020 --size 96x6 --passes 2 --out ../runs/size/96x6.bin

`--size LARGEURxCOUCHES`. Après chaque passage sur les données : temps par lot et mesures de
validation. Une taille par processus : lab2/runs/size.sh en lance plusieurs en même temps.
"""

import argparse
import math
import time

import numpy as np
import torch

from data import in_validation, load as load_data
from model import Net, export
from train import BATCH, LR, VAL_SHARE, WEIGHT_DECAY, batches, learn, show, validate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--size", required=True, help="LARGEURxCOUCHES, par exemple 96x6")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int)
    args = ap.parse_args()

    hidden, layers = (int(x) for x in args.size.split("x"))
    torch.manual_seed(0)
    if args.threads:
        torch.set_num_threads(args.threads)
    recs, keys = load_data(args.data)
    is_val = in_validation(keys, VAL_SHARE)
    val_idx, train_idx = np.flatnonzero(is_val), np.flatnonzero(~is_val)
    net = Net(hidden, layers)
    params = sum(p.numel() for p in net.parameters())
    per_pass = math.ceil(len(train_idx) / BATCH)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=args.passes * per_pass, pct_start=0.1)
    weight = recs["weight"][train_idx].astype(np.float64)
    rng = np.random.default_rng(0)
    print(f"{args.size} : {params} paramètres ; {len(train_idx)} positions d'entraînement, {len(val_idx)} de validation")
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
