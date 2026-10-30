"""Apprentissage hors ligne d'un réseau sur des parties déjà jouées, mesuré sur des parties neuves
(lab13/lab.md, D3).

    python fit.py --train ../../lab11/data/p1/c004 … --test ../data/prof ../data/ord \
        --init ../nets/c130.bin --lr 5e-5 --epochs 4 --out ../runs/d3-lr5e-5.bin

Positions de `--train` tirées avec remise en proportion de leur poids, comme `train.py`, pendant
`--epochs` passages moyens ; taux maximal `--lr` en un cycle (montée sur 5 % des lots). Tous les
`--every` lots, et au départ : les mesures de `agree.measures` sur chaque dossier de `--test` (toutes
leurs positions, en 32 bits). Le réseau de la meilleure perte de politique sur le premier `--test` est
écrit dans `--out`.
"""

import argparse
import time

import numpy as np
import torch

from agree import measures
from data import load_packed
from model import export, load as load_net
from train import BATCH, METRICS, WEIGHT_DECAY, batches, learn, show


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True, help="dossiers de cycles (`selfplay --out`)")
    ap.add_argument("--test", nargs="+", required=True, help="dossiers de parties neuves, le juge d'abord")
    ap.add_argument("--init", required=True, help="le réseau de départ (.bin)")
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--epochs", type=float, required=True, help="passages moyens sur les positions")
    ap.add_argument("--every", type=int, default=2000, help="lots entre deux mesures")
    ap.add_argument("--value-mix", type=float, default=0.0, help="comme train.py")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    start = time.time()
    recs, keys = load_packed(args.train)
    tests = [(path, load_packed([path])[0]) for path in args.test]
    steps = int(args.epochs * len(recs) / BATCH)
    print(f"{len(recs)} positions de {len(np.unique(keys))} parties ; {steps} lots ; chargement "
          f"{time.time() - start:.0f} s ; {device}", flush=True)

    net = load_net(args.init).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    weight = recs["weight"].astype(np.float64)
    order = np.random.default_rng(args.seed).choice(len(recs), size=steps * BATCH, p=weight / weight.sum())

    best = None

    def judge(done, sums, count):
        nonlocal best
        line = f"lot {done}"
        if count:
            line += f" (apprentissage : {show(sums / count)})"
        for i, (path, t) in enumerate(tests):
            m = measures(net, t, np.arange(len(t)), device)
            line += f"\n    {path} : {show(m)}"
            if i == 0 and (best is None or m[0] < best):
                best = m[0]
                export(net, args.out)
                line += " ← meilleur"
        net.train()
        print(line, flush=True)

    judge(0, None, 0)
    start = time.time()
    sums, count = np.zeros(len(METRICS)), 0
    for i, b in enumerate(batches(recs, order, BATCH, device)):
        sums += learn(net, opt, sched, b, args.value_mix)
        count += 1
        if (i + 1) % args.every == 0 or i + 1 == steps:
            judge(i + 1, sums, count)
            sums, count = np.zeros(len(METRICS)), 0
    print(f"fini en {time.time() - start:.0f} s ; meilleure politique sur {args.test[0]} : {best:.3f}")


if __name__ == "__main__":
    main()
