"""Les têtes construites du choix du peuple et du déclin, apprises hors ligne et mesurées par type
(lab13/lab.md, D13).

    python d13.py --name h1 --train ../data/tgt-train --test ../data/anc-test --init ../nets/c130.bin \
        --head-lr 1e-3 --trunk-lr 0

`--plain` : sans les têtes (le témoin). Apprentissage comme `fit.py` (positions tirées selon leur poids,
4 passages, un cycle) ; taux `--trunk-lr` pour le réseau de base (0 : figé), `--head-lr` pour les
têtes. Puis, par type de décision (`d9.kinds`), l'écart de perte de politique au réseau de départ sur
`--test`.
"""

import argparse
import time

import numpy as np
import torch

from d9 import TYPES, per_position
from data import load_packed
from heads import Heads, export_heads
from model import load as load_net
from train import BATCH, WEIGHT_DECAY, batches, learn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--plain", action="store_true")
    ap.add_argument("--trunk-lr", type=float, default=0.0)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--value-mix", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-decline", action="store_true", help="la tête du peuple seule")
    ap.add_argument("--out", help="le réseau appris, tête du peuple comprise (`heads.export_heads`)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    recs, _ = load_packed(args.train)
    test, _ = load_packed([args.test])
    tidx = np.arange(len(test))
    net = Heads(load_net(args.init), plain=args.plain, decline=not args.no_decline).to(device)
    ref = per_position(net, test, tidx, device)

    groups = []
    if args.trunk_lr > 0:
        groups.append({"params": list(net.base.parameters()), "lr": args.trunk_lr})
    else:
        for p in net.base.parameters():
            p.requires_grad_(False)
    if not args.plain:
        groups.append({"params": net.head_parameters(), "lr": args.head_lr})
    steps = int(args.epochs * len(recs) / BATCH)
    opt = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=steps, pct_start=0.05)
    weight = recs["weight"].astype(np.float64)
    order = np.random.default_rng(args.seed).choice(len(recs), size=steps * BATCH, p=weight / weight.sum())
    start = time.time()
    net.train()
    sums, count = 0.0, 0
    for i, b in enumerate(batches(recs, order, BATCH, device)):
        sums += learn(net, opt, sched, b, args.value_mix)[0]
        count += 1
        if (i + 1) % 2000 == 0 or i + 1 == steps:
            print(f"lot {i + 1}/{steps} : politique (apprentissage) {sums / count:.3f}", flush=True)
            sums, count = 0.0, 0
    print(f"{args.name} : appris en {time.time() - start:.0f} s ({steps} lots)")
    if args.out:
        export_heads(net, args.out)

    ce0, ent, acc0, _, kind = ref
    ce, _, acc, _, _ = per_position(net, test, tidx, device)
    print(f"\n| Type | Positions | Perte au départ | Écart ({args.name}) | Coup de la cible (départ → fin) |")
    print("| --- | --- | --- | --- | --- |")
    for k, name in [(k, n) for k, n in enumerate(TYPES)] + [(None, "tous")]:
        m = np.ones(len(tidx), bool) if k is None else kind == k
        d = ce[m] - ce0[m]
        print(f"| {name} | {m.mean():.1%} | {ce0[m].mean():.3f} | {d.mean():+.3f} ± {d.std() / np.sqrt(m.sum()):.3f} | "
              f"{acc0[m].mean():.1%} → {acc[m].mean():.1%} |")


if __name__ == "__main__":
    main()
