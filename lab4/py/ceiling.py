"""Le plafond de la valeur, depuis la sortie de `lab4 ceiling` : par nombre de joueurs et manches
restantes, l'entropie croisée de la victoire pour une prédiction uniforme, pour le réseau (sa valeur
à la position), et pour la meilleure valeur possible face à ces joueurs : la répartition des
vainqueurs des autres suites de la même position (chaque suite jugée sans elle, a priori de
Dirichlet `ALPHA` ; un peu pessimiste avec peu de suites). « Atteint » : la part de l'écart entre
l'uniforme et le plafond que le réseau a déjà comblée. `--nets` : la valeur d'autres réseaux, lue
sur les mêmes positions (`F.pos`), à la place de celle du réseau qui a joué.

    python ceiling.py ../runs/ceiling/c130.tsv [--nets ../nets/c130.bin autre.bin]
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

ALPHA = 0.5

ap = argparse.ArgumentParser()
ap.add_argument("tsv")
ap.add_argument("--nets", nargs="*", default=[])
args = ap.parse_args()

positions, suites = {}, defaultdict(list)
for line in open(args.tsv):
    f = line.rstrip("\n").split("\t")
    if f[0] == "position":
        i, n, rnd, last, actor = int(f[1]), int(f[2]), int(f[3]), int(f[4]), int(f[5])
        positions[i] = (n, last - rnd + 1, actor, np.array([float(x) for x in f[6 : 6 + n]]))
    elif f[0] == "suite":
        suites[int(f[1])].append([float(x) for x in f[2:]])


def predictions(path):
    """La victoire que prédit le réseau `path` pour chaque joueur réel, à chaque position."""
    import torch

    from data import RECORD
    from model import load as load_net
    from train import batches

    recs = np.fromfile(args.tsv + ".pos", dtype=RECORD)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = load_net(path).to(device).eval()
    out = []
    with torch.no_grad():
        for b in batches(recs, np.arange(len(recs)), 1024, device):
            _, value = net(b["node"], b["glob"], b["adj"], b["regions"])
            present = torch.arange(5, device=device)[None] < b["players"][:, None]
            out.append(torch.softmax(value[:, :5].masked_fill(~present, -1e9), 1).cpu().numpy())
    rel = np.concatenate(out)
    preds = {}
    for i, (n, _, actor, _) in positions.items():
        preds[i] = np.array([rel[i][(q - actor) % n] for q in range(n)])
    return preds


def table(label, preds):
    groups = defaultdict(lambda: np.zeros(4))
    for i, (n, left, _, _) in positions.items():
        wins = np.array(suites[i])
        k, total = len(wins), wins.sum(0)
        stage = "6+" if left >= 6 else "4–5" if left >= 4 else "2–3" if left >= 2 else "1"
        for w in wins:
            rest = (total - w + ALPHA) / (k - 1 + n * ALPHA)
            ceiling = -(w * np.log(rest)).sum()
            model = -(w * np.log(np.clip(preds[i], 1e-6, 1))).sum()
            for key in ((n, stage), (n, "tout")):
                groups[key] += (1, np.log(n), model, ceiling)
    print(f"\n{label}")
    print("joueurs  manches restantes  suites  uniforme  réseau  plafond  atteint")
    for n in sorted({n for n, _ in groups}):
        for stage in ("6+", "4–5", "2–3", "1", "tout"):
            if (n, stage) not in groups:
                continue
            c, uni, model, ceiling = groups[(n, stage)]
            uni, model, ceiling = uni / c, model / c, ceiling / c
            reached = (uni - model) / (uni - ceiling) if uni > ceiling else float("nan")
            print(f"{n:>7}  {stage:>17}  {int(c):>6}  {uni:8.3f}  {model:6.3f}  {ceiling:7.3f}  {reached:7.0%}")


if args.nets:
    for path in args.nets:
        table(f"{Path(path).parent.name}/{Path(path).stem}", predictions(path))
else:
    table("le réseau qui a joué", {i: p[3] for i, p in positions.items()})
