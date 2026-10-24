"""La valeur de plusieurs réseaux sur les mêmes positions : les parties de validation des cycles donnés
(jamais vues à l'apprentissage), par nombre de joueurs, en tout et en début de partie (6 manches
ou plus à jouer). Entropie croisée avec le vainqueur réel (la justesse), entropie des prédictions
(la confiance : plus basse, plus sûre d'elle) et brier. Une valeur qui devient trop sûre d'elle voit
son entropie baisser et son entropie croisée monter.

    python track.py --data ../../lab3/data/deep7/c12{6,7,8,9} ../../lab3/data/deep7/c130 --nets ../runs/w1/c13{0,1}.bin
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from data import in_validation, load
from model import load as load_net
from train import VAL_SHARE, batches

G_LEFT = 6

ap = argparse.ArgumentParser()
ap.add_argument("--data", nargs="+", required=True)
ap.add_argument("--nets", nargs="+", required=True)
args = ap.parse_args()

recs, keys = load(args.data, keep=lambda k: in_validation(k, VAL_SHARE))
print(f"{len(recs)} positions de validation, {len(np.unique(keys))} parties", flush=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
win = recs["win"].astype(np.float64)
players = recs["players"].astype(int)
early = recs["glob"][:, G_LEFT].astype(int) >= 6
groups = [(f"{n} j.", players == n) for n in (2, 3, 4, 5)] + [(f"{n} j. début", (players == n) & early) for n in (3, 4, 5)]
print("réseau            " + "  ".join(f"{name:>22}" for name, _ in groups))
print("                  " + "  ".join(f"{'perte  entropie  brier':>22}" for _ in groups))
for path in args.nets:
    net = load_net(path).to(device).eval()
    preds = []
    with torch.no_grad():
        for b in batches(recs, np.arange(len(recs)), 4096, device):
            _, value, _ = net(b["node"], b["glob"], b["adj"], b["regions"])
            present = torch.arange(5, device=device)[None] < b["players"][:, None]
            preds.append(torch.softmax(value[:, :5].masked_fill(~present, -1e9), 1).cpu().numpy())
    p = np.clip(np.concatenate(preds).astype(np.float64), 1e-12, 1)
    loss = -(win * np.log(p)).sum(1)
    entropy = -(p * np.log(p)).sum(1)
    brier = ((p - win) ** 2).sum(1)
    cells = [f"{loss[m].mean():5.3f}  {entropy[m].mean():8.3f}  {brier[m].mean():5.3f}" for _, m in groups]
    label = f"{Path(path).parent.name}/{Path(path).stem}"
    print(f"{label:16}  " + "  ".join(f"{c:>22}" for c in cells), flush=True)
