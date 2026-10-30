"""Ce que des réseaux ont retenu d'un professeur : leurs mesures sur des parties neuves (lab14/lab.md, D2).

    python agree.py --data ../data/prof --nets ../../lab11/runs/p1/c000.bin ../../lab11/runs/p1/c010.bin

Sur toutes les positions des parties de `--data` (aucune n'a servi à l'apprentissage) : les mesures de
`train.validate` (politique, valeur, pièces, propriétaires, précision, brier), puis l'accord avec le
coup préféré de la cible (`policy` : la politique améliorée de la recherche qui a joué ces parties).
Le premier réseau de `--nets` est la référence (c130) : l'accord est aussi mesuré séparément là où
son coup préféré diffère de celui de la cible (« écarts » : ce que la recherche ajoute à son réseau) et
là où il est le même (« accords » : ce qu'un élève doit garder).
"""

import argparse

import numpy as np
import torch

from data import load_packed
from model import load as load_net
from train import METRICS, batches, losses, show


def preferred(net, recs, idx, device):
    """Le coup préféré de `net` et celui de la cible, à chaque position de `idx`."""
    net.eval()
    mine, target = [], []
    with torch.no_grad():
        for b in batches(recs, idx, 2048, device):
            logits = net(b["node"], b["glob"], b["adj"], b["regions"])[0].masked_fill(~b["legal"], -1e9)
            mine.append(logits.argmax(1).cpu().numpy())
            target.append(b["policy"].argmax(1).cpu().numpy())
    return np.concatenate(mine), np.concatenate(target)


def measures(net, recs, idx, device):
    """`train.validate` sur toutes les positions de `idx`, en 32 bits."""
    net.eval()
    sums, count = np.zeros(len(METRICS)), 0
    with torch.no_grad():
        for b in batches(recs, idx, 2048, device):
            sums += np.array([x.item() for x in losses(net, b)]) * len(b["players"])
            count += len(b["players"])
    return sums / max(count, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="dossiers de parties (`selfplay --out`)")
    ap.add_argument("--nets", nargs="+", required=True, help="réseaux (.bin), la référence d'abord")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    recs, keys = load_packed(args.data)
    idx = np.arange(len(recs))
    print(f"{len(recs)} positions de {len(np.unique(keys))} parties ; {device}")
    ref = None
    for path in args.nets:
        net = load_net(path).to(device)
        mine, target = preferred(net, recs, idx, device)
        if ref is None:
            ref = mine
            apart = ref != target
            print(f"la cible diffère de la référence à {apart.mean():.1%} des positions ({apart.sum()})")
        agree = mine == target
        print(
            f"{path} : {show(measures(net, recs, idx, device))} ; coup de la cible {agree.mean():.1%}, "
            f"aux écarts {agree[apart].mean():.1%}, aux accords {agree[~apart].mean():.1%} ; "
            f"coup de la référence {(mine == ref).mean():.1%}",
            flush=True,
        )


if __name__ == "__main__":
    main()
