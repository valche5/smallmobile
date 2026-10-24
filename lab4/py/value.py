"""Où les réseaux se trompent : leurs erreurs sur les parties de validation, par genre de position.

    python value.py --data ../data/gen2 --nets ../nets/gen1.bin ../nets/gen2.bin

Genres : phase (choix d'un peuple, conquête, redéploiement avec ou sans jetons en main, défense),
à 2 joueurs ou à 3 à 5. Pour chaque réseau : perte de la valeur (celle de train.py), brier, biais
(victoire prédite moins victoire réelle du joueur qui agit), perte et précision de la politique.

Puis la cohérence de la valeur pendant un tour : entre deux positions enregistrées de suite dans le
même tour, sans hasard entre elles (hors dé de la dernière conquête), une valeur juste ne bouge
presque pas. Variation moyenne de la victoire prédite du joueur, par passage.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import in_validation, load as load_data
from model import load as load_net
from train import VAL_SHARE, batches

# entrées de la partie (lab4/brain/src/encode.rs)
G_ROUND = 4
G_PHASE = 12
G_HAND = 28
G_COINS = [67 + q * 81 + 1 for q in range(5)]
PICK, CONQUEST, REDEPLOY, DEFEND = range(4)

KINDS = ["choix d'un peuple", "conquête", "redéploiement, jetons en main", "redéploiement, main vide", "défense"]
PASSAGES = {
    "conquête → conquête": (CONQUEST, CONQUEST),
    "conquête → redéploiement": (CONQUEST, REDEPLOY),
    "redéploiement → redéploiement": (REDEPLOY, REDEPLOY),
}


def kinds(recs):
    """Le genre de chaque position (indice dans KINDS)."""
    g = recs["glob"]
    phase = g[:, G_PHASE : G_PHASE + 5].argmax(1)
    kind = np.select([phase == PICK, phase == CONQUEST, phase == REDEPLOY], [0, 1, 2], 4)
    return np.where((phase == REDEPLOY) & (g[:, G_HAND] == 0), 3, kind), phase


@torch.no_grad()
def evaluate(net, recs, device):
    """Par position, dans l'ordre : perte de la valeur, brier, victoire prédite et réelle du joueur qui
    agit, perte et justesse de la politique."""
    out = {k: [] for k in ("val", "brier", "pred", "real", "pol", "acc")}
    for b in batches(recs, np.arange(len(recs)), 2048, device):
        logits, value, _ = net(b["node"], b["glob"], b["adj"], b["regions"])
        logits = logits.masked_fill(~b["legal"], -1e9)
        present = torch.arange(5, device=value.device)[None] < b["players"][:, None]
        vlog = value[:, :5].masked_fill(~present, -1e9)
        p = torch.softmax(vlog, 1)
        out["val"].append(-(b["win"] * F.log_softmax(vlog, 1)).sum(1))
        out["brier"].append(((p - b["win"]) ** 2).sum(1))
        out["pred"].append(p[:, 0])
        out["real"].append(b["win"][:, 0])
        out["pol"].append(-(b["policy"] * F.log_softmax(logits, 1)).sum(1))
        out["acc"].append((logits.argmax(1) == b["policy"].argmax(1)).float())
    return {k: torch.cat(v).cpu().numpy() for k, v in out.items()}


def same_turn(recs, keys):
    """Pour chaque position sauf la dernière : la suivante est-elle du même tour (même partie, même
    manche, mêmes pièces, vues du même joueur) ?"""
    g = recs["glob"]
    coins = g[:, G_COINS]
    return (keys[1:] == keys[:-1]) & (g[1:, G_ROUND] == g[:-1, G_ROUND]) & (coins[1:] == coins[:-1]).all(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--nets", nargs="+", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    recs, keys = load_data(args.data, keep=lambda k: in_validation(k, VAL_SHARE))
    print(f"{len(recs)} positions de validation, {len(np.unique(keys))} parties")
    kind, phase = kinds(recs)
    two = recs["players"] == 2
    results = {Path(p).stem: evaluate(load_net(p).to(device).eval(), recs, device) for p in args.nets}

    print("\nErreurs par genre de position (valeur : perte, brier, biais ; politique : perte, précision)")
    groups = [("tout", np.ones(len(recs), bool))]
    for i, name in enumerate(KINDS):
        groups.append((name, kind == i))
        groups += [("  à 2 joueurs", (kind == i) & two), ("  à 3 à 5", (kind == i) & ~two)]
    real = next(iter(results.values()))["real"]
    for name, sel in groups:
        if not sel.any():
            continue
        print(f"{name:32} {sel.sum():7}  réel {real[sel].mean():.3f}")
        for net, r in results.items():
            print(
                f"    {net:10} valeur {r['val'][sel].mean():.3f} brier {r['brier'][sel].mean():.3f} "
                f"biais {(r['pred'] - r['real'])[sel].mean():+.3f} | politique {r['pol'][sel].mean():.3f} "
                f"précision {r['acc'][sel].mean():.1%}"
            )

    print("\nVictoire prédite du joueur, d'une position à la suivante du même tour (moyenne, moyenne absolue)")
    linked = same_turn(recs, keys)
    for name, (a, b) in PASSAGES.items():
        sel = linked & (phase[:-1] == a) & (phase[1:] == b)
        if not sel.any():
            continue
        print(f"{name:32} {sel.sum():7}")
        for net, r in results.items():
            d = (r["pred"][1:] - r["pred"][:-1])[sel]
            print(f"    {net:10} {d.mean():+.4f}  |{np.abs(d).mean():.4f}|")


if __name__ == "__main__":
    main()
