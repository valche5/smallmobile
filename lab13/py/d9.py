"""Les pertes de politique par type de décision (lab13/lab.md, D9).

    python d9.py --test ../data/anc-test --train ../data/anc-train --nets ../nets/c130.bin ../runs/d9-lr1.5e-4-last.bin …

Sur toutes les positions de `--test` et `--sample` positions de `--train` tirées au hasard : par type
de décision (`kinds`), la part des positions et de la perte de politique du premier réseau, la perte de
politique de chaque réseau, l'entropie de la cible et le coup de la cible.
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from data import load_packed
from model import load as load_net
from train import batches

TYPES = ("peuple", "déclin", "conquête", "pouvoirs", "jetons", "défense")
# lab13/brain/src/moves.rs : clés des coups, sortes des régions ; encode.rs : G_PHASE (Pick, Conquest,
# Redeploy, Defend, GameOver)
PICKS = range(0, 7)
DECLINE = 7
REGION, SPAN = 32, 64
POWERS = (5, 6, 7)  # Hero, Fortress, Camp
G_PHASE = 12


def kinds(legal, glob):
    """Le type de chaque position (indice de `TYPES`), d'après ses coups possibles et sa phase."""
    phase = glob[:, G_PHASE : G_PHASE + 5].argmax(1)
    region = legal[:, REGION:].reshape(len(legal), -1, SPAN).any(2)
    out = np.full(len(legal), 2)
    out[phase == 2] = 4
    out[(phase == 2) & region[:, list(POWERS)].any(1)] = 3
    out[phase == 3] = 5
    out[legal[:, DECLINE]] = 1
    out[legal[:, list(PICKS)].any(1)] = 0
    return out


def per_position(net, recs, idx, device):
    """Perte de politique, entropie de la cible, coup de la cible, coup du réseau et type, par position."""
    net.eval()
    cols = [[] for _ in range(5)]
    with torch.no_grad():
        for b in batches(recs, idx, 2048, device):
            logits = net(b["node"], b["glob"], b["adj"], b["regions"])[0].masked_fill(~b["legal"], -1e9)
            t = b["policy"]
            ce = -(t * F.log_softmax(logits, 1)).sum(1)
            ent = -(t * torch.log(t.clamp(min=1e-12))).sum(1)
            kind = kinds(b["legal"].cpu().numpy(), b["glob"].cpu().numpy())
            for c, x in zip(cols, (ce, ent, (logits.argmax(1) == t.argmax(1)).float(), logits.argmax(1), kind)):
                c.append(x.cpu().numpy() if torch.is_tensor(x) else x)
    return [np.concatenate(c) for c in cols]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--sample", type=int, default=40_000)
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--nets", nargs="+", required=True, help="réseaux (.bin), la référence (c130) d'abord")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for label, path, sample in (("neuves", args.test, None), ("apprises", args.train, args.sample)):
        recs, _ = load_packed([path])
        idx = np.arange(len(recs))
        if sample:
            idx = np.sort(np.random.default_rng(args.seed).choice(len(recs), sample, replace=False))
        runs = [per_position(load_net(p).to(device), recs, idx, device) for p in args.nets]
        ce0, ent, _, ref, kind = runs[0]
        excess = ce0 - ent
        print(f"\n## {label} ({path}, {len(idx)} positions)\n")
        head = "| Type | Positions | Perte de c130 | Excès de c130 | Entropie | " + " | ".join(
            f"Perte {p.split('/')[-1]}" for p in args.nets[1:]) + " | Coup de la cible (" + ", ".join(
            p.split("/")[-1] for p in args.nets) + ") |"
        print(head)
        print("|" + " --- |" * (head.count("|") - 1))
        for k, name in [(k, n) for k, n in enumerate(TYPES)] + [(None, "tous")]:
            m = np.ones(len(idx), bool) if k is None else kind == k
            if not m.any():
                continue
            se = lambda x: x[m].std() / np.sqrt(m.sum())
            cells = [
                name,
                f"{m.mean():.1%}",
                f"{ce0[m].mean():.3f} ({ce0[m].sum() / ce0.sum():.0%})",
                f"{excess[m].mean():.3f} ({excess[m].sum() / excess.sum():.0%})",
                f"{ent[m].mean():.3f}",
            ]
            cells += [f"{r[0][m].mean():.3f} ({r[0][m].mean() - ce0[m].mean():+.3f} ± {se(r[0] - ce0):.3f})" for r in runs[1:]]
            cells.append(", ".join(f"{r[2][m].mean():.1%}" for r in runs))
            print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
