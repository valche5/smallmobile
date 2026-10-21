"""Entraîne le réseau sur des parties d'entraînement (`lab2 selfplay`), puis l'exporte pour Rust.

    python train.py --data ../data/gen2 --init ../nets/gen1.bin --out ../nets/gen2.bin

Sans `--init`, un réseau neuf (`model.Net`).

Cibles : la politique (politique améliorée de la recherche à la racine, Gumbel AlphaZero), la
victoire (part de chaque joueur à la fin de la partie) et, en appoint, les pièces gagnées d'ici la
fin. Les positions sont vues du joueur qui agit.

Recette fixe, reprise de lab1 : AdamW, taux 5e-4 en un cycle, 24 000 lots de 512 positions, mais
au plus 4 passages sur les données.
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import ILLEGAL, in_validation, load as load_data
from model import GAIN_SCALE, Net, export, load as load_net, unpack_adj

LR = 5e-4
WEIGHT_DECAY = 1e-4
BATCH = 512
STEPS = 24_000
# au-delà, la valeur apprend les parties par cœur (un seul résultat pour toutes leurs positions)
PASSES = 4
# poids des pièces gagnées dans la perte
GAIN_WEIGHT = 0.25
# part des parties gardées pour la validation (`in_validation`)
VAL_SHARE = 0.05


def batches(recs, idx, size, device):
    for start in range(0, len(idx), size):
        sel = np.sort(idx[start : start + size])
        recs_sel = recs[sel]
        # champs copiés à part : un enregistrement n'aligne pas ses nombres sur 4 octets
        r = {name: np.ascontiguousarray(recs_sel[name]) for name in recs_sel.dtype.names}
        policy = torch.from_numpy(r["policy"].astype(np.int32)).to(device)
        legal = policy != ILLEGAL
        policy = torch.where(legal, policy, 0).float()
        yield dict(
            node=torch.from_numpy(r["node"]).to(device),
            glob=torch.from_numpy(r["glob"]).to(device),
            adj=unpack_adj(torch.from_numpy(r["adj"].astype(np.int64)).to(device)),
            regions=torch.from_numpy(r["regions"].astype(np.int64)).to(device),
            players=torch.from_numpy(r["players"].astype(np.int64)).to(device),
            legal=legal,
            policy=policy / policy.sum(1, keepdim=True).clamp(min=1),
            win=torch.from_numpy(r["win"]).to(device),
            gain=torch.from_numpy(r["gain"]).to(device) / GAIN_SCALE,
        )


def losses(net, b):
    """Pertes et mesures : politique, valeur, pièces, précision de la politique, brier."""
    logits, value = net(b["node"], b["glob"], b["adj"], b["regions"])
    logits = logits.masked_fill(~b["legal"], -1e9)
    pol = -(b["policy"] * F.log_softmax(logits, 1)).sum(1).mean()
    present = torch.arange(5, device=value.device)[None] < b["players"][:, None]
    vlog = value[:, :5].masked_fill(~present, -1e9)
    val = -(b["win"] * F.log_softmax(vlog, 1)).sum(1).mean()
    gain = (((value[:, 5:] - b["gain"]) ** 2) * present).sum(1).mean()
    # précision : le coup préféré de la recherche est-il le préféré du réseau ?
    acc = (logits.argmax(1) == b["policy"].argmax(1)).float().mean()
    # la victoire prédite, comparée à la cible (erreur quadratique, « Brier »)
    brier = ((torch.softmax(vlog, 1) - b["win"]) ** 2).sum(1).mean()
    return pol, val, gain, acc, brier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--init", help="le réseau de départ (.bin) ; sans lui, un réseau neuf")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, help="cœurs de PyTorch (plusieurs entraînements à la fois)")
    ap.add_argument("--seed", type=int, default=0, help="ordre des lots, poids d'un réseau neuf")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.threads:
        torch.set_num_threads(args.threads)
    recs, keys = load_data(args.data)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    # validation sur des parties entières, choisies par une règle fixe (`in_validation`) : aucune
    # position d'une partie vue à l'entraînement, ni par cette génération ni par les suivantes
    is_val = in_validation(keys, VAL_SHARE)
    val_idx, train_idx = np.flatnonzero(is_val), np.flatnonzero(~is_val)
    print(
        f"{len(recs)} positions de {len(np.unique(keys))} parties "
        f"({len(val_idx)} positions de {len(np.unique(keys[is_val]))} parties pour la validation)"
    )

    net = (load_net(args.init) if args.init else Net()).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    per_epoch = math.ceil(len(train_idx) / BATCH)
    steps = min(STEPS, PASSES * per_epoch)
    epochs = math.ceil(steps / per_epoch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)
    done = 0

    for epoch in range(epochs):
        net.train()
        start = time.time()
        sums = np.zeros(5)
        count = 0
        train = rng.permutation(train_idx)
        for b in batches(recs, train, BATCH, device):
            if done == steps:
                break
            done += 1
            pol, val, gain, acc, brier = losses(net, b)
            loss = pol + val + GAIN_WEIGHT * gain
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            sched.step()
            sums += [x.item() for x in (pol, val, gain, acc, brier)]
            count += 1
        net.eval()
        # moyennes pondérées par le nombre de positions de chaque lot
        vs, vc = np.zeros(5), 0
        with torch.no_grad():
            for b in batches(recs, val_idx, 2048, device):
                vs += np.array([x.item() for x in losses(net, b)]) * len(b["players"])
                vc += len(b["players"])
        t, v = sums / max(count, 1), vs / max(vc, 1)
        print(
            f"époque {epoch + 1}/{epochs} ({time.time() - start:.0f} s, {done}/{steps} mises à jour) | entraînement : "
            f"politique {t[0]:.3f} valeur {t[1]:.3f} pièces {t[2]:.3f} précision {t[3]:.1%} | validation : "
            f"politique {v[0]:.3f} valeur {v[1]:.3f} pièces {v[2]:.3f} précision {v[3]:.1%} brier {v[4]:.3f}",
            flush=True,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    export(net, out)
    print(f"écrit : {out}")


if __name__ == "__main__":
    main()
