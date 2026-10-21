"""Entraîne le réseau sur des parties d'entraînement (`lab1 selfplay`, `lab1 judge`), puis l'exporte pour
Rust.

    python train.py --data ../data/gen12 ../data/gen12-judged --init ../nets/gen11.pt --out ../nets/gen12.bin

Cibles : la politique (visites de la recherche à la racine), la victoire (part de chaque joueur à la
fin) et, en appoint, les pièces gagnées d'ici la fin ; la tête « coup réel », le coup joué ensuite
dans la partie. Les positions sont vues du joueur qui agit. Les positions rejugées (`lab1 judge`, sans
visites) n'apprennent que la valeur ; les coups observés d'autres joueurs que l'IA qui apprend
(tables variées) n'apprennent que la tête « coup réel ».

Recette fixe (`lab1/JOURNAL.md`) : AdamW, taux 5e-4 en un cycle, 24 000 lots de 512 positions.
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import ILLEGAL, OBSERVED, in_validation, load
from model import GAIN_SCALE, Net, export, unpack_adj, with_behaviour

LR = 5e-4
WEIGHT_DECAY = 1e-4
BATCH = 512
STEPS = 24_000
# poids des pièces gagnées et de la tête « coup réel » dans la perte
GAIN_WEIGHT = 0.25
BEH_WEIGHT = 1.0
# part des parties gardées pour la validation (`in_validation`)
VAL_SHARE = 0.05


def batches(recs, idx, size, device):
    for start in range(0, len(idx), size):
        sel = np.sort(idx[start : start + size])
        recs_sel = recs[sel]
        # champs copiés à part : un enregistrement n'aligne pas ses nombres sur 4 octets
        r = {name: np.ascontiguousarray(recs_sel[name]) for name in recs_sel.dtype.names}
        visits = torch.from_numpy(r["visits"].astype(np.int32)).to(device)
        legal = visits != ILLEGAL
        visits = torch.where(legal, visits, 0).float()
        observed = torch.from_numpy((r["flags"] & OBSERVED) != 0).to(device)
        played = torch.from_numpy(r["played"].astype(np.int64)).to(device)
        known = played != ILLEGAL
        played = torch.where(known, played, 0)
        yield dict(
            node=torch.from_numpy(r["node"]).to(device),
            glob=torch.from_numpy(r["glob"]).to(device),
            adj=unpack_adj(torch.from_numpy(r["adj"].astype(np.int64)).to(device)),
            regions=torch.from_numpy(r["regions"].astype(np.int64)).to(device),
            players=torch.from_numpy(r["players"].astype(np.int64)).to(device),
            legal=legal,
            policy=visits / visits.sum(1, keepdim=True).clamp(min=1),
            win=torch.from_numpy(r["win"]).to(device),
            gain=torch.from_numpy(r["gain"]).to(device) / GAIN_SCALE,
            has_value=~observed,
            has_policy=legal.any(1) & (visits.sum(1) > 0) & ~observed,
            played=played,
            has_played=known & legal.gather(1, played[:, None])[:, 0],
        )


def losses(net, b):
    """Pertes et mesures : politique, valeur, pièces, précision de la politique, brier, puis tête
    « coup réel » (perte et précision)."""
    logits, value, beh = net(b["node"], b["glob"], b["adj"], b["regions"])
    logits = logits.masked_fill(~b["legal"], -1e9)
    hp, hv = b["has_policy"].float(), b["has_value"].float()
    pol = (-(b["policy"] * F.log_softmax(logits, 1)).sum(1) * hp).sum() / hp.sum().clamp(min=1)
    present = torch.arange(5, device=value.device)[None] < b["players"][:, None]
    vlog = value[:, :5].masked_fill(~present, -1e9)
    val = (-(b["win"] * F.log_softmax(vlog, 1)).sum(1) * hv).sum() / hv.sum().clamp(min=1)
    gain = ((((value[:, 5:] - b["gain"]) ** 2) * present).sum(1) * hv).sum() / hv.sum().clamp(min=1)
    # précision : le coup le plus visité est-il le préféré du réseau ?
    acc = ((logits.argmax(1) == b["policy"].argmax(1)).float() * hp).sum() / hp.sum().clamp(min=1)
    # la victoire prédite, comparée à la cible (erreur quadratique, « Brier »)
    brier = (((torch.softmax(vlog, 1) - b["win"]) ** 2).sum(1) * hv).sum() / hv.sum().clamp(min=1)
    hb = b["has_played"].float()
    beh = beh.masked_fill(~b["legal"], -1e9)
    bce = (F.cross_entropy(beh, b["played"], reduction="none") * hb).sum() / hb.sum().clamp(min=1)
    bacc = ((beh.argmax(1) == b["played"]).float() * hb).sum() / hb.sum().clamp(min=1)
    return pol, val, gain, acc, brier, bce, bacc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--init", required=True, help="le réseau de départ (.pt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, help="cœurs de PyTorch (plusieurs entraînements à la fois)")
    ap.add_argument("--seed", type=int, default=0, help="ordre des lots")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.threads:
        torch.set_num_threads(args.threads)
    recs, keys = load(args.data)
    rng = np.random.default_rng(args.seed)
    # validation sur des parties entières, choisies par une règle fixe (`in_validation`) : aucune
    # position d'une partie vue à l'entraînement, ni par cette génération ni par les suivantes
    is_val = in_validation(keys, VAL_SHARE)
    val_idx, train_idx = np.flatnonzero(is_val), np.flatnonzero(~is_val)
    print(
        f"{len(recs)} positions de {len(np.unique(keys))} parties "
        f"({len(val_idx)} positions de {len(np.unique(keys[is_val]))} parties pour la validation)"
    )

    state = torch.load(args.init, map_location="cpu")
    net = Net(state["hidden"], state["layers"])
    net.load_state_dict(with_behaviour(state["weights"]))
    net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    per_epoch = math.ceil(len(train_idx) / BATCH)
    epochs = math.ceil(STEPS / per_epoch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS, pct_start=0.1)
    done = 0

    for epoch in range(epochs):
        net.train()
        start = time.time()
        sums = np.zeros(7)
        count = 0
        train = rng.permutation(train_idx)
        for b in batches(recs, train, BATCH, device):
            if done == STEPS:
                break
            done += 1
            pol, val, gain, acc, brier, bce, bacc = losses(net, b)
            loss = pol + val + GAIN_WEIGHT * gain + BEH_WEIGHT * bce
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            sched.step()
            sums += [x.item() for x in (pol, val, gain, acc, brier, bce, bacc)]
            count += 1
        net.eval()
        # moyennes pondérées par le nombre de positions qui ont chaque cible (politique, valeur, coup joué)
        vs = np.zeros(7)
        vc = np.zeros(7)
        with torch.no_grad():
            for b in batches(recs, val_idx, 2048, device):
                n = [b[k].sum().item() for k in ("has_policy", "has_value", "has_played")]
                w = np.array(n)[[0, 1, 1, 0, 1, 2, 2]]
                vs += [x.item() for x in losses(net, b)] * w
                vc += w
        t, v = sums / max(count, 1), vs / np.maximum(vc, 1)
        print(
            f"époque {epoch + 1}/{epochs} ({time.time() - start:.0f} s, {done}/{STEPS} mises à jour) | entraînement : politique {t[0]:.3f} "
            f"valeur {t[1]:.3f} pièces {t[2]:.3f} précision {t[3]:.1%} coup réel {t[5]:.3f} ({t[6]:.1%}) | validation : "
            f"politique {v[0]:.3f} valeur {v[1]:.3f} pièces {v[2]:.3f} précision {v[3]:.1%} brier {v[4]:.3f} "
            f"coup réel {v[5]:.3f} ({v[6]:.1%})",
            flush=True,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    export(net, out)
    torch.save({"hidden": net.hidden, "layers": len(net.layers), "weights": net.state_dict()}, out.with_suffix(".pt"))
    print(f"écrit : {out} et {out.with_suffix('.pt')}")


if __name__ == "__main__":
    main()
