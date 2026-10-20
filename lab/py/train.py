"""Entraîne le réseau sur des parties d'entraînement (`lab selfplay`), puis l'exporte pour Rust.

    python train.py --data ../data/gen1 ../data/normal@0.2 --out ../nets/gen2.bin [--init ../nets/gen1.pt]

`dossier@0.2` : n'en prendre que 20 % (par parties entières) ; `dossier:policy` : seulement pour la
politique (sa valeur n'est pas apprise) ; `dossier:value` : seulement pour la valeur.

Cibles : la politique (visites de la recherche à la racine), la victoire (part de chaque joueur à la
fin) et, en appoint, les pièces gagnées d'ici la fin ; la tête « coup réel », le coup joué ensuite
dans la partie (fichiers *.v5.bin). Les positions sont vues du joueur qui agit. Les coups observés
d'autres joueurs que l'IA qui apprend (tables variées) n'apprennent que la tête « coup réel ».
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import GLOBAL_V1, GLOBAL_V2, ILLEGAL, OBSERVED, in_validation, load
from model import GAIN_SCALE, Net, export, unpack_adj, widen


def batches(recs, idx, size, device, temp, value_mask, policy_mask, inputs=None):
    """`inputs` : les entrées de la partie gardées (les suivantes à zéro : témoin d'un encodage)."""
    for start in range(0, len(idx), size):
        sel = np.sort(idx[start : start + size])
        recs_sel = recs[sel]
        # champs copiés à part : un enregistrement (5 866 octets) n'aligne pas ses nombres sur 4 octets
        r = {name: np.ascontiguousarray(recs_sel[name]) for name in recs_sel.dtype.names}
        visits = torch.from_numpy(r["visits"].astype(np.int32)).to(device)
        legal = visits != ILLEGAL
        visits = torch.where(legal, visits, 0).float()
        if temp != 1.0:
            # cibles aiguisées : visites^(1/T), le coup le plus visité pèse davantage
            visits = (visits / visits.amax(1, keepdim=True).clamp(min=1)) ** (1.0 / temp)
        glob = torch.from_numpy(r["glob"])
        if inputs:
            glob[:, inputs:] = 0
        observed = torch.from_numpy((r["flags"] & OBSERVED) != 0).to(device)
        played = torch.from_numpy(r["played"].astype(np.int64)).to(device)
        known = played != ILLEGAL
        played = torch.where(known, played, 0)
        yield dict(
            node=torch.from_numpy(r["node"]).to(device),
            glob=glob.to(device),
            adj=unpack_adj(torch.from_numpy(r["adj"].astype(np.int64)).to(device)),
            regions=torch.from_numpy(r["regions"].astype(np.int64)).to(device),
            players=torch.from_numpy(r["players"].astype(np.int64)).to(device),
            legal=legal,
            policy=visits / visits.sum(1, keepdim=True).clamp(min=1),
            win=torch.from_numpy(r["win"]).to(device),
            gain=torch.from_numpy(r["gain"]).to(device) / GAIN_SCALE,
            qgain=torch.from_numpy(r["qgain"]).to(device) / GAIN_SCALE,
            has_value=torch.from_numpy(value_mask[sel]).to(device) & ~observed,
            has_policy=legal.any(1) & (visits.sum(1) > 0) & torch.from_numpy(policy_mask[sel]).to(device) & ~observed,
            played=played,
            has_played=known & legal.gather(1, played[:, None])[:, 0],
        )


def losses(net, b, q_weight=0.0):
    """Pertes et mesures : politique, valeur, pièces, précision de la politique, brier, puis tête
    « coup réel » (perte et précision)."""
    logits, value, beh = net(b["node"], b["glob"], b["adj"], b["regions"])
    logits = logits.masked_fill(~b["legal"], -1e9)
    hp, hv = b["has_policy"].float(), b["has_value"].float()
    pol = (-(b["policy"] * F.log_softmax(logits, 1)).sum(1) * hp).sum() / hp.sum().clamp(min=1)
    present = torch.arange(5, device=value.device)[None] < b["players"][:, None]
    vlog = value[:, :5].masked_fill(~present, -1e9)
    val = (-(b["win"] * F.log_softmax(vlog, 1)).sum(1) * hv).sum() / hv.sum().clamp(min=1)
    # pièces : le résultat de la partie, mêlé à ce qu'attendait la recherche quand elle le sait
    target = b["gain"]
    if q_weight > 0:
        known = ~torch.isnan(b["qgain"])
        target = torch.where(known, (1 - q_weight) * target + q_weight * torch.nan_to_num(b["qgain"]), target)
    gain = ((((value[:, 5:] - target) ** 2) * present).sum(1) * hv).sum() / hv.sum().clamp(min=1)
    # précision : le coup le plus visité est-il le préféré du réseau ?
    acc = ((logits.argmax(1) == b["policy"].argmax(1)).float() * hp).sum() / hp.sum().clamp(min=1)
    # la victoire prédite, comparée à la cible (erreur quadratique, « Brier »)
    brier = (((torch.softmax(vlog, 1) - b["win"]) ** 2).sum(1) * hv).sum() / hv.sum().clamp(min=1)
    hb = b["has_played"].float()
    if beh is None:
        zero = logits.new_zeros(())
        return pol, val, gain, acc, brier, zero, zero
    beh = beh.masked_fill(~b["legal"], -1e9)
    bce = (F.cross_entropy(beh, b["played"], reduction="none") * hb).sum() / hb.sum().clamp(min=1)
    bacc = ((beh.argmax(1) == b["played"]).float() * hb).sum() / hb.sum().clamp(min=1)
    return pol, val, gain, acc, brier, bce, bacc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--steps", type=int, help="nombre de mises à jour (remplace --epochs) : à comparer entre essais")
    ap.add_argument("--threads", type=int, help="cœurs de PyTorch (plusieurs entraînements à la fois)")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--gain-weight", type=float, default=0.25)
    ap.add_argument("--beh-weight", type=float, default=1.0, help="poids de la tête « coup réel »")
    ap.add_argument("--val-share", type=float, default=0.05)
    ap.add_argument("--q-weight", type=float, default=0.5, help="part des pièces attendues par la recherche")
    ap.add_argument("--lazy", action="store_true", help="lire les données à la demande (plus grosses que la mémoire)")
    ap.add_argument("--policy-temp", type=float, default=1.0, help="cible de politique : visites^(1/T)")
    ap.add_argument("--seed", type=int, default=0, help="tirage des parties gardées (`@part`) et ordre des lots")
    ap.add_argument(
        "--old-inputs", action="store_true", help="entrées des versions 2 et 3 mises à zéro (témoin de l'encodage)"
    )
    ap.add_argument(
        "--no-history", action="store_true", help="entrées de la version 3 (histoire) à zéro (témoin de l'histoire)"
    )
    args = ap.parse_args()
    inputs = GLOBAL_V1 if args.old_inputs else GLOBAL_V2 if args.no_history else None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.threads:
        torch.set_num_threads(args.threads)
    recs, keys, value_mask, policy_mask = load(args.data, seed=args.seed, with_value=True, lazy=args.lazy)
    rng = np.random.default_rng(args.seed)
    # validation sur des parties entières, choisies par une règle fixe (`in_validation`) : aucune
    # position d'une partie vue à l'entraînement, ni par cette génération ni par les suivantes
    is_val = in_validation(keys, args.val_share)
    val_idx, train_idx = np.flatnonzero(is_val), np.flatnonzero(~is_val)
    print(
        f"{len(recs)} positions de {len(np.unique(keys))} parties "
        f"({len(val_idx)} positions de {len(np.unique(keys[is_val]))} parties pour la validation)"
    )

    net = Net(args.hidden, args.layers)
    if args.init:
        state = torch.load(args.init, map_location="cpu")
        net = Net(state["hidden"], state["layers"])
        net.load_state_dict(widen(state["weights"]))
    net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    per_epoch = math.ceil(len(train_idx) / args.batch)
    steps = args.steps or args.epochs * per_epoch
    epochs = math.ceil(steps / per_epoch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    done = 0

    for epoch in range(epochs):
        net.train()
        start = time.time()
        sums = np.zeros(7)
        count = 0
        train = rng.permutation(train_idx)
        for b in batches(recs, train, args.batch, device, args.policy_temp, value_mask, policy_mask, inputs):
            if done == steps:
                break
            done += 1
            pol, val, gain, acc, brier, bce, bacc = losses(net, b, args.q_weight)
            loss = pol + val + args.gain_weight * gain + args.beh_weight * bce
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
            for b in batches(recs, val_idx, 2048, device, args.policy_temp, value_mask, policy_mask, inputs):
                n = [b[k].sum().item() for k in ("has_policy", "has_value", "has_played")]
                w = np.array(n)[[0, 1, 1, 0, 1, 2, 2]]
                vs += [x.item() for x in losses(net, b, args.q_weight)] * w
                vc += w
        t, v = sums / max(count, 1), vs / np.maximum(vc, 1)
        print(
            f"époque {epoch + 1}/{epochs} ({time.time() - start:.0f} s, {done}/{steps} mises à jour) | entraînement : politique {t[0]:.3f} "
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
