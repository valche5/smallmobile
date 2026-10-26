"""L'apprentissage du réseau de lab8 sur les parties d'imitation (`lab8 record`).

Pertes : la politique de la recherche à chaque pas du décodeur (entropie croisée, forçage par
l'enseignant), la victoire finale et celle du prochain tour (entropies croisées), les pièces gagnées
d'ici là (erreur quadratique), les propriétaires au prochain tour.

La validation compare au réseau du joueur imité (c130), sur les mêmes plans : sa politique a priori et
sa valeur au début de chaque plan sont enregistrées avec les parties.

    python train.py --data D1,D2 --out ../nets/x.bin [--init y.bin | --size 256,6,8,2] --steps N
    python train.py --data D --size 256,6,8,2 --bench 50      débit seul
"""

import argparse
import math
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

from data import Data, to_device
from model import GAIN_SCALE, NEG, OWNERS, Net, export, load

LR = 3e-4
WEIGHT_DECAY = 0.01
BATCH = 256
WARMUP = 500
GAIN_WEIGHT = 0.25
OWN_WEIGHT = 0.5
# plans de validation mesurés (les premiers des parties mises de côté)
VAL_PLANS = 20000
WORKERS = 4
AHEAD = 8
PHASES = ("choix", "conquête", "redéploiement", "défense", "fin")


def value_losses(v, win, gain, present):
    vlog = v[:, :5].float().masked_fill(~present, NEG)
    ce = -(win * F.log_softmax(vlog, 1)).sum(1)
    g = (((v[:, 5:].float() - gain / GAIN_SCALE) ** 2) * present).sum(1)
    return ce, g


def losses(net, t):
    """Pertes et mesures par plan (pol, acc : moyennes sur ses pas)."""
    heads, own, logits = net(t["node"], t["glob"], t["dist"], t["regions"], t["prev"], t["dyn"])
    ok = t["step_ok"]
    logits = logits.float().masked_fill(~t["legal"], NEG)
    logp = F.log_softmax(logits, -1)
    steps = ok.sum(1).clamp(min=1)
    pol = (-(t["policy"] * logp).sum(-1) * ok).sum(1) / steps
    acc = ((logits.argmax(-1) == t["key"]) & ok).sum(1) / steps
    present = torch.arange(5, device=heads.device)[None] < t["players"][:, None]
    val, gain = value_losses(heads[:, :10], t["win"], t["gain"], present)
    nxt, ngain = value_losses(heads[:, 10:], t["next_win"], t["next_gain"], present)
    owners = F.cross_entropy(own.float().reshape(-1, OWNERS), t["owner"].reshape(-1), ignore_index=255, reduction="none")
    owners = owners.view(len(steps), -1).sum(1) / t["regions"]
    return {"pol": pol, "acc": acc, "val": val, "gain": gain, "next": nxt, "next_gain": ngain, "own": owners}


def baseline(t):
    """Les mêmes mesures pour le réseau de c130 seul (a priori, valeur au début du plan)."""
    ok = t["step_ok"]
    prior = t["prior"].clamp(min=1e-5).masked_fill(~t["legal"], 1e-30)
    logp = torch.log(prior / prior.sum(-1, keepdim=True))
    steps = ok.sum(1).clamp(min=1)
    pol = (-(t["policy"] * logp).sum(-1) * ok).sum(1) / steps
    acc = ((t["prior"].masked_fill(~t["legal"], -1).argmax(-1) == t["key"]) & ok).sum(1) / steps
    present = torch.arange(5, device=ok.device)[None] < t["players"][:, None]
    pw = t["prior_win"].clamp(min=1e-6).masked_fill(~present, 1.0)
    val = -(t["win"] * torch.log(pw)).sum(1)
    return {"pol": pol, "acc": acc, "val": val}


def total(m):
    return (m["pol"] + m["val"] + m["next"] + GAIN_WEIGHT * (m["gain"] + m["next_gain"]) + OWN_WEIGHT * m["own"]).mean()


def batches(data, idx, size, rng=None):
    """Lots numpy préparés d'avance par plusieurs fils."""
    order = idx if rng is None else rng.permutation(idx)
    chunks = [order[i : i + size] for i in range(0, len(order), size)]
    with ThreadPoolExecutor(WORKERS) as pool:
        pending = [pool.submit(data.take, c) for c in chunks[:AHEAD]]
        for i in range(len(chunks)):
            b = pending.pop(0).result()
            if i + AHEAD < len(chunks):
                pending.append(pool.submit(data.take, chunks[i + AHEAD]))
            yield b


def validate(net, data, idx, device):
    """Moyennes par plan sur `idx` : le réseau, c130 ; et la précision par phase."""
    net.eval()
    sums, base, n = {}, {}, 0
    by_phase = np.zeros((len(PHASES), 2))
    with torch.no_grad():
        for b in batches(data, idx, 512):
            t = to_device(b, device)
            with torch.autocast(device, dtype=torch.bfloat16):
                m = losses(net, t)
            c = baseline(t)
            for k, v in m.items():
                sums[k] = sums.get(k, 0.0) + v.float().sum().item()
            for k, v in c.items():
                base[k] = base.get(k, 0.0) + v.float().sum().item()
            ph = t["phase"].cpu().numpy()
            for p in range(len(PHASES)):
                sel = ph == p
                by_phase[p] += [m["acc"].float().cpu().numpy()[sel].sum(), sel.sum()]
            n += len(b["players"])
    net.train()
    return {k: v / n for k, v in sums.items()}, {k: v / n for k, v in base.items()}, by_phase


def show(m):
    return " ".join(f"{k} {v:.1%}" if k == "acc" else f"{k} {v:.4f}" for k, v in m.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dossiers de `lab8 record`, séparés par des virgules")
    ap.add_argument("--out")
    ap.add_argument("--init", help="réseau de départ (.bin)")
    ap.add_argument("--size", default="256,6,8,2", help="réseau neuf : largeur, couches, têtes, couches du décodeur")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--bench", type=int, help="mesurer le débit sur ce nombre de lots, puis s'arrêter")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    start = time.time()
    data = Data(a.data.split(","))
    train_idx = np.flatnonzero(~data.val)
    val_idx = np.flatnonzero(data.val)[:VAL_PLANS]
    print(f"{len(data)} plans ({len(train_idx)} pour l'apprentissage, {len(val_idx)} mesurés) ; "
          f"chargement {time.time() - start:.0f} s ; {device}", flush=True)
    net = (load(a.init) if a.init else Net(*map(int, a.size.split(",")))).to(device)
    print(f"{sum(p.numel() for p in net.parameters()):,} paramètres", flush=True)
    decay = [p for n, p in net.named_parameters() if p.dim() >= 2 and "rel" not in n]
    other = [p for n, p in net.named_parameters() if not (p.dim() >= 2 and "rel" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": WEIGHT_DECAY}, {"params": other, "weight_decay": 0.0}], lr=a.lr)
    steps = a.bench or a.steps
    lr = lambda i: min(1.0, (i + 1) / WARMUP) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, i / steps))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr)
    order = rng.choice(train_idx, size=steps * a.batch, replace=len(train_idx) < steps * a.batch)
    net.train()
    sums, count, seen, t0 = {}, 0, 0, time.time()
    for i, b in enumerate(batches(data, order, a.batch)):
        if a.bench and i == 5:
            torch.cuda.synchronize() if device == "cuda" else None
            t0, seen = time.time(), 0
        t = to_device(b, device)
        with torch.autocast(device, dtype=torch.bfloat16):
            m = losses(net, t)
            loss = total(m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        for k, v in m.items():
            sums[k] = sums.get(k, 0.0) + v.detach().float().mean()
        count += 1
        seen += len(b["players"])
        if (i + 1) % 200 == 0 or i + 1 == steps:
            secs = time.time() - t0
            avg = {k: (v / count).item() for k, v in sums.items()}
            sums, count = {}, 0
            print(f"lot {i + 1}/{steps} : {show(avg)} ; {seen / secs:.0f} plans/s", flush=True)
        if not a.bench and ((i + 1) % a.val_every == 0 or i + 1 == steps):
            ours, c130, by_phase = validate(net, data, val_idx, device)
            print(f"  validation : {show(ours)}")
            print(f"  c130 seul  : {show(c130)}")
            print("  précision par phase : " + ", ".join(
                f"{PHASES[p]} {by_phase[p, 0] / by_phase[p, 1]:.1%}" for p in range(len(PHASES)) if by_phase[p, 1]), flush=True)
            if a.out:
                export(net, a.out)
    if a.bench:
        torch.cuda.synchronize() if device == "cuda" else None
        print(f"débit : {seen / (time.time() - t0):.0f} plans/s ({a.batch} par lot)")


if __name__ == "__main__":
    main()
