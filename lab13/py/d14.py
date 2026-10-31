"""La valeur est-elle le levier ? (lab13/lab.md, D14 ; aussi les mesures de valeur de D15).

    python d14.py --data ../data/judge --nets ../nets/c130.bin [../runs/d15-….bin …]

Sur toutes les positions de `--data` (parties neuves jouées sans bruit par `selfplay --revalue`) :
l'entropie croisée au résultat et le Brier de la valeur de chaque réseau (le premier est la
référence), de la valeur de la recherche à la racine (`value`, enregistrée par le selfplay), de la
valeur par fin simulée (`roll_win`) et de la moyenne à parts égales de la référence et de la fin
simulée ; l'erreur quadratique des pièces à gagner d'ici la fin (têtes des réseaux, fin simulée) au
gain réel, en pièces². Par type de décision (`d9.kinds`) et par nombre de joueurs. Les différences à
la référence sont appariées ; « ± » : la demi-largeur de l'intervalle à 95 % par rééchantillonnage
des parties entières (`--boot` tirages).
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from d9 import TYPES, kinds
from data import load_packed
from model import GAIN_SCALE, load as load_net
from train import batches

EPS = 1e-6


def normalized(p, present):
    """Des parts de victoire propres : positives, nulles pour les joueurs absents, de somme 1."""
    p = p.clamp(min=0) * present
    return p / p.sum(1, keepdim=True).clamp(min=EPS)


def cross_entropy(z, p):
    return -(z * torch.log(p.clamp(min=EPS))).sum(1)


def brier(z, p):
    return ((p - z) ** 2).sum(1)


def per_position(nets, recs, idx, device):
    """Par position : entropie croisée, Brier et erreur des pièces de chaque estimateur (dictionnaires
    nom → tableau), l'écart de la part de victoire du joueur qui agit entre fin simulée et référence,
    le type et le nombre de joueurs."""
    names = [n for n, _ in nets] + ["recherche", "fin simulée", "moyenne"]
    ce = {n: [] for n in names}
    br = {n: [] for n in names}
    gerr = {n: [] for n in [n for n, _ in nets] + ["fin simulée"]}
    shift, kind, players = [], [], []
    for _, net in nets:
        net.eval()
    with torch.no_grad():
        for b in batches(recs, idx, 2048, device):
            present = (torch.arange(5, device=device)[None] < b["players"][:, None]).float()
            z = b["win"]
            gain = b["gain"] * GAIN_SCALE
            ref = None
            for name, net in nets:
                _, value, _ = net(b["node"], b["glob"], b["adj"], b["regions"])
                p = torch.softmax(value[:, :5].masked_fill(present == 0, -1e9), 1)
                ref = p if ref is None else ref
                ce[name].append(cross_entropy(z, p).cpu().numpy())
                br[name].append(brier(z, p).cpu().numpy())
                gerr[name].append((((value[:, 5:] * GAIN_SCALE - gain) ** 2) * present).sum(1).cpu().numpy())
            search = normalized(b["value"], present)
            roll = normalized(b["roll_win"], present)
            mean = 0.5 * (ref + roll)
            for name, p in (("recherche", search), ("fin simulée", roll), ("moyenne", mean)):
                ce[name].append(cross_entropy(z, p).cpu().numpy())
                br[name].append(brier(z, p).cpu().numpy())
            gerr["fin simulée"].append((((b["roll_gain"] * GAIN_SCALE - gain) ** 2) * present).sum(1).cpu().numpy())
            shift.append((roll[:, 0] - ref[:, 0]).cpu().numpy())
            kind.append(kinds(b["legal"].cpu().numpy(), b["glob"].cpu().numpy()))
            players.append(b["players"].cpu().numpy())
    cat = lambda d: {k: np.concatenate(v) for k, v in d.items()}
    return cat(ce), cat(br), cat(gerr), np.concatenate(shift), np.concatenate(kind), np.concatenate(players)


class Boot:
    """Rééchantillonnage des parties entières (mêmes tirages pour toutes les différences : appariées)."""

    def __init__(self, keys, draws, seed):
        _, self.game = np.unique(keys, return_inverse=True)
        games = self.game.max() + 1
        self.draws = np.random.default_rng(seed).integers(0, games, (draws, games))
        self.games = games

    def interval(self, x, mask):
        """La moyenne de `x` sur `mask`, et la demi-largeur de son intervalle à 95 %."""
        sums = np.bincount(self.game[mask], weights=x[mask], minlength=self.games)
        counts = np.bincount(self.game[mask], minlength=self.games).astype(np.float64)
        means = sums[self.draws].sum(1) / np.maximum(counts[self.draws].sum(1), 1)
        lo, hi = np.percentile(means, [2.5, 97.5])
        return x[mask].mean(), (hi - lo) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="parties jouées avec `selfplay --revalue`")
    ap.add_argument("--nets", nargs="+", required=True, help="réseaux (.bin), la référence d'abord")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=14)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    recs, keys = load_packed([args.data])
    assert recs.roll_win is not None, f"{args.data} : pas de valeur par fin simulée (selfplay --revalue)"
    idx = np.arange(len(recs))
    nets = [(p.split("/")[-1], load_net(p).to(device)) for p in args.nets]
    ce, br, gerr, shift, kind, players = per_position(nets, recs, idx, device)
    ref = nets[0][0]
    boot = Boot(keys, args.boot, args.seed)
    print(f"{len(idx)} positions de {boot.games} parties ({args.data}) ; référence {ref} ; ± : intervalle à 95 %")

    def table(title, groups):
        print(f"\n## {title}\n")
        others = [n for n in ce if n != ref]
        head = (
            "| Groupe | Positions | CE " + ref + " | " + " | ".join(f"CE {n} (Δ)" for n in others)
            + f" | Brier {ref} / fin simulée | victoire du joueur : fin simulée − {ref} | pièces² "
            + " / ".join(gerr) + " |"
        )
        print(head)
        print("|" + " --- |" * (head.count("|") - 1))
        for name, m in groups:
            if not m.any():
                continue
            cells = [name, f"{m.mean():.1%}", f"{ce[ref][m].mean():.4f}"]
            for n in others:
                d, half = boot.interval(ce[n] - ce[ref], m)
                cells.append(f"{ce[n][m].mean():.4f} ({d:+.4f} ± {half:.4f})")
            cells.append(f"{br[ref][m].mean():.4f} / {br['fin simulée'][m].mean():.4f}")
            s, half = boot.interval(shift, m)
            cells.append(f"{s:+.4f} ± {half:.4f} (|·| {np.abs(shift[m]).mean():.4f})")
            cells.append(" / ".join(f"{g[m].mean():.2f}" for g in gerr.values()))
            print("| " + " | ".join(cells) + " |")

    table("Toutes les positions", [("tous", np.ones(len(idx), bool))])
    table("Par type de décision", [(t, kind == k) for k, t in enumerate(TYPES)])
    table("Par nombre de joueurs", [(f"{n} joueurs", players == n) for n in range(2, 6)])


if __name__ == "__main__":
    main()
