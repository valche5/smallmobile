"""Où la valeur se trompe : sur les parties de validation des cycles donnés, l'entropie croisée de la
victoire prédite par manches restantes et nombre de joueurs, pour une prédiction uniforme, un
repère des pièces actuelles, le réseau, et la recherche qui a joué (`value` de la forme compacte).

    python stages.py --net ../nets/c130.bin --data ../../lab3/data/deep7/c12{6,7,8,9} ../../lab3/data/deep7/c130
"""
import argparse

import numpy as np
import torch

from data import Packed, in_validation, load, sources
from model import load as load_net
from train import VAL_SHARE, batches

ap = argparse.ArgumentParser()
ap.add_argument("--net", required=True)
ap.add_argument("--data", nargs="+", required=True)
args = ap.parse_args()

G_LEFT, G_COINS = 6, [67 + q * 81 + 1 for q in range(5)]
cycles = args.data
recs, keys = load(cycles, keep=lambda k: in_validation(k, VAL_SHARE))
print(len(recs), 'positions de validation', len(np.unique(keys)), 'parties', flush=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = load_net(args.net).to(device).eval()
win = recs['win'].astype(np.float64); players = recs['players'].astype(int)
present = np.arange(5)[None] < players[:, None]
ce = lambda p: -(win * np.log(np.clip(p, 1e-9, 1))).sum(1)
pn = []
with torch.no_grad():
    for b in batches(recs, np.arange(len(recs)), 4096, device):
        _, value = net(b['node'], b['glob'], b['adj'], b['regions'])
        pres = torch.arange(5, device=device)[None] < b['players'][:, None]
        pn.append(torch.softmax(value[:, :5].masked_fill(~pres, -1e9), 1).cpu().numpy())
pn = np.concatenate(pn).astype(np.float64)
# la recherche : `value` est la part de victoire attendue (load() ne la garde pas : relue)
pk = Packed(sources(cycles)); sel = np.flatnonzero(in_validation(pk.keys, VAL_SHARE))
ps = np.clip(pk.value[sel].astype(np.float64), 0, 1); ps = np.where(present, ps, 0); ps /= ps.sum(1, keepdims=True)
coins = recs['glob'][:, G_COINS].astype(np.float64)
def base(t):
    z = np.where(present, coins / t, -1e9); z -= z.max(1, keepdims=True); e = np.exp(z) * present; return e / e.sum(1, keepdims=True)
t = min([1, 2, 3, 5, 8, 12, 20, 30], key=lambda t: ce(base(t)).mean()); pb = base(t)
uni = present / players[:, None]
left = recs['glob'][:, G_LEFT].astype(int)
print(f'repère des pièces : échelle {t}')
print('joueurs  manches restantes  positions  uniforme  pièces  réseau  recherche')
for n in (2, 3, 4, 5):
    for lo, hi, name in ((0, 1, '0–1'), (2, 3, '2–3'), (4, 5, '4–5'), (6, 20, '6+')):
        m = (players == n) & (left >= lo) & (left <= hi)
        if m.sum() < 200: continue
        print(f'{n:>7}  {name:>17}  {m.sum():>9}  {ce(uni)[m].mean():8.3f}  {ce(pb)[m].mean():6.3f}  {ce(pn)[m].mean():6.3f}  {ce(ps)[m].mean():9.3f}')
    m = players == n
    print(f'{n:>7}  {"tout":>17}  {m.sum():>9}  {ce(uni)[m].mean():8.3f}  {ce(pb)[m].mean():6.3f}  {ce(pn)[m].mean():6.3f}  {ce(ps)[m].mean():9.3f}')
