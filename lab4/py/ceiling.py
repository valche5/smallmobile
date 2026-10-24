"""Le plafond de la valeur, depuis la sortie de `lab4 ceiling` : par nombre de joueurs et manches
restantes, l'entropie croisée de la victoire pour une prédiction uniforme, pour le réseau (sa valeur
à la position), et pour la meilleure valeur possible face à ces joueurs : la répartition des
vainqueurs des autres suites de la même position (chaque suite jugée sans elle, a priori de
Dirichlet `ALPHA` ; un peu pessimiste avec peu de suites). « Atteint » : la part de l'écart entre
l'uniforme et le plafond que le réseau a déjà comblée.

    python ceiling.py ../runs/ceiling/c130.tsv
"""
import sys
from collections import defaultdict

import numpy as np

ALPHA = 0.5

positions, suites = {}, defaultdict(list)
for line in open(sys.argv[1]):
    f = line.rstrip("\n").split("\t")
    if f[0] == "position":
        i, n, rnd, last = int(f[1]), int(f[2]), int(f[3]), int(f[4])
        positions[i] = (n, last - rnd + 1, np.array([float(x) for x in f[6 : 6 + n]]))
    elif f[0] == "suite":
        suites[int(f[1])].append([float(x) for x in f[2:]])

groups = defaultdict(lambda: np.zeros(4))
for i, (n, left, net) in positions.items():
    wins = np.array(suites[i])
    k = len(wins)
    total = wins.sum(0)
    stage = "6+" if left >= 6 else "4–5" if left >= 4 else "2–3" if left >= 2 else "1"
    for w in wins:
        rest = (total - w + ALPHA) / (k - 1 + n * ALPHA)
        ceiling = -(w * np.log(rest)).sum()
        model = -(w * np.log(np.clip(net, 1e-6, 1))).sum()
        for key in ((n, stage), (n, "tout")):
            groups[key] += (1, np.log(n), model, ceiling)

print("joueurs  manches restantes  suites  uniforme  réseau  plafond  atteint")
for n in sorted({n for n, _ in groups}):
    for stage in ("6+", "4–5", "2–3", "1", "tout"):
        if (n, stage) not in groups:
            continue
        c, uni, model, ceiling = groups[(n, stage)]
        uni, model, ceiling = uni / c, model / c, ceiling / c
        reached = (uni - model) / (uni - ceiling) if uni > ceiling else float("nan")
        print(f"{n:>7}  {stage:>17}  {int(c):>6}  {uni:8.3f}  {model:6.3f}  {ceiling:7.3f}  {reached:7.0%}")
