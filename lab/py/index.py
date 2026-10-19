"""Écrit à côté de chaque fichier de positions son fichier `.games` (voir `data.games`), une fois
pour toutes : une lecture complète des fichiers anciens, qui ne l'ont pas.

    python index.py ../data/label1 ../data/gen7 ...
"""

import sys
from pathlib import Path

from data import dtype_of, games

for d in sys.argv[1:]:
    files = sorted(Path(d).rglob("*.bin"))
    total = 0
    for f in files:
        dtype = dtype_of(f)
        total += len(games(f, dtype, f.stat().st_size // dtype.itemsize))
    print(f"{d} : {len(files)} fichiers, {total} parties", flush=True)
