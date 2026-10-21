"""Temporaire (21 octobre) : passe les données et les réseaux de gen12, faits avant l'encodage unique,
au format d'aujourd'hui ; à supprimer une fois fait (`lab/JOURNAL.md`).

    python migrate.py data ../data/gen12-mixed ../data/gen12-mixed-v6   positions *.v5.bin → *.v6.bin
    python migrate.py net ancien.pt nouveau.bin                          réseau à 806 ou 926 entrées

Les positions v5 avaient 926 entrées de la partie (700 de base, 106 de la version 2, l'histoire),
puis les pièces et la victoire attendues par la recherche : on garde les 700 de base et l'histoire.
Un réseau ne passe que si ses poids des entrées de la version 2 sont nuls (mêmes sorties).
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import torch

from data import GLOBAL, KEYS, NODE, RECORD, REGIONS, SUFFIX
from model import Net, export

BASE, V2, V3 = 700, 806, 926
V5 = np.dtype(
    [
        ("players", "u1"),
        ("regions", "u1"),
        ("full", "u1"),
        ("flags", "u1"),
        ("win", "<f4", 5),
        ("gain", "<f4", 5),
        ("visits", "<u2", KEYS),
        ("adj", "<u8", REGIONS),
        ("node", "u1", (REGIONS, NODE)),
        ("glob", "u1", V3),
        ("qgain", "<f4", 5),
        ("qwin", "<f4", 5),
        ("played", "<u2"),
    ]
)
assert GLOBAL == BASE + V3 - V2 and V5.itemsize == 5988


def data(src, dst):
    files = sorted(Path(src).rglob("*.v5.bin"))
    for f in files:
        out = Path(dst) / f.relative_to(src).with_name(f.name.replace(".v5.bin", SUFFIX))
        out.parent.mkdir(parents=True, exist_ok=True)
        count = f.stat().st_size // V5.itemsize
        assert count * V5.itemsize == f.stat().st_size, f
        old = np.memmap(f, dtype=V5, mode="r", shape=(count,)) if count else np.zeros(0, V5)
        new = np.memmap(out, dtype=RECORD, mode="w+", shape=(count,)) if count else np.zeros(0, RECORD)
        for a in range(0, count, 1 << 16):
            o, n = old[a : a + (1 << 16)], new[a : a + (1 << 16)]
            for name in RECORD.names:
                if name == "glob":
                    n["glob"][:, :BASE] = o["glob"][:, :BASE]
                    n["glob"][:, BASE:] = o["glob"][:, V2:]
                else:
                    n[name] = o[name]
        if count:
            new.flush()
        del old, new
        for side in (".games", ".moves"):
            if f.with_name(f.name + side).exists():
                shutil.copy(f.with_name(f.name + side), out.with_name(out.name + side))
    print(f"{src} : {len(files)} fichiers → {dst}", flush=True)


def net(src, dst):
    state = torch.load(src, map_location="cpu")
    w = dict(state["weights"])
    g = w["glob_in.weight"]
    assert g.shape[1] in (V2, V3), g.shape
    assert torch.count_nonzero(g[:, BASE:V2]) == 0, "poids des entrées de la version 2 non nuls"
    history = g[:, V2:] if g.shape[1] == V3 else g.new_zeros(g.shape[0], V3 - V2)
    w["glob_in.weight"] = torch.cat([g[:, :BASE], history], 1)
    model = Net(state["hidden"], state["layers"], behaviour="beh_node.weight" in w)
    model.load_state_dict(w)
    dst = Path(dst)
    export(model, dst)
    torch.save({"hidden": model.hidden, "layers": len(model.layers), "weights": model.state_dict()}, dst.with_suffix(".pt"))
    print(f"{src} → {dst}")


if __name__ == "__main__":
    {"data": data, "net": net}[sys.argv[1]](*sys.argv[2:])
