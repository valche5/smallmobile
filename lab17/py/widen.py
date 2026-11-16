"""c130 (lab13 : 77 entrées par région, 820 de la partie, 9 sortes de coups par région) passé à lab17
(lab17/lab.md) : les entrées des extensions, ajoutées après celles de lab13, reçoivent des poids nuls ;
les sorties des coups neufs copient celles d'un coup proche de c130 (la conquête payée et la Catapulte
celles de la conquête, le Chaudron celles du campement, la reprise des Intrépides la fin des
conquêtes). Au jeu de base, le réseau calcule exactement ce que calcule c130.

    python widen.py ../nets/c130.bin ../nets/c130w.bin
"""

import sys

import numpy as np
import torch

from data import GLOBAL, NODE
from model import Net, export, scales

OLD_NODE, OLD_GLOBAL, OLD_KINDS = 77, 820, 9
# sortes de coups de lab17/brain/src/moves.rs (`KINDS`) : les neuves copient celle-ci
COPY_KIND = {9: 0, 10: 0, 11: 7}  # Hire ← Conquer, Catapult ← Conquer, Cauldron ← Camp
# clés globales : REGROUP (17) ← END_CONQUEST (8)
COPY_KEY = {17: 8}


def read(path):
    import struct

    tensors = {}
    with open(path, "rb") as f:
        assert f.read(5) == b"SMNN1", f"{path} : pas un fichier de poids SMNN1"
        (count,) = struct.unpack("<I", f.read(4))
        for _ in range(count):
            (size,) = struct.unpack("<I", f.read(4))
            name = f.read(size).decode()
            (ndim,) = struct.unpack("<I", f.read(4))
            shape = struct.unpack(f"<{ndim}I", f.read(4 * ndim))
            tensors[name] = np.frombuffer(f.read(4 * int(np.prod(shape))), dtype="<f4").reshape(shape).copy()
    return tensors


def widen_cols(w, new):
    return np.concatenate([w, np.zeros((w.shape[0], new - w.shape[1]), np.float32)], axis=1)


def main():
    src, dst = sys.argv[1:3]
    t = read(src)
    node_scale, glob_scale = scales()
    assert t["node_in.weight"].shape[1] == OLD_NODE and t["glob_in.weight"].shape[1] == OLD_GLOBAL
    assert t["node_pol.weight"].shape[0] == OLD_KINDS
    assert np.array_equal(t.pop("node_scale"), node_scale[:OLD_NODE]), "échelles des régions différentes"
    assert np.array_equal(t.pop("global_scale"), glob_scale[:OLD_GLOBAL]), "échelles de la partie différentes"
    t["node_in.weight"] = widen_cols(t["node_in.weight"], NODE)
    t["glob_in.weight"] = widen_cols(t["glob_in.weight"], GLOBAL)
    w, b = t["node_pol.weight"], t["node_pol.bias"]
    t["node_pol.weight"] = np.concatenate([w, np.stack([w[COPY_KIND[k]] for k in sorted(COPY_KIND)])])
    t["node_pol.bias"] = np.concatenate([b, np.array([b[COPY_KIND[k]] for k in sorted(COPY_KIND)], np.float32)])
    for key, frm in COPY_KEY.items():
        t["glob_pol2.weight"][key] = t["glob_pol2.weight"][frm]
        t["glob_pol2.bias"][key] = t["glob_pol2.bias"][frm]
    h = t["node_in.weight"].shape[0]
    layers = sum(1 for k in t if k.endswith(".n1.weight"))
    net = Net(h, layers)
    if "own.weight" not in t:
        t["own.weight"] = net.own.weight.detach().numpy()
        t["own.bias"] = net.own.bias.detach().numpy()
    net.load_state_dict({k: torch.from_numpy(v) for k, v in t.items()})
    export(net, dst)
    print(f"{dst} : {h}×{layers}, entrées {OLD_NODE} → {NODE} et {OLD_GLOBAL} → {GLOBAL}, sortes {OLD_KINDS} → 12")


if __name__ == "__main__":
    main()
