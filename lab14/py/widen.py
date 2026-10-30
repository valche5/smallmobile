"""Un réseau de lab13 (77 entrées par région) passé à lab14 (93) : `node_in` reçoit des poids nuls
pour les entrées de la menace, il calcule donc exactement la même chose (lab14/lab.md, c130x).

    python widen.py ../../lab13/nets/c130.bin ../nets/c130x.bin
"""

import struct
import sys

import numpy as np
import torch

from data import NODE
from model import Net, export, scales

OLD_NODE = 77


def read(path):
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


def main():
    src, dst = sys.argv[1:3]
    t = read(src)
    node_scale, _ = scales()
    assert t["node_in.weight"].shape[1] == OLD_NODE, "pas un réseau à 77 entrées par région"
    assert np.array_equal(t.pop("node_scale"), node_scale[:OLD_NODE]), "échelles de lab13 différentes"
    t.pop("global_scale")
    w = t["node_in.weight"]
    t["node_in.weight"] = np.concatenate([w, np.zeros((w.shape[0], NODE - OLD_NODE), np.float32)], axis=1)
    layers = sum(1 for k in t if k.endswith(".n1.weight"))
    net = Net(w.shape[0], layers)
    net.load_state_dict({k: torch.from_numpy(v) for k, v in t.items()})
    export(net, dst)
    print(f"{dst} : {w.shape[0]}×{layers}, {NODE - OLD_NODE} entrées de poids nuls")


if __name__ == "__main__":
    main()
