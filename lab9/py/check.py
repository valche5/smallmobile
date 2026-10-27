"""Les sorties de PyTorch sur les plans de `lab9 fixture` (fixture.in), pour les comparer au réseau Rust
(`lab9 check-net`, et le test lab9/tests/net.rs) : fixture.out, par plan les 20 têtes puis les 608
logits de chaque pas (f32).

    python check.py DIR --net poids.bin
    python check.py DIR --random 32,RTR,4,1 [--seed 1]    réseau au hasard écrit dans DIR/net.bin
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from model import DYN, GLOBAL, NODE, REGIONS, export, load, sized


def plans(raw):
    """Les plans de fixture.in : (node, glob, dist, regions, prev, dyn)."""
    at = 0
    while at < len(raw):
        regions = raw[at + 1]
        at += 2
        node = raw[at : at + REGIONS * NODE].reshape(REGIONS, NODE)
        at += REGIONS * NODE
        glob = raw[at : at + GLOBAL]
        at += GLOBAL
        dist = raw[at : at + REGIONS * REGIONS].reshape(REGIONS, REGIONS)
        at += REGIONS * REGIONS
        steps = raw[at]
        at += 1
        prev, dyn = [], []
        for _ in range(steps):
            prev.append(int(raw[at]) | int(raw[at + 1]) << 8)
            dyn.append(raw[at + 2 : at + 2 + DYN])
            at += 2 + DYN
        yield node, glob, dist, int(regions), prev, np.array(dyn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--net")
    ap.add_argument("--random", help="largeur,disposition (ou nombre de blocs T),têtes,couches du décodeur")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    d = Path(a.dir)
    torch.manual_seed(a.seed)
    if a.random:
        net = sized(a.random)
        # tout au hasard, sorties comprises : chaque poids compte dans la comparaison
        with torch.no_grad():
            for name, p in net.named_parameters():
                norm = name.endswith(".weight") and "ln" in name.split(".")[-2]
                p.copy_(torch.randn_like(p) * 0.2 + (1.0 if norm else 0.0))
        export(net, d / "net.bin")
    else:
        net = load(a.net)
    net.eval()
    raw = np.fromfile(d / "fixture.in", np.uint8)
    out = []
    count = 0
    with torch.no_grad():
        for node, glob, dist, regions, prev, dyn in plans(raw):
            t = lambda x, dt: torch.as_tensor(np.asarray(x), dtype=dt)[None]
            heads, _, logits = net(
                t(node, torch.uint8), t(glob, torch.uint8), t(dist, torch.uint8), torch.tensor([regions]),
                t(prev, torch.long), t(dyn, torch.uint8),
            )
            out.append(heads[0].numpy().astype("<f4"))
            out.append(logits[0].numpy().astype("<f4").ravel())
            count += 1
    np.concatenate(out).tofile(d / "fixture.out")
    print(f"{count} plans")


if __name__ == "__main__":
    main()
